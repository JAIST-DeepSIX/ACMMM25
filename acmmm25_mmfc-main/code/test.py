import wandb
from train_for_classification_2 import make_model
import torch
from sklearn.metrics import classification_report, accuracy_score, ConfusionMatrixDisplay, confusion_matrix
import pandas as pd
import matplotlib.pyplot as plt
import json
import os
def make_data(json_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def get_summary_context(sample):
    """
    Trích xuất context từ sample["summary"] bằng cách lấy phần sau token 'The answer is: \nassistant\n'.
    Nếu không tồn tại token, trả về chuỗi rỗng.
    """
    split_token = "The answer is: \nassistant\n"
    summary = sample.get("summary", "")

    if split_token in summary:
        new_context = summary.split(split_token, 1)[1].strip()
    else:
        new_context = ""
    return new_context
    

def make_message(sample):
    message_template = """ [INSTRUCTION]
    Here are the input:
    Context: [CONTEXT]
    Claim: [CLAIM]
    Question: [QUESTION]
    Choices: 
    [CHOICES]
    The answer is: """
    instruction_prompt = """You are an effective multi-modal fact-checking model. Your task is to verify the factual accuracy of claims by analyzing multimodal inputs. The given inputs include: a claim in textual form, which can be a news headline, a sentence from an article, or a social media post; an additional context in textual form, such as the full text of the news article, a related social media discussion, or other supplementary information. You must determine the factual accuracy of the claim based on all the provided inputs. They need to assign one of four possible labels to the claim: \"True\", \"False\", \"Partially True\", and \"Not Verifiable\" by choosing one of the following letters: A, B, C, D. Answer which must be in the format of a single letter (A, B, C or D)."""
    prototype = {
        "question": "Based on the provided information, please determine whether the claim is factual.",
        "choices": [
            {
                "id": "A",
                "choice": "True"
            },
            {
                "id": "B",
                "choice": "False"
            },
            {
                "id": "C",
                "choice": "Partially True"
            },
            {
                "id": "D",
                "choice": "Not Verifiable"
            }
        ],
    }
    summary_context= get_summary_context(sample=sample)
    choices = []
    for choice in prototype["choices"]:
        choices.append(f"{choice['id']}. {prototype["choices"]}")
    choices = "\n".join(choices)
    
    message = message_template.replace("[INSTRUCTION]", instruction_prompt)\
        .replace("[CONTEXT]", summary_context)\
        .replace("[CLAIM]", sample["claim"])\
        .replace("[QUESTION]", prototype["question"])\
        .replace("[CHOICES]", choices)
    return message
def make_message_for_case(sample, case):
    """
    Đưa ra 2 phương án tương ứng case để bắt mô hình chọn lại đáp án chính xác
    """
    message_template = """ [INSTRUCTION]
    Here are the input:
    Context: [CONTEXT]
    Claim: [CLAIM]
    Question: [QUESTION]
    Choices: 
    [CHOICES] 
    DEFINITION: [DEFINITION]
    The answer is: 
    """
    prototype = {
        "question": "Based on the provided information, please determine whether the claim is factual.",
        "choices": [
            {
                "id": "A",
                "choice": "True"
            },
            {
                "id": "B",
                "choice": "False"
            },
            {
                "id": "C",
                "choice": "Partially True"
            },
            {
                "id": "D",
                "choice": "Not Verifiable"
            }
        ],
    }
    if case == 1:
        definition = """Definition: 
        - The choice is A - True if it is fully supported by the CONTEXT. All critical facts are correct, clearly stated or strongly implied. There are no misleading or incorrect elements.
        - The choice is C - Partially True if it is partially supported by the CONTEXT. Some parts are correct, but others are missing context, oversimplified, or inaccurate. It mixes both correct and incorrect elements."""
        instruction_prompt = """You are an effective multi-modal fact-checking model. Your task is to verify the factual accuracy of claims by analyzing multimodal inputs. The given inputs include: a claim in textual form, which can be a news headline, a sentence from an article, or a social media post; an additional context in textual form, such as the full text of the news article, a related social media discussion, or other supplementary information. You must determine the factual accuracy of the claim based on all the provided inputs. They need to assign one of four possible labels to the claim: \"True\" and \"Partially True\" by choosing one of the following letters: A, C  or number: 1, 3. Answer which must be in the format of a single letter (A or C) or single  (1 or 3)."""
        choices = []
        for choice in prototype["choices"]:
            if choice['id'] in ["A", "C"]:
                choices.append(f"{choice['id']}. {prototype["choices"]}")
        choices = "\n".join(choices)
    elif case == 2:
        definition = """Definition: 
        - The choice is B - False if it is clearly contradicted by the CONTEXT or includes factually incorrect information. It misrepresents the content or introduces inaccurate claims.
        - The choice is D - Not Verifiable if it cannot be confirmed or refuted using only by the CONTEXT. It may be plausible, but the sample does not provide enough explicit evidence to support or deny it."""    
        instruction_prompt = """You are an effective multi-modal fact-checking model. Your task is to verify the factual accuracy of claims by analyzing multimodal inputs. The given inputs include: a claim in textual form, which can be a news headline, a sentence from an article, or a social media post; an additional context in textual form, such as the full text of the news article, a related social media discussion, or other supplementary information. You must determine the factual accuracy of the claim based on all the provided inputs. They need to assign one of four possible labels to the claim: \"False\" and \"Not Verifiable\" by choosing one of the following letters: B, D or number: 2, 4. Answer which must be in the format of a single letter (B or D) or single  (2 or 4)."""
        for choice in prototype["choices"]:
            if choice['id'] in ["B", "D"]:
                choices.append(f"{choice['id']}. {prototype["choices"]}")
        choices = "\n".join(choices)
    summary_context= get_summary_context(sample=sample)

    
    message = message_template.replace("[INSTRUCTION]", instruction_prompt)\
        .replace("[CONTEXT]", summary_context)\
        .replace("[CLAIM]", sample["claim"])\
        .replace("[QUESTION]", prototype["question"])\
        .replace("[DEFINITION]", definition)\
        .replace("[CHOICES]", choices)

    return message

def predict(sample):
    messages = make_message(sample)
    print("Messages: ", messages)
    with torch.no_grad():
        inputs = tokenizer(
            text=messages,
            return_tensors="pt"
        ).to("cuda")
        
        for _ in range(10):
            logits = model(**inputs).logits
            predicted_class_id = logits.argmax().item()
            response = model.config.id2label[predicted_class_id]
            print("Response: ", response)
            if response in ["A", "B", "C", "D"]:
                return response
            
def predictByCase(sample, case=1):
    messages = make_message_for_case(sample, case)
    print("Messages: ", messages)
    with torch.no_grad():
        inputs = tokenizer(
            text=messages,
            return_tensors="pt"
        ).to("cuda")
        for _ in range(10):
            logits = model(**inputs).logits
            predicted_class_id = logits.argmax().item()
            response = model.config.id2label[predicted_class_id]
            print("Response: ", response)
            if case == 1:
                if response in ["A", "C"]:
                    return response
            elif case == 2:
                if response in ["B", "D"]:
                    return response
        return response
            
def full_predict(sample):         
    response1 = predict(sample=sample)
    print("Response 1: ", response1)
    if response1 in ["A", "C"]:
        case = 1
    elif response1 in ["B", "D"]:
        case = 2
    response2 = predictByCase(sample=sample, case=case)
    print("Response 2: ", response2)
    return response1, response2

def draw_confusion_matrix(save_dir, y_true, y_pred):

    # Bước 3: Tạo confusion matrix
    labels = sorted(list(set(y_true) | set(y_pred)))  # đảm bảo lấy tất cả nhãn có thể
    cm1 = confusion_matrix(y_true=y_true, y_pred=y_pred, labels=labels)
    # cm2 = confusion_matrix(y_true=y_true, y_pred=y_pred2, labels=labels)
    # Bước 4: Hiển thị
    disp = ConfusionMatrixDisplay(confusion_matrix=cm1, display_labels=labels)
    disp.plot(xticks_rotation=45, cmap='Blues')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    # Lưu ảnh
    plt.savefig(save_dir + '/confusion_matrix1.png', dpi=300)
    # disp = ConfusionMatrixDisplay(confusion_matrix=cm2, display_labels=labels)
    # disp.plot(xticks_rotation=45, cmap='Blues')
    # plt.title('Confusion Matrix')
    # plt.tight_layout()
    # # Lưu ảnh
    # plt.savefig(save_dir + '/confusion_matrix2.png', dpi=300)
    
def compute_accurate(y_pre, y_true):
    return accuracy_score(y_true=y_true, y_pred=y_pre)
if __name__ == "__main__":
    wandb.login()
    wandb.init(
            # Set the project where this run will be logged
            project="Test 72B 20-5",
            # We pass a run name (otherwise it’ll be randomly assigned, like sunshine-lollypop-10)
            name=f"Test 72B 20-5",
            # Track hyperparameters and run metadata
            config={
                "learning_rate": 0.01,
                "architecture": "CNN",
                "dataset": "CIFAR-100",
                "epochs": 100,
            })
    # This part for training
    # model_id="Qwen/Qwen2.5-14B-Instruct"
    # path_load_model = "/home/s2511001/Study/Competition/zeroshotSTrain/train/gwen25-14B-classification-19-5"
    model_id="Qwen/Qwen2.5-72B-Instruct"
    path_load_model = "/home/s2511001/Study/Competition/zeroshotSTrain/train/gwen25-72B-classification-19-5"   
    
    save_report_dir="/home/s2511001/Study/Competition/zeroshotSTrain/test/run72B-20-5"
    
    id2label = {0: "A", 
               1: "B", 
               2: "C",
               3: "D"}
    label2id = {"A": 0, 
                "B": 1,
                "C": 2,
                "D": 3}
    data = make_data(json_path="/home/s2511001/Study/Competition/zeroshotSTrain/data/test_qwen2.5_vl_summary.json")
    model, processor, tokenizer = make_model(model_id=model_id, 
                                             num_labels=4, 
                                             id2label=id2label, 
                                             label2id=label2id, 
                                             path_load_model=path_load_model)

    Y_pre = []
    # Y_old_pre = []
    Y_true = []
    count = 0
    outputs = []
    for sample in data:
        count = count + 1
        y_true = sample['label']
        Y_true.append(y_true)        
        # response1, response2 = full_predict(sample=sample)
        # Y_old_pre.append(response1)
        # Y_pre.append(response2)
        y_pred = predict(sample=sample)
        Y_pre.append(y_pred)
        # output = {
        #     "response 1" : response1,
        #     "response 2" : response2,
        #     "true" : y_true
        # }
        output = {
            "Pred" : y_pred,
            "True" : y_true
        }
        outputs.append(output)
    file_path = os.path.join(save_report_dir, "output.json")
    # Ghi ra file Excel
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(outputs, f, ensure_ascii=False, indent=2)
    f.close()
    draw_confusion_matrix(save_dir=save_report_dir, y_true=Y_true, y_pred=Y_pre)
    accurate1 = compute_accurate(y_pre=Y_pre, y_true=Y_true)
    print(f"Accurate 1: {accurate1}")
    # accurate2 = compute_accurate(y_pre=Y_old_pre, y_true=Y_true)
    # print(accurate2)
    report1 = classification_report(y_true=Y_true, y_pred=Y_pre, digits=4)
    print(report1)
    # report2 = classification_report(y_true=Y_true, y_pred=Y_old_pre, digits=4)
    # print(report2)
    with open(save_report_dir + "/note.txt", "w", encoding="utf-8") as f:
        f.write(f"Accurate 1: {accurate1}\n")
        f.write(report1)
        # f.write(f"Accurate 2: {accurate2}\n")
        # f.write(report2)        
    