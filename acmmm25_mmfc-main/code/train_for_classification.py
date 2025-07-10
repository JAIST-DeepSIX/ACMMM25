from transformers import pipeline, EarlyStoppingCallback ,TrainingArguments, Trainer, AutoModelForSequenceClassification, DataCollatorWithPadding, BitsAndBytesConfig, AutoProcessor, AutoTokenizer
import torch
from sklearn.metrics import classification_report
import json
import evaluate
from datasets import Dataset, DatasetDict
from peft import LoraConfig, get_peft_model, TaskType
import wandb
import numpy as np

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
def make_model(model_id, path_load_model = None,num_labels=None, id2label=None, label2id=None):
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True, 
        bnb_4bit_use_double_quant=True, 
        bnb_4bit_quant_type="nf4", 
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    
    if num_labels is None:
        if path_load_model is None:
            model = AutoModelForSequenceClassification.from_pretrained(
                model_id,
                device_map="auto", 
                torch_dtype=torch.bfloat16, 
                quantization_config=bnb_config
            )

        else:
            model = AutoModelForSequenceClassification.from_pretrained(
                path_load_model,
                device_map="auto", 
                torch_dtype=torch.bfloat16, 
                quantization_config=bnb_config
            )

 
    else:
        if path_load_model is None:
            model = AutoModelForSequenceClassification.from_pretrained(
                model_id,
                device_map="auto", 
                torch_dtype=torch.bfloat16, 
                quantization_config=bnb_config,
                num_labels=num_labels, id2label=id2label, label2id=label2id
            )

        else:
            model = AutoModelForSequenceClassification.from_pretrained(
                path_load_model,
                device_map="auto", 
                torch_dtype=torch.bfloat16, 
                quantization_config=bnb_config,
                num_labels=num_labels, id2label=id2label, label2id=label2id
            )


    processor = AutoProcessor.from_pretrained(model_id)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    return model, processor, tokenizer

def preprocess_fn(samples):
    return tokenizer(samples["text"], truncation=True)

def make_data(json_path, label2id):

    with open(json_path, 'r', encoding='utf-8') as f:
        ordata = json.load(f)
    data = []
    for sample in ordata:
        data.append({
            "text" : make_message(sample=sample),
            "label" : label2id[sample["label"]]
        })
        
    return data

def make_message(sample):
    message_template = """ [INSTRUCTION]
    Here are the input:
    Context: [CONTEXT]
    Claim: [CLAIM]
    Question: [QUESTION]
    Choices: [CHOICES]
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



def predict_label(sample, model, processor):
    message = make_message(sample=sample)
    text = processor.apply_chat_template(message, tokenize=False, add_generation_prompt=True)
    inputs = processor(
        text=[text],
        padding=True,
        return_tensors="pt"
    )
    inputs = inputs.to("cuda")
    generated_ids = model.generate(**inputs, 
                                max_new_tokens=10, 
                                temperature = 0.5, 
                                do_sample=True)            
    response = processor.decode(generated_ids[0][len(inputs.input_ids[0]):], 
                                skip_special_tokens=True, 
                                clean_up_tokenization_spaces=False)
    return response

def make_report(y_pre, y_true, save_dir):
    report = classification_report(y_true=y_true, y_pred=y_pre, digits=4)
    print(report)
    with open(save_dir + "/note.txt", "w", encoding="utf-8") as f:
        f.write(report)

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return accuracy.compute(predictions=predictions, references=labels)


if __name__ == "__main__":
    wandb.login()
    wandb.init(
            # Set the project where this run will be logged
            project="Train 14B 16-5",
            # We pass a run name (otherwise it’ll be randomly assigned, like sunshine-lollypop-10)
            name=f"Train 14B 16-5",
            # Track hyperparameters and run metadata
            config={
                "learning_rate": 0.01,
                "architecture": "CNN",
                "dataset": "CIFAR-100",
                "epochs": 100,
            })
    # This part for training
    model_id="Qwen/Qwen2.5-72B-Instruct"
    path_load_model = "/home/s2511001/Study/Competition/zeroshotSTrain/train/gwen25-72B-classification-19-5"    
    # save_report_dir="/home/s2511001/Study/Competition/zeroshotSTrain/test/run3"
    
    id2label = {0: "A", 
               1: "B", 
               2: "C",
               3: "D"}
    label2id = {"A": 0, 
                "B": 1,
                "C": 2,
                "D": 3}
    data = {
        "train" : make_data(json_path="/home/s2511001/Study/Competition/zeroshotSTrain/data/train_qwen2.5_vl_summary.json",
                     label2id=label2id),
        "test" : make_data(json_path="/home/s2511001/Study/Competition/zeroshotSTrain/data/dev_qwen2.5_vl_summary.json",
                     label2id=label2id)
    }
    model, processor, tokenizer = make_model(model_id=model_id, 
                                             num_labels=4, 
                                             id2label=id2label, 
                                             label2id=label2id,
                                             path_load_model=path_load_model)
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.pad_token_id
    lora_config = LoraConfig(
        lora_alpha=16,
        lora_dropout=0.1,
        r=2,
        bias="none",
        task_type=TaskType.SEQ_CLS,
        target_modules=["q_proj","v_proj",],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    data = DatasetDict({
        "train": Dataset.from_list(data["train"]),
        "test": Dataset.from_list(data["test"])
    })
    tokenized_data = data.map(preprocess_fn, batched=True)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    accuracy = evaluate.load("accuracy")
    training_args = TrainingArguments(
        output_dir="./train/gwen25-72B-classification-19-5",
        learning_rate=2e-5,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        eval_steps = 10,
        num_train_epochs=20,
        logging_steps=10,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        save_total_limit = 2
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_data["train"],
        eval_dataset=tokenized_data["test"],
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    )
    trainer.train()
    trainer.save_model(training_args.output_dir)
    # This part for evaluating
    # pipe = pipeline("text-classification", model="/home/s2511001/Study/Competition/zeroshotSTrain/train/my_awesome_model")
    # eval_data = make_data(json_path="/home/s2511001/Study/Competition/zeroshotSTrain/data/dev_qwen2.5_vl_summary.json",
    #                  label2id=label2id)
    # y_pres = []
    # y_trues = []
    # for sample in data:
    #     y_pre = predict_label(sample=sample, model=model,processor=processor)
    #     y_true = sample['label']
    #     print(f"Predict: {y_pre} ======= True label: {y_true}")
    #     y_pres.append(y_pre)
    #     y_trues.append(y_true)
    #     break
    # try:
    #     make_report(y_true=y_trues,y_pre=y_pres, save_dir=save_report_dir)
    # except Exception as error:
    #     print("Error: ", error)


