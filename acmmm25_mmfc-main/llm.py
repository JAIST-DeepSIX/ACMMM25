from transformers import AutoTokenizer, AutoModelForCausalLM, AutoProcessor, BitsAndBytesConfig, LlavaForConditionalGeneration, LlavaNextProcessor, LlavaNextForConditionalGeneration, Qwen2_5_VLForConditionalGeneration, AutoModelForSequenceClassification
from PIL import Image
from qwen_vl_utils import process_vision_info
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix
# from llm2vec import LLM2Vec
import numpy as np
import torch
from data import *
from tqdm import tqdm
import re


class ClaimVerificationDataset(torch.utils.data.Dataset):
    def __init__(self, claim_verification_data, images_db):
        self._data = claim_verification_data

        self._encoded = []
        for d in self._data:
            self._encoded.append(self._encode_one_sample(d, images_db))

    def _encode_one_sample(self, sample, images_db):
        claim = sample['claim']
        text_evidence = sample['context']
        image_evidence = images_db.loc[(images_db.id == sample['image_id'])]['image'].values[0]
        label = sample['correct_choice']
        lst_choices = sample['choices']
        question = sample['question']

        encoded_sample = {}
        encoded_sample["claim"] = claim
        encoded_sample["correct_choice"] = label
        encoded_sample['text_evidence'] = text_evidence
        encoded_sample['image_evidence'] = image_evidence
        encoded_sample['choices'] = lst_choices
        encoded_sample['question'] = question

        return encoded_sample
    
    def __len__(self):
        return len(self._encoded)

    def __getitem__(self, idx):
        return self._encoded[idx]

    def to_list(self):
        return self._encoded


class ClaimVerificationDatasetWithSum(torch.utils.data.Dataset):
    def __init__(self, claim_verification_data, images_db):
        self._data = claim_verification_data

        self._encoded = []
        for d in self._data:
            self._encoded.append(self._encode_one_sample(d, images_db))

    def _encode_one_sample(self, sample, images_db):
        claim = sample['claim']
        text_evidence = sample['context']
        image_evidence = images_db.loc[(images_db.id == sample['image_id'])]['image'].values[0]
        label = sample['correct_choice']
        lst_choices = sample['choices']
        question = sample['question']

        encoded_sample = {}
        encoded_sample["claim"] = claim
        encoded_sample["correct_choice"] = label
        encoded_sample['text_evidence'] = text_evidence
        encoded_sample['image_evidence'] = image_evidence
        encoded_sample['choices'] = lst_choices
        encoded_sample['question'] = question
        encoded_sample['summary'] = sample['summary'].split("assistant\n")[-1]

        return encoded_sample
    
    def __len__(self):
        return len(self._encoded)

    def __getitem__(self, idx):
        return self._encoded[idx]

    def to_list(self):
        return self._encoded


class ClaimVerificationDatasetSum(ClaimVerificationDataset):
    def __init__(self, claim_verification_data):
        super(ClaimVerificationDataset, self).__init__()
        self._data = claim_verification_data

        self._encoded = []
        for d in self._data:
            self._encoded.append(self._encode_one_sample(d))
    
    def _encode_one_sample(self, sample):
        claim = sample['claim']
        text_evidence = sample['text']
        true_label = sample['label']
        summary = sample['summary'].split("assistant\n")[-1]

        encoded_sample = {}
        encoded_sample["claim"] = claim
        encoded_sample["correct_choice"] = true_label
        encoded_sample['text_evidence'] = text_evidence
        encoded_sample['summary'] = summary
        encoded_sample['label'] = true_label

        return encoded_sample

def load_peft_model_vision(peft_model_name, device="auto", quantile=True, flash_attention=True):
    min_pixels = 256*28*28
    max_pixels = 1280*28*28
    processor = AutoProcessor.from_pretrained(
        peft_model_name,
        token="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",
        min_pixels=min_pixels, max_pixels=max_pixels
    )
    # processor.tokenizer.padding_side = "left"

    quantization_config = BitsAndBytesConfig(
        llm_int8_threshold=6.0,
        llm_int8_has_fp16_weight=False,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        load_in_4bit=True,
        load_in_8bit=False
    )

    if quantile:
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        peft_model_name,
        quantization_config=quantization_config,
        token="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",
        device_map=device,
        use_flash_attention_2=flash_attention,
    )
    else:
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        peft_model_name,
        token="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",
        device_map=device,
        use_flash_attention_2=flash_attention,
    )

    return processor, model


def load_peft_model_text(peft_model_name, device="auto", quantile=True, flash_attention=True):
    processor = AutoTokenizer.from_pretrained(
        peft_model_name,
        model_max_length=2048,
        padding_side="left",
        truncation_side="left",
        token="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"
    )

    quantization_config = BitsAndBytesConfig(
        llm_int8_threshold=6.0,
        llm_int8_has_fp16_weight=False,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        load_in_4bit=True,
        load_in_8bit=False,
    )

    if quantile:
        model = AutoModelForCausalLM.from_pretrained(
        peft_model_name,
        quantization_config=quantization_config,
        token="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",
        device_map=device,
        use_flash_attention_2=flash_attention
    )
    else:
        model = AutoModelForCausalLM.from_pretrained(
        peft_model_name,
        token="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",
        device_map=device,
        use_flash_attention_2=flash_attention
    )

    return processor, model


def make_verification_prompt_text_only(claim, text_evidence, summary, label=None, is_train=False): 
    if not is_train:
        # print("Inference prompt")   
        prompt = f"""
        You are an assistant that help choosing the correct truthfulness label for a claim.
            The CLAIM: {claim}
            The CONTEXT: {text_evidence}
            The IMAGE: {summary}

            Truthfulness choices:
            A. True.
            B. False.
            C. Partially True.
            D. Not verifiable.  

            Think step-by-step to predict the truthfulness of the CLAIM. Response one correct Truthfulness choices: A, B, C, or D without explanation.
            <RESPONSE>: 
            """
    else:
        # print("Fine-tuning prompt")
        assert label != None
        prompt = f"""
        You are an assistant that help choosing the correct truthfulness label for a claim.
            The CLAIM: {claim}
            The CONTEXT: {text_evidence}
            The IMAGE: {summary}

            Truthfulness choices:
            A. True.
            B. False.
            C. Partially True.
            D. Not verifiable.  

            Think step-by-step to predict the truthfulness of the CLAIM. Response one correct Truthfulness choices: A, B, C, or D without explanation.
            <RESPONSE>: {label}
            """
    return prompt


@torch.inference_mode()
def do_inference_vision(model, processor, prompt, new_token=10):
    # image_file = Image.open(image)
    text = processor.apply_chat_template(
        prompt, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(prompt)
    # inputs = processor(text=prompt, images=image_file, padding=True, return_tensors="pt").to(model.device)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )

    inputs = inputs.to(model.device)
    
    output_ids = model.generate(
        **inputs,
        max_new_tokens=new_token,
        do_sample=False
    )
    return processor.decode(output_ids[0], skip_special_tokens=True, clean_up_tokenization_spaces=True)


@torch.inference_mode()
def do_inference_text(model, processor, prompt, new_token=10):
    inputs = processor(prompt, return_tensors="pt").to(model.device)

    model.generation_config.pad_token_id = processor.pad_token_id
    output_ids = model.generate(
        **inputs,
        max_new_tokens=new_token,
        do_sample=False,
    )
    return processor.decode(output_ids[0])


def perform_verify(data, img_path, model, processor, summary=False):
    results = []
    print("--Verifying--")
    for d in tqdm(data):
        # prompt, image = make_verification_prompt(d['claim'], d['text_evidence'], d['image_evidence'], img_path)
        prompt = build_messages(d, img_path, summary=summary)
        result = do_inference_vision(model, processor, prompt, new_token=10)
        results.append({
            "claim": d['claim'],
            "text": d['text_evidence'],
            "image": d['image_evidence'],
            "response": result,
            "label": d['correct_choice']
        })
    return results


def perform_verify_text(data, model, processor):
    results = []
    print("--Verifying text --")
    is_show = True
    for d in tqdm(data):
        prompt = make_verification_prompt_text_only(d['claim'], d['text_evidence'], d['summary'])
        if is_show:
            print(prompt)
            print("============")
            is_show = False
        result = do_inference_text(model, processor, prompt, new_token=10)
        results.append({
            "claim": d['claim'],
            "text": d['text_evidence'],
            "response": result,
            "summary": d['summary'],
            "label": d['correct_choice']
        })
    return results


def perform_generation(data, img_path, model, processor):
    results = []
    print("--Generation--")
    for d in tqdm(data):
        try:
            prompt = build_messages2(d, img_path)
            result = do_inference_vision(model, processor, prompt, new_token=128)
            results.append({
                "claim": d['claim'],
                "text": d['text_evidence'],
                "image": d['image_evidence'],
                "summary": result,
                "label": d['correct_choice']
            })
        except Exception as e:
            print(e)
            print("-------")
            print(d['claim'])
            print(d['image_evidence'])
    return results


def retrieve_verification_results(data):
    def get_choice_label(text):
        p = re.compile(r'[{A}{B}{C}{D}]+')
        if len(p.findall(text)) > 0:
            return p.findall(text)[0]
        return "D"
    
    def filter_results(response):
        response = response.split("assistant\n")[-1]
        return get_choice_label(response)

    label2idx = {
        'A': 3,
        'B': 2,
        'C': 1,
        'D': 0
    }

    ground_truth = []
    predict = []
    for d in data:
        predict.append(label2idx[filter_results(d['response'])])
        ground_truth.append(label2idx[d['label']])
        d['predict'] = filter_results(d['response'])
    
    return ground_truth, predict, data


def retrieve_verification_results_text(data):
    def get_choice_label(text):
        p = re.compile(r'[{A}{B}{C}{D}]+')
        if len(p.findall(text)) > 0:
            return p.findall(text)[0]
        return "D"
    
    def filter_results(response):
        response = response.split("<RESPONSE>: ")[-1]
        return get_choice_label(response)

    label2idx = {
        'A': 3,
        'B': 2,
        'C': 1,
        'D': 0
    }

    ground_truth = []
    predict = []
    for d in data:
        predict.append(label2idx[filter_results(d['response'])])
        ground_truth.append(label2idx[d['label']])
        d['predict'] = filter_results(d['response'])
    
    return ground_truth, predict, data


def build_messages(sample, image_folder, summary=False):
    instruction_prompt = """You are an effective multi-modal fact-checking model. Your task is to verify the factual accuracy of claims by analyzing multimodal inputs. The given inputs include: a claim in textual form, which can be a news headline, a sentence from an article, or a social media post; an accompanying image related to the claim; and an additional context in textual form, such as the full text of the news article, a related social media discussion, or other supplementary information. You must determine the factual accuracy of the claim based on all the provided inputs. They need to assign one of four possible labels to the claim: \"True\", \"False\", \"Partially True\", and \"Not Verifiable\" by choosing one of the following letters: A, B, C, D. The answer must be in the format of a single letter (A, B, C or D) without any additional text or explanation. Do not include any other information in your answer. Please provide the answer only."""
    
    if not summary:
        user_prompt_template = """CONTEXT: [CONTEXT]
        CLAIM: [CLAIM]
        [CHOICES]
        The answer is:"""
        
        choices = []
        for choice in sample["choices"]:
            choices.append(f"{choice['id']}. {choice['choice']}")
        choices = "\n".join(choices)
        
        user_prompt = user_prompt_template.replace("[CONTEXT]", sample["text_evidence"])\
            .replace("[CLAIM]", sample["claim"])\
            .replace("[CHOICES]", choices)
    else:
        user_prompt_template = """CONTEXT: [CONTEXT]
        CLAIM: [CLAIM]
        SUMMARY OF IMAGE: [SUM_IMG]
        [CHOICES]
        The answer is:"""
        
        choices = []
        for choice in sample["choices"]:
            choices.append(f"{choice['id']}. {choice['choice']}")
        choices = "\n".join(choices)
        
        user_prompt = user_prompt_template.replace("[CONTEXT]", sample["text_evidence"])\
            .replace("[CLAIM]", sample["claim"])\
            .replace("[CHOICES]", choices)\
            .replace("[SUM_IMG]", sample["summary"])

    messages = [
        {
            "role": "system",
            "content": [
                {"type": "text", "text": instruction_prompt}
            ]
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "Here are the given inputs:\nIMAGE:"
                },
                {
                    "type": "image",
                    "image": "file://{}/image/{}".format(image_folder, sample['image_evidence'].split("/")[-1]),
                },
                {"type": "text", "text": user_prompt},
            ],
        }
    ]
    return messages


def build_messages2(sample, image_folder):
    instruction_prompt = """You are an effective multi-modal visual description model. Your task is to generate a short description that summary keypoints between the image and the context. The given inputs include: an  image that may come from social media posts; and a context in textual form, such as the full text of the news article, a related social media discussion, or other supplementary information. You should generate a short paragraph which is no more than 100 tokens to summary keypoints between the image and the context."""
    
    user_prompt_template = """CONTEXT: [CONTEXT]
    The answer is: """
    
    
    user_prompt = user_prompt_template.replace("[CONTEXT]", sample["text_evidence"])
    messages = [
        {
            "role": "system",
            "content": [
                {"type": "text", "text": instruction_prompt}
            ]
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "Here are the given inputs:\nIMAGE:"
                },
                {
                    "type": "image",
                    "image": "file://{}/image/{}".format(image_folder, sample['image_evidence'].split("/")[-1]),
                },
                {"type": "text", "text": user_prompt},
            ],
        }
    ]
    return messages
    

if __name__ == '__main__':
    PATH = "/home/sonlt/drive/data/acmm25/fact_checking"
    image_db = read_image_path_only("{}/image".format(PATH))
    train, dev, test = read_data("{}/json".format(PATH))

    processor, model = load_peft_model_text("/data/son_models/Qwen2.5-VL-72B-Instruct", flash_attention=False)

    # train_claim = ClaimVerificationDataset(train, image_db)
    dev_claim = ClaimVerificationDataset(dev, image_db)
    # test_claim = ClaimVerificationDataset(test, image_db)
    # result = perform_verify(test_claim, PATH, model, processor)
    # result = perform_generation(test_claim, PATH, model, processor)
    print("dev")
    result = perform_generation(dev_claim, PATH, model, processor)
    
    with open('./dev_qwen2.5_72_summary.json', 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=4)
    f.close()

    # with open("./test_qwen2.5_vl_raw_new.json", "r") as f:
    #     result = json.load(f)
    # f.close()

    # g, p, new_results = retrieve_verification_results(result)
    
    # print("Test result micro: {}\n".format(f1_score(g, p, average='micro')))
    # print("Test result macro: {}\n".format(f1_score(g, p, average='macro')))
    # print("Test result Accuracy: {}\n".format(accuracy_score(g, p)))
    # print(confusion_matrix(g, p, labels=[0, 1, 2, 3]))

    # with open('./test_qwen2.5_vl.json', 'w', encoding='utf-8') as f:
    #     json.dump(new_results, f, ensure_ascii=False, indent=4)
    # f.close()