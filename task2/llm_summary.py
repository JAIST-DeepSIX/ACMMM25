from data import *
from model import *
from llm import *
from sklearn.metrics import f1_score, confusion_matrix
import argparse
from trl import SFTTrainer, SFTConfig
from peft import LoraConfig
from datasets import load_dataset

def parser_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epoch', type=int, default=10)
    parser.add_argument('--val', default=True, action='store_true')
    parser.add_argument('--data_path', type=str, default="/home/s2320014/data")
    parser.add_argument('--test', default=False, action='store_true')
    parser.add_argument('--n_gpu', type=int, default=None)
    args = parser.parse_args()
    return args


def make_prompt_sample(d):
    prompt = make_verification_prompt_text_only(d['claim'], d['text'], d['summary'].split("assistant\n")[-1], d['label'], is_train=True)
    d["prompt"] = prompt
    return d


def formatting_prompts_func(example):
    return example['prompt']


if __name__ == '__main__':
    PATH = "/home/sonlt/drive/data/acmm25/fact_checking"
    image_db = read_image_path_only("{}/image".format(PATH))
    train, dev, test = read_data("{}/json".format(PATH))

    processor, model = load_peft_model_vision("/data/huggingface_models/Qwen2.5-VL-72B-Instruct", flash_attention=False)

    train_claim = ClaimVerificationDataset(train, image_db)
    dev_claim = ClaimVerificationDataset(dev, image_db)
    test_claim = ClaimVerificationDataset(test, image_db)
    print("train")
    result = perform_generation(train_claim, PATH, model, processor)
    
    with open('./train_qwen2.5_vl_summary.json', 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=4)
    f.close()

    print("dev")
    result = perform_generation(dev_claim, PATH, model, processor)
    
    with open('./dev_qwen2.5_vl_summary.json', 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=4)
    f.close()

    print("test")
    result = perform_generation(test_claim, PATH, model, processor)
    
    with open('./test_qwen2.5_vl_summary.json', 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=4)
    f.close()