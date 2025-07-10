"""
⣇⣿⠘⣿⣿⣿⡿⡿⣟⣟⢟⢟⢝⠵⡝⣿⡿⢂⣼⣿⣷⣌⠩⡫⡻⣝⠹⢿⣿⣷
⡆⣿⣆⠱⣝⡵⣝⢅⠙⣿⢕⢕⢕⢕⢝⣥⢒⠅⣿⣿⣿⡿⣳⣌⠪⡪⣡⢑⢝⣇
⡆⣿⣿⣦⠹⣳⣳⣕⢅⠈⢗⢕⢕⢕⢕⢕⢈⢆⠟⠋⠉⠁⠉⠉⠁⠈⠼⢐⢕⢽
⡗⢰⣶⣶⣦⣝⢝⢕⢕⠅⡆⢕⢕⢕⢕⢕⣴⠏⣠⡶⠛⡉⡉⡛⢶⣦⡀⠐⣕⢕
⡝⡄⢻⢟⣿⣿⣷⣕⣕⣅⣿⣔⣕⣵⣵⣿⣿⢠⣿⢠⣮⡈⣌⠨⠅⠹⣷⡀⢱⢕
⡝⡵⠟⠈⢀⣀⣀⡀⠉⢿⣿⣿⣿⣿⣿⣿⣿⣼⣿⢈⡋⠴⢿⡟⣡⡇⣿⡇⡀⢕
⡝⠁⣠⣾⠟⡉⡉⡉⠻⣦⣻⣿⣿⣿⣿⣿⣿⣿⣿⣧⠸⣿⣦⣥⣿⡇⡿⣰⢗⢄
⠁⢰⣿⡏⣴⣌⠈⣌⠡⠈⢻⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣬⣉⣉⣁⣄⢖⢕⢕⢕
⡀⢻⣿⡇⢙⠁⠴⢿⡟⣡⡆⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣷⣵⣵⣿
⡻⣄⣻⣿⣌⠘⢿⣷⣥⣿⠇⣿⣿⣿⣿⣿⣿⠛⠻⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿
⣷⢄⠻⣿⣟⠿⠦⠍⠉⣡⣾⣿⣿⣿⣿⣿⣿⢸⣿⣦⠙⣿⣿⣿⣿⣿⣿⣿⣿⠟
⡕⡑⣑⣈⣻⢗⢟⢞⢝⣻⣿⣿⣿⣿⣿⣿⣿⠸⣿⠿⠃⣿⣿⣿⣿⣿⣿⡿⠁⣠
⡝⡵⡈⢟⢕⢕⢕⢕⣵⣿⣿⣿⣿⣿⣿⣿⣿⣿⣶⣶⣿⣿⣿⣿⣿⠿⠋⣀⣈⠙
⡝⡵⡕⡀⠑⠳⠿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⠿⠛⢉⡠⡲⡫⡪⡪⡣
"""
import yaml
import torch
import json
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

from tqdm import tqdm
from utils.data_utils import load_json
from transformers import AutoModelForCausalLM, AutoTokenizer

class ClaimGenerator():
    def __init__(self, prompt_path, model_name, cache_dir):
        with open(prompt_path,"r",encoding='utf-8') as file:
            self.prompt = yaml.load(file, yaml.FullLoader)

        self.model = AutoModelForCausalLM.from_pretrained(model_name, 
                                                          device_map="auto", 
                                                          torch_dtype=torch.float16, 
                                                          attn_implementation="flash_attention_2", 
                                                          cache_dir=cache_dir)
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name,
                                                       cache_dir=cache_dir)

    def get_response(self, text):
        user_prompt = self.prompt["user"].format(text=text)
        message = [
                {"role": "system", "content": self.prompt["system"]},
                {"role": "user", "content": user_prompt}
            ]
        chat_text = self.tokenizer.apply_chat_template(
            message,
            tokenize=False,
            add_generation_prompt=True
        )
        model_inputs = self.tokenizer([chat_text], return_tensors="pt").to(self.model.device)

        generated_ids = self.model.generate(
            **model_inputs,
            max_new_tokens=2048,
            do_sample=True,
            temperature=0.3, ### 0.3 for instruction following
        )
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]

        response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

        print(response)
        
        try:
            response = json.loads(response)
        except Exception as e:
            print(e)
        
        return response
    

def main():
    """
        Input: System_answer
        Output: Claim 
        Few-shot prompting LLM
    """
    model_name = "Qwen/Qwen2.5-72B-Instruct"
    claim_prompt_path = "prompt/1_extract_segment.yaml"
    data = load_json("data/json/test.json")
    output_path = "output/1_to_segment.json"
    model_path = "/mnt/data/hoangcqn/model"
    
    claim_generator = ClaimGenerator(model_name=model_name, prompt_path=claim_prompt_path, cache_dir = model_path)

    # Main process
    for sample in tqdm(data):
        description = sample["system1_answer"]
        response = claim_generator.get_response(description)

        sample["response"] = response

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
                       
if __name__ == "__main__":
    main()