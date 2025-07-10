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
import json
import os 
import torch
import yaml
import base64
# from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor, Qwen2VLForConditionalGeneration
from PIL import Image
from pathlib import Path
from tqdm import tqdm
from utils.data_utils import load_json

from openai import OpenAI
# Set OpenAI's API key and API base to use vLLM's API server.
openai_api_key = "EMPTY"
openai_api_base = "http://localhost:8000/v1"

client = OpenAI(
    api_key=openai_api_key,
    base_url=openai_api_base,
)

class ClaimLabeler:
    def __init__(self, prompt_path):
        with open(prompt_path, "r") as f:
            self.prompt = yaml.load(f, yaml.FullLoader)
        
    def prepare_image(self, image_id):    
        image_folder = Path("data/images")              # image folder
        image_path = f"{image_folder}/{image_id}"       # image path  
            
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")
            
    def get_response(self, sample):
        # Build prompt
        # Prepare input image
        input_image = self.prepare_image(sample["image_id"])
                
        multiple_preds = sample["prediction"]
        choices = []
        for k in multiple_preds:
            for choice in sample["choices"]:
                if k == choice["id"]:
                    choices.append(choice)

        print("len choices: ", len(choices))
        
        # Fill in the placeholders, build a clean JSON
        choices_blob = json.dumps(choices, ensure_ascii=False, indent=2)
        
        # Prepare input for image, claim
        filled_user = self.prompt["user"].format(system1_answer = sample["system1_answer"], keywords=choices_blob)
        
        # Build messages
        messages = [
                {"role": "system", "content": self.prompt["system"]},
                {"role": "user", 
                    "content": [
                        {"type": "text", "text": "Here are the given inputs:\nIMAGE:"},
                        {"type": "image_url", "image_url": {"url": f"data:image;base64,{input_image}"}},
                        {"type": "text", "text": filled_user}
                    ]}
        ]
        
        chat_response = client.chat.completions.create(
            model="Qwen/Qwen2.5-VL-72B-Instruct",
            messages=messages,
            temperature=0.6,
            top_p=0.95,
            response_format={ "type": "json_object" }
        )

        return chat_response
    
def filter_multiple_preds_sample(sample):
    if len(sample["prediction"]) > 1:
        return True
    return False

def main():
    """
    """
    # define path
    data = load_json("output/4_CoT_ClaimKeyword_wPredict.json") # data
    prompt_path = "prompt/5_verifier_final_keyword_fs.yaml"     # yaml verify_claim prompt
    outpath = "output/5_pre_final.json"                         # output json 
    

    labeler = ClaimLabeler(prompt_path=prompt_path)

    for sample in tqdm(data, desc="Fewshot image & system answer"):
        print(filter_multiple_preds_sample(sample))
        if filter_multiple_preds_sample(sample) is False:
            sample["final_shot"] = "Skipped sample with no multiple predictions"
            continue
        
        sample["final_shot"] = labeler.get_response(sample).choices[0].message.content
        print(f"verification: {sample["final_shot"]}")
        
    with open(outpath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"Successfully saved to result to {outpath}")

if __name__ == "__main__":
    main()