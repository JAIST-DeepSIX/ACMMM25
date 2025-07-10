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
import yaml
import base64
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
    def __init__(self, prompt_path, assistant_prompt_path):
        with open(prompt_path, "r") as f:
            self.prompt = yaml.load(f, yaml.FullLoader)
            
        with open(assistant_prompt_path, "r") as f:
            self.assistant_prompt = yaml.load(f, yaml.FullLoader)
        
    def prepare_image(self, image_id):    
        image_folder = Path("data/images")            # image folder
        image_path = f"{image_folder}/{image_id}" # image path  
            
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")
    
    def prepare_fs_examples(self):
        """
        Prepare examples for few-shot learning
        Output: list of dictionaries
        [
            {
                "image_id": image_id,
                "user_prompt": user_prompt,
                "assistant_prompt": assistant_prompt
            },
            ...
        ]
        """
        train_data = load_json("data/json/train.json")
        
        res = []
        ex_img_ids = ["021fffa3d66f9b77.jpg", "124d86a6d3953a13.jpg", "e79be58ffbf5bba0.jpg"]
        ex_claim_by_ids = {
            "021fffa3d66f9b77.jpg": "Both men are dressed in formal purple suits.",
            "124d86a6d3953a13.jpg": "There is a bouquet of flowers on the podium.",
            "e79be58ffbf5bba0.jpg": "Bicycle gear and chain are metallic gray and black.",
        }
        for sample in train_data:
            if sample["image_id"] in ex_img_ids:
                image_id = sample["image_id"]
                ## input are image, claim
                user_prompt = self.prompt["user"].format(text = ex_claim_by_ids[image_id], keywords=json.dumps(sample["choices"], ensure_ascii=False, indent=2))
                res.append({
                    "image_id": image_id,
                    "user_prompt": user_prompt,
                    "assistant_prompt": self.assistant_prompt[f"example_{image_id}"]
                })
                
        return res
        
    def get_response(self, sample_lv, user_input):
        # Build prompt
        # Prepare input image
        input_image = self.prepare_image(sample_lv["image_id"])
        
        #### Prepare input text
        choices = sample_lv["choices"]
        # # Fill in the placeholders
        # # build a clean JSON blob of choices to avoid stray braces
        choices_blob = json.dumps(choices, ensure_ascii=False, indent=2)
        
        ### Prepare input for image, claim
        filled_user = self.prompt["user"].format(text = user_input["claim"], keywords=choices_blob)
        
        # Prepare few-shot examples
        fs_examples = self.prepare_fs_examples()
        print(f"fs_examples: {fs_examples}")
        
        messages = [
            {"role": "system", 
            "content": self.prompt["system"]},
            {"role": "user", "content": [
                {"type": "text", "text": "Here are the given inputs:\nIMAGE:"},
                {"type": "image_url", "image_url": {"url": f"data:image;base64,{self.prepare_image(fs_examples[0]["image_id"])}"}},
                {"type": "text", "text": fs_examples[0]["assistant_prompt"]},
            ],
            },
            {"role": "assistant", 
             "content": [
                {"type": "text", "text": fs_examples[0]["user_prompt"]} # 1
            ]}, 
            {"role": "user", 
             "content": [
                {"type": "text", "text": "Here are the given inputs:\nIMAGE:"},
                {"type": "image_url", "image_url": {"url": f"data:image;base64,{self.prepare_image(fs_examples[1]["image_id"])}"}},
                {"type": "text", "text": fs_examples[1]["assistant_prompt"]},
            ],
            },
            {"role": "assistant", "content": [
                {"type": "text", "text": fs_examples[1]["user_prompt"]} # 2
            ]},
            {"role": "user", 
             "content": [
                {"type": "text", "text": "Here are the given inputs:\nIMAGE:"},
                {"type": "image_url", "image_url": {"url": f"data:image;base64,{self.prepare_image(fs_examples[2]["image_id"])}"}},
                {"type": "text", "text": fs_examples[2]["assistant_prompt"]},
            ],
            },
            {"role": "assistant", "content": [
                {"type": "text", "text": fs_examples[2]["user_prompt"]} # 3
            ]},
            {"role": "user", 
             "content": [
                {"type": "text", "text": "Here are the given inputs:\nIMAGE:"},
                {"type": "image_url", "image_url": {"url": f"data:image;base64,{input_image}"}},
                {"type": "text", "text": filled_user}
            ]},
        ]
        
        chat_response = client.chat.completions.create(
            model="Qwen/Qwen2.5-VL-72B-Instruct",
            messages=messages,
            temperature=0.6,
            top_p=0.95,
            response_format={ "type": "json_object" }
        )

        return chat_response
def main():
    """
    """
    
    # define path
    data = load_json("output/3_data_onlyclaim_wkeyphrase.json")                                 # data
    prompt_path = "prompt/4_fs_verifier_vllm/4_fs_verifier_vllm_cot.yaml"                       # yaml verify_claim prompt
    assistant_prompt_path = "prompt/4_fs_verifier_vllm/4_fs_verifier_vllm_cot_assistant.yaml"   # yaml assistant prompt
    outpath = "output/4_CoT_ClaimKeyword_wPredict.json"                                         # output json 
    

    labeler = ClaimLabeler(prompt_path=prompt_path, assistant_prompt_path=assistant_prompt_path)

    for sample in tqdm(data, desc="Fewshot image & system answer"): 
        for segment in sample["response"]:
                for claim in segment["claims"]:
                    claim["verification"] = labeler.get_response(sample, claim).choices[0].message.content
                    print(f"verification: {claim["verification"]}")
        
    with open(outpath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"Successfully saved to result to {outpath}")

if __name__ == "__main__":
    main()