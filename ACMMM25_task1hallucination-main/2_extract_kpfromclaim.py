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
import torch
import json
import yaml
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1, 2, 3"
from tqdm import tqdm
from utils.data_utils import load_json
from transformers import AutoModelForCausalLM, AutoTokenizer

class ClaimKW_Extractor():
    def __init__(self, prompt_path, model_name, model_path):
        with open(prompt_path,"r",encoding="utf-8") as file:
            self.prompt = yaml.load(file, yaml.FullLoader)

        self.model = AutoModelForCausalLM.from_pretrained(model_name, 
                                                          device_map="auto", 
                                                          torch_dtype=torch.float16, 
                                                          attn_implementation="flash_attention_2",
                                                          cache_dir=model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name,cache_dir = model_path)
        
    def get_response(self, claim, keywords):

        # user_prompt = self.prompt["user"].format(text=text)
        user_prompt =  self.prompt["user"]
        user_prompt = user_prompt.replace("{claim}", claim).replace("{keywords}", str(keywords))
        
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
            max_new_tokens=1024,
            do_sample=True,
            temperature=0.5, ### 0.3 for instruction following
        )
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]

        response = self.tokenizer.batch_decode(generated_ids, 
                                               skip_special_tokens=True)[0]

        try:
            response = json.loads(response)
        except Exception as e:
            print(e)
        
        return response

def enrich_claims_with_keywords(data, extractor):
    for sample in tqdm(data):
        # build the keyword list once per sample
        text_B = sample["choices"][1]["choice"]
        text_C = sample["choices"][2]["choice"]
        text_D = sample["choices"][3]["choice"]
        keywords = [f"B. {text_B}", f"C. {text_C}", f"D. {text_D}"]
        print(keywords)
        for seg in sample["response"]:
            new_claims = []
            for claim_dict in seg["claims"]:
                claim_text = claim_dict["claim"]
                print(claim_text)
                # 1) ask the extractor about this single claim
                resp = extractor.get_response(claim=claim_text, keywords=keywords)
                print(f"this is resp: ")
                # 2) pull out its "keywords" list (flatten if it's wrapped in a list)
                if isinstance(resp, list):
                    flat = []
                    for entry in resp:
                        flat.extend(entry.get("keywords", []))
                else:
                    flat = resp.get("keywords", [])

                # 3) build your enriched claim dict
                enriched = {
                    "claim": claim_text,
                    "keywords": flat
                }
                print(enriched)
                new_claims.append(enriched)

            # overwrite the old list
            seg["claims"] = new_claims

    return data
    
def main():
    """
    Load the data and run the model
    """
    # Load the data
    data = load_json("output/1_to_segment.json")
    # Define path
    model_path = ""                                         #cache_dir
    prompt_path = "prompt/2_extract_claim_keywords.yaml"
    # Initialize the model
    extractor = ClaimKW_Extractor(prompt_path=prompt_path, 
                                  model_name="Qwen/Qwen2.5-32B-Instruct",
                                  model_path=model_path
                                  )
    
    enriched = enrich_claims_with_keywords(data, extractor)
    with open("output/2_to_keyphrase.json", "w", encoding="utf-8") as file:
        json.dump(enriched, file, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    main()