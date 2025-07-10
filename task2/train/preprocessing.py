import json
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
        choices.append(f"{choice['id']}. {choice["choice"]}")
    choices = "\n".join(choices)
    
    message = message_template.replace("[INSTRUCTION]", instruction_prompt)\
        .replace("[CONTEXT]", summary_context)\
        .replace("[CLAIM]", sample["claim"])\
        .replace("[QUESTION]", prototype["question"])\
        .replace("[CHOICES]", choices)
    return message
