from transformers import AutoModelForSequenceClassification, BitsAndBytesConfig, AutoProcessor, AutoTokenizer
import torch
from preprocessing import make_message

def make_model(model_id,num_labels , path_load_model = None, id2label=None, label2id=None):
    """"
    Create model, processor and tokenizer for sequence classification task.
    """
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True, 
        bnb_4bit_use_double_quant=True, 
        bnb_4bit_quant_type="nf4", 
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    
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