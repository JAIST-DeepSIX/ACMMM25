from transformers import EarlyStoppingCallback ,TrainingArguments, Trainer, DataCollatorWithPadding
from datasets import Dataset, DatasetDict
from peft import LoraConfig, get_peft_model, TaskType
from preprocessing import make_data
from model import make_model
from postprocessing import compute_metrics

def train(model_id="Qwen/Qwen2.5-72B-Instruct", 
            path_load_model = None, 
            summary_data_path="./data",
            train_save_dir="./train/1",
            learning_rate=2e-5,
            per_device_train_batch_size=2,
            per_device_eval_batch_size=2,
            eval_steps = 15,
            num_train_epochs=25,
            logging_steps=15,
            weight_decay=0.01,
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            save_total_limit = 2,
            early_stopping_patience=3
          ):
    """
    Train a model for sequence classification task using LoRA.
    Args:
        model_id (str): HuggingFace model ID.
        path_load_model (str): Path to a pretrained model to continue training.
        sumary_data_path (str): Path to the directory containing summary data.
        train_save_dir (str): Directory to save the trained model.
        learning_rate (float): Learning rate for training.
        per_device_train_batch_size (int): Batch size for training.
        per_device_eval_batch_size (int): Batch size for evaluation.
        eval_steps (int): Number of steps between evaluations.
        num_train_epochs (int): Number of epochs for training.
        logging_steps (int): Number of steps between logging.
        weight_decay (float): Weight decay for optimization.
        eval_strategy (str): Evaluation strategy during training.
        save_strategy (str): Model saving strategy during training.
        load_best_model_at_end (bool): Whether to load the best model at the end of training.
        save_total_limit (int): Maximum number of saved models.
        early_stopping_patience (int): Patience for early stopping.
    
    Note: You can adjust the parameters as needed for your training configuration.
    """
    id2label = {0: "A", 
               1: "B", 
               2: "C",
               3: "D"}
    label2id = {"A": 0, 
                "B": 1,
                "C": 2,
                "D": 3}
    data = {
        "train" : make_data(json_path=f"{summary_data_path}/train_qwen2.5_vl_summary.json",
                     label2id=label2id),
        "test" : make_data(json_path=f"{summary_data_path}/dev_qwen2.5_vl_summary.json",
                     label2id=label2id)
    }
    model, _, tokenizer = make_model(model_id=model_id,
                                            path_load_model=path_load_model, 
                                            num_labels=4, 
                                            id2label=id2label, 
                                            label2id=label2id, 
                                            )
    
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
    
    def preprocess_fn(samples):
        return tokenizer(samples["text"], truncation=True)
    
    
    tokenized_data = data.map(preprocess_fn, batched=True)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    training_args = TrainingArguments(
        output_dir=train_save_dir,
        learning_rate=learning_rate,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_eval_batch_size,
        eval_steps = eval_steps,
        num_train_epochs=num_train_epochs,
        logging_steps=logging_steps,
        weight_decay=weight_decay,
        eval_strategy=eval_strategy,
        save_strategy=save_strategy,
        load_best_model_at_end=load_best_model_at_end,
        save_total_limit = save_total_limit
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_data["train"],
        eval_dataset=tokenized_data["test"],
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=early_stopping_patience)],
    )
    trainer.train()
    trainer.save_model(training_args.output_dir)





