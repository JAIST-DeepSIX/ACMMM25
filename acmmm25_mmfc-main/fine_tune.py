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
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # ## INFERENCE
    # processor, model = load_peft_model_text("./model_dump/Qwen2.5-14B-Instruct-10ep", flash_attention=False)
    # with open("./test_qwen2.5_vl_summary.json", "r") as f:
    #     test = json.load(f)
    # f.close()
    # test_claim = ClaimVerificationDatasetSum(test)
    # result = perform_verify_text(test_claim, model, processor)

    # g, p, new_results = retrieve_verification_results_text(result)
    
    # print("Test result micro: {}\n".format(f1_score(g, p, average='micro')))
    # print("Test result macro: {}\n".format(f1_score(g, p, average='macro')))
    # print("Test result Accuracy: {}\n".format(accuracy_score(g, p)))
    # print(confusion_matrix(g, p, labels=[0, 1, 2, 3]))

    # with open('./qwen2.5-14f-10.json', 'w', encoding='utf-8') as f:
    #     json.dump(new_results, f, ensure_ascii=False, indent=4)
    # f.close()


    ## FINE_TUNE
    processor, model = load_peft_model_text("/data/huggingface_models/Qwen2.5-14B-Instruct", flash_attention=True)
    # processor, model = load_peft_model_text("./model_dump/Qwen2.5-14B-Instruct-30ep", flash_attention=True)
    with open("./train_qwen2.5_vl_summary.json", "r") as f:
        train = json.load(f)
    f.close()

    with open("./dev_qwen2.5_vl_summary.json", "r") as f:
        dev = json.load(f)
    f.close()

    with open("./test_qwen2.5_vl_summary.json", "r") as f:
        test = json.load(f)
    f.close()

    dataset_train = [make_prompt_sample(d) for d in train]
    dataset_dev = [make_prompt_sample(d) for d in dev]
    
    print("Loaded {} prompts".format(len(dataset_train)))
    print(f"Some training examples:")
    for i in range(0, 1):
        print(json.dumps(dataset_train[i], ensure_ascii=False))
        print("-" * 10)
    
    peft_config = LoraConfig(
        r=16,
        lora_alpha=16,
        bias="none",
        task_type="CAUSAL_LM",
    )

    training_args = SFTConfig(
        output_dir="./model_dump/Qwen2.5-14B-Instruct-50ep",
        num_train_epochs=50,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir="./model_dump/Qwen2.5-14B-Instruct-50ep",
        logging_steps=100,
        max_seq_length=1024,
        packing=True,
        save_total_limit=2
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset_train,
        eval_dataset=dataset_dev,
        peft_config=peft_config,
        formatting_func=formatting_prompts_func
    )

    trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)

    # Step 6: Save the model
    trainer.save_model(training_args.output_dir)

    
    # args = parser_args()
    # image_db = read_image_path_only("{}/image".format(args.data_path))
    # train, dev, test = read_data("{}/json".format(args.data_path))

    # train_claim = ClaimVerificationDatasetSum(train, image_db)
    # dev_claim = ClaimVerificationDatasetSum(dev, image_db)


    # if args.n_gpu:
    #     device = torch.device('cuda:{}'.format(args.n_gpu) if torch.cuda.is_available() else 'cpu')
    # else:
    #     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # if args.test:
    #     model = torch.load(args.model_path, map_location=device)
    # else:
    #     model, loss, name_pt = train_model(train_claim, batch_size=args.batch_size,
    #                                  epoch=args.epoch, is_val=args.val, val_data=dev_claim, device=device, save_checkpoint=False)

    # gtd, prdd = predict(dev_claim, model, args.batch_size, device=device)
    # print("Dev result micro: {}\n".format(f1_score(gtd, prdd, average='micro')))
    # print("Dev result macro: {}\n".format(f1_score(gtd, prdd, average='macro')))

    # output_dfd = pd.DataFrame({'predict': prdd, 'ground_truth': gtd})
    # output_dfd.to_csv('predict_dev.csv')

    # gt, prd = predict(test_claim, model, args.batch_size, device=device)
    # print("Test result micro: {}\n".format(f1_score(gt, prd, average='micro')))
    # print("Test result macro: {}\n".format(f1_score(gt, prd, average='macro')))

    # output_df = pd.DataFrame({'predict': prd, 'ground_truth': gt})
    # output_df.to_csv('predict_test.csv')