from data import *
from model import *
from llm import *
from sklearn.metrics import f1_score, confusion_matrix
import argparse


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


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    ## INFERENCE
    print("infer")
    processor, model = load_peft_model_text("/data/huggingface_models/Qwen2.5-14B-Instruct", flash_attention=False)

    with open("./results/dev_qwen2.5_vl_summary.json", "r") as f:
        test = json.load(f)
    f.close()
    test_claim = ClaimVerificationDatasetSum(test)
    result = perform_verify_text(test_claim, model, processor)

    g, p, new_results = retrieve_verification_results_text(result)
    
    print("Test result micro: {}\n".format(f1_score(g, p, average='micro')))
    print("Test result macro: {}\n".format(f1_score(g, p, average='macro')))
    print("Test result Accuracy: {}\n".format(accuracy_score(g, p)))
    print(confusion_matrix(g, p, labels=[0, 1, 2, 3]))

    with open('./qwen2.5-14B.json', 'w', encoding='utf-8') as f:
        json.dump(new_results, f, ensure_ascii=False, indent=4)
    f.close()