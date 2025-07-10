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
    ## INFERENCE VISION
    print("infer vision")
    PATH = "/home/sonlt/drive/data/acmm25/fact_checking"
    image_db = read_image_path_only("{}/image".format(PATH))
    train, dev, test = read_data("{}/json".format(PATH))
    
    processor, model = load_peft_model_vision("/data/huggingface_models/Qwen2.5-VL-72B-Instruct", flash_attention=False)
    test_claim = ClaimVerificationDataset(dev, image_db)
    result = perform_verify(test_claim, PATH, model, processor, summary=False)

    g, p, new_results = retrieve_verification_results(result)
    print("Test result micro: {}\n".format(f1_score(g, p, average='micro')))
    print("Test result macro: {}\n".format(f1_score(g, p, average='macro')))
    print("Test result Accuracy: {}\n".format(accuracy_score(g, p)))
    print(confusion_matrix(g, p, labels=[0, 1, 2, 3]))

    with open('./qwen2.5VL-72-dev.json', 'w', encoding='utf-8') as f:
        json.dump(new_results, f, ensure_ascii=False, indent=4)
    f.close()
