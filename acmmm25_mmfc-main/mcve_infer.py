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
    # MCVE
    args = parser_args()
    image_db = read_image_path_only("{}/image".format("/home/sonlt/drive/data/acmm25/fact_checking"))
    train, dev, test = read_data("{}/json".format("/home/sonlt/drive/data/acmm25/fact_checking"))

    train_claim = MCVEClaimVerificationDataset(train, image_db)
    dev_claim = MCVEClaimVerificationDataset(dev, image_db)
    test_claim = MCVEClaimVerificationDataset(test, image_db)

    if args.n_gpu:
        device = torch.device('cuda:{}'.format(args.n_gpu) if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if args.test:
        model = torch.load("./model_dump/model_verification_roberta-base_longformer_vit_18-04_15-04/best_model.pt", map_location=device, weights_only=False)

        gt, prd = predict(dev_claim, model, args.batch_size, device=device)
        print("Test result micro: {}\n".format(f1_score(gt, prd, average='micro')))
        print("Test result macro: {}\n".format(f1_score(gt, prd, average='macro')))
        print(confusion_matrix(gt, prd, labels=[0, 1, 2, 3]))

        output_df = pd.DataFrame({'predict': prd, 'ground_truth': gt})
        output_df.to_csv('MCVE_dev.csv')
    else:
        model, loss, name_pt = train_model(train_claim, batch_size=args.batch_size,
                                     epoch=args.epoch, is_val=args.val, val_data=dev_claim, device=device, save_checkpoint=False)
