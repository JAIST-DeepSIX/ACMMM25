from transformers import ViTImageProcessor, ViTModel, BigBirdModel, BigBirdTokenizer
from transformers import BeitImageProcessor, BeitModel, DeiTModel, DeiTImageProcessor
import numpy as np
from transformers import AutoTokenizer, AutoModel
from transformers import LongformerTokenizer, LongformerModel
import torch.nn.functional as F
from PIL import Image
import math
import random

from torch import optim
import numpy as np
import torch.nn as nn
import torch
from sklearn.metrics import f1_score, confusion_matrix
import copy
from tqdm import tqdm, trange
import datetime
import os


class MultiModalClassificationLarge(nn.Module):
    def vision_model(self, type='vit'):
        if type == 'vit':
            processor = ViTImageProcessor.from_pretrained("google/vit-large-patch16-224-in21k")
            model = ViTModel.from_pretrained("google/vit-large-patch16-224-in21k")
        if type == 'beit':
            processor = BeitImageProcessor.from_pretrained("microsoft/beit-large-patch16-224-pt22k")
            model = BeitModel.from_pretrained("microsoft/beit-large-patch16-224-pt22k")
        if type == 'deit':
            processor = DeiTImageProcessor.from_pretrained("facebook/deit-large-distilled-patch16-224")
            model = DeiTModel.from_pretrained("facebook/deit-large-distilled-patch16-224")

        model.requires_grad_(False)
        return processor, model

    def text_model(self, pt="roberta-large"):
        processor = AutoTokenizer.from_pretrained(pt)
        model = AutoModel.from_pretrained(pt)
        print(pt)
        return processor, model


    def text_model_long(self, pt="longformer"):
        if pt == 'longformer':
            processor = LongformerTokenizer.from_pretrained("allenai/longformer-large-4096")
            model = LongformerModel.from_pretrained("allenai/longformer-large-4096")
        if pt == 'bigbird':
            model = BigBirdModel.from_pretrained("google/bigbird-roberta-large")
            processor = BigBirdTokenizer.from_pretrained("google/bigbird-roberta-large")

        model.requires_grad_(False)
        return processor, model


    def __init__(self, device, claim_pt="roberta-large", vision_pt='vit', long_pt="longformer"):
        super(MultiModalClassificationLarge, self).__init__()
        self._claim_pt = claim_pt
        self._vision_pt = vision_pt
        self._long_pt = long_pt
        self.text_attention = nn.MultiheadAttention(embed_dim=1024, num_heads=4, vdim=1024, kdim=1024)
        self.image_attention = nn.MultiheadAttention(embed_dim=1024, num_heads=4, vdim=1024, kdim=1024)
        self._text_processor, self._text_model = self.text_model(self._claim_pt)
        self._long_text_processor, self._long_text_model = self.text_model_long(self._long_pt)
        self._image_processor, self._vision_model = self.vision_model(self._vision_pt)
        self._device = device

        self.conv = nn.Conv1d(1024, 100, stride=1, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(1024, 100, stride=1, kernel_size=5, padding=2)
        self.conv3 = nn.Conv1d(1024, 100, stride=1, kernel_size=7, padding=3)
        self.fc1 = nn.Linear(1024 * 2, 1024)

        self.pool = nn.MaxPool1d(2, 2)
        self.softmax = nn.Softmax(dim=1)
        self.leaky_relu = nn.LeakyReLU()

        self.fc_claim = nn.Linear(1024, 4)
        self.fc_evidence = nn.Linear(300, 4)

        self.dropout = nn.Dropout(0.2)

    def forward(self, claim_features, label=None):
        device = self._device
        self._vision_model.to(device)
        self._text_model.to(device)
        self._long_text_model.to(device)

        Hc = []
        Ht = []
        Hm = []
        Lb = []

        if label is not None:
            label = label
            for l in label:
                Lb.append(l)

        for claim_feature in claim_features:
            claim = claim_feature['claim']
            text_evidence = claim_feature['text_evidence']
            image_evidence = [Image.open(simg).convert('RGB') for simg in claim_feature['image_evidence']]
            # if len(text_evidence) == 0:
            #     text_evidence.append("")
            # if len(image_evidence) == 0:
            #     blank_img = np.zeros((50, 50, 3), np.uint8)
            #     blank_img.fill(255)
            #     image_evidence.append(blank_img)

            claim_encoded = self._text_processor(claim, return_tensors="pt", padding=True, truncation=True,
                                                 max_length=100).to(device)
            claim_f = self._text_model(**claim_encoded).last_hidden_state.mean(dim=1).to(device)
            # claim_f = self._text_model(**claim_encoded).pooler_output.to(device)

            text_encoded = self._long_text_processor(text_evidence, return_tensors="pt", padding=True,
                                                     truncation=True).to(device)
            text_feature = self._long_text_model(**text_encoded).last_hidden_state.mean(dim=1).to(device)
            # text_feature = self._long_text_model(**text_encoded).pooler_output.to(device)

            image_encoded = self._image_processor(image_evidence, return_tensors="pt").to(device)
            image_feature = self._vision_model(**image_encoded).last_hidden_state.mean(dim=1).to(device)
            # image_feature = self._vision_model(**image_encoded).pooler_output.to(device)

            text_feature = torch.mean(text_feature, 0, keepdim=True)
            image_feature = torch.mean(image_feature, 0, keepdim=True)

            # text_feature = torch.sum(text_feature, 0, keepdim=True)
            # image_feature = torch.mean(image_feature, 0, keepdim=True)

            Hc.append(claim_f)
            Ht.append(text_feature)
            Hm.append(image_feature)

        Hc = torch.cat(Hc)
        Ht = torch.cat(Ht)
        Hm = torch.cat(Hm)

        if Lb:
            Lb = torch.stack(Lb)

        text_evidence_features = Ht
        image_evidence_features = Hm

        attention_claim_text, _ = self.text_attention(Hc, text_evidence_features, text_evidence_features)
        attention_claim_img, _ = self.image_attention(Hc, image_evidence_features, image_evidence_features)

        fused_text = self.leaky_relu(self.fc1(torch.cat([attention_claim_text * Hc, attention_claim_text - Hc], 1)))
        fused_img = self.leaky_relu(self.fc1(torch.cat([attention_claim_img * Hc, attention_claim_img - Hc], 1)))

        claim_out = self.fc_claim(Hc)
        claim_out = self.softmax(claim_out)

        # Conv modules
        c1_t = F.relu(self.conv(fused_text.T).T)
        c2_t = F.relu(self.conv2(fused_text.T).T)
        c3_t = F.relu(self.conv3(fused_text.T).T)
        conv_t = torch.cat([c1_t, c2_t, c3_t], 1).to(device)

        c1_i = F.relu(self.conv(fused_img.T).T)
        c2_i = F.relu(self.conv2(fused_img.T).T)
        c3_i = F.relu(self.conv2(fused_img.T).T)
        conv_i = torch.cat([c1_i, c2_i, c3_i], 1).to(device)

        combine = torch.cat([conv_t, conv_i], 1).to(device)
        combine = self.pool(combine)
        claim_evidence_out = self.fc_evidence(combine)
        claim_evidence_out = self.dropout(claim_evidence_out)
        claim_evidence_out = self.softmax(claim_evidence_out)
        # end conv module

        # Full
        out = torch.mean(torch.stack([claim_out, claim_evidence_out], 0), 0).to(device)

        return out, Lb

class MultiModalClassification(nn.Module):
    def vision_model(self, type='vit'):
        if type == 'vit':
            processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")
            model = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k")
        if type == 'beit':
            processor = BeitImageProcessor.from_pretrained("microsoft/beit-base-patch16-224-pt22k")
            model = BeitModel.from_pretrained("microsoft/beit-base-patch16-224-pt22k")
        if type == 'deit':
            processor = DeiTImageProcessor.from_pretrained("facebook/deit-base-distilled-patch16-224")
            model = DeiTModel.from_pretrained("facebook/deit-base-distilled-patch16-224")

        model.requires_grad_(False)
        return processor, model

    def text_model(self, pt="roberta-base"):
        processor = AutoTokenizer.from_pretrained(pt)
        model = AutoModel.from_pretrained(pt)
        print(pt)
        return processor, model


    def text_model_long(self, pt="longformer"):
        if pt == 'longformer':
            processor = LongformerTokenizer.from_pretrained("allenai/longformer-base-4096")
            model = LongformerModel.from_pretrained("allenai/longformer-base-4096")
        if pt == 'bigbird':
            model = BigBirdModel.from_pretrained("google/bigbird-roberta-base")
            processor = BigBirdTokenizer.from_pretrained("google/bigbird-roberta-base")

        model.requires_grad_(False)
        return processor, model


    def __init__(self, device, claim_pt="roberta-base", vision_pt='vit', long_pt="longformer"):
        super(MultiModalClassification, self).__init__()
        self._claim_pt = claim_pt
        self._vision_pt = vision_pt
        self._long_pt = long_pt
        self.text_attention = nn.MultiheadAttention(embed_dim=768, num_heads=4, vdim=768, kdim=768)
        self.image_attention = nn.MultiheadAttention(embed_dim=768, num_heads=4, vdim=768, kdim=768)
        self._text_processor, self._text_model = self.text_model(self._claim_pt)
        self._long_text_processor, self._long_text_model = self.text_model_long(self._long_pt)
        self._image_processor, self._vision_model = self.vision_model(self._vision_pt)
        self._device = device

        self.conv = nn.Conv1d(768, 100, stride=1, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(768, 100, stride=1, kernel_size=5, padding=2)
        self.conv3 = nn.Conv1d(768, 100, stride=1, kernel_size=7, padding=3)
        self.fc1 = nn.Linear(768 * 2, 768)

        self.pool = nn.MaxPool1d(2, 2)
        self.softmax = nn.Softmax(dim=1)
        self.leaky_relu = nn.LeakyReLU()

        self.fc_claim = nn.Linear(768, 4)
        self.fc_evidence = nn.Linear(300, 4)

        self.dropout = nn.Dropout(0.2)

    def forward(self, claim_features, label=None):
        device = self._device
        self._vision_model.to(device)
        self._text_model.to(device)
        self._long_text_model.to(device)

        Hc = []
        Ht = []
        Hm = []
        Lb = []

        if label is not None:
            label = label
            for l in label:
                Lb.append(l)

        for claim_feature in claim_features:
            claim = claim_feature['claim']
            text_evidence = claim_feature['text_evidence']
            image_evidence = [Image.open(simg).convert('RGB') for simg in claim_feature['image_evidence']]
            # if len(text_evidence) == 0:
            #     text_evidence.append("")
            # if len(image_evidence) == 0:
            #     blank_img = np.zeros((50, 50, 3), np.uint8)
            #     blank_img.fill(255)
            #     image_evidence.append(blank_img)

            claim_encoded = self._text_processor(claim, return_tensors="pt", padding=True, truncation=True,
                                                 max_length=100).to(device)
            claim_f = self._text_model(**claim_encoded).last_hidden_state.mean(dim=1).to(device)
            # claim_f = self._text_model(**claim_encoded).pooler_output.to(device)

            text_encoded = self._long_text_processor(text_evidence, return_tensors="pt", padding=True,
                                                     truncation=True).to(device)
            text_feature = self._long_text_model(**text_encoded).last_hidden_state.mean(dim=1).to(device)
            # text_feature = self._long_text_model(**text_encoded).pooler_output.to(device)

            image_encoded = self._image_processor(image_evidence, return_tensors="pt").to(device)
            image_feature = self._vision_model(**image_encoded).last_hidden_state.mean(dim=1).to(device)
            # image_feature = self._vision_model(**image_encoded).pooler_output.to(device)

            text_feature = torch.mean(text_feature, 0, keepdim=True)
            image_feature = torch.mean(image_feature, 0, keepdim=True)

            # text_feature = torch.sum(text_feature, 0, keepdim=True)
            # image_feature = torch.mean(image_feature, 0, keepdim=True)

            Hc.append(claim_f)
            Ht.append(text_feature)
            Hm.append(image_feature)

        Hc = torch.cat(Hc)
        Ht = torch.cat(Ht)
        Hm = torch.cat(Hm)

        if Lb:
            Lb = torch.stack(Lb)

        text_evidence_features = Ht
        image_evidence_features = Hm

        attention_claim_text, _ = self.text_attention(Hc, text_evidence_features, text_evidence_features)
        attention_claim_img, _ = self.image_attention(Hc, image_evidence_features, image_evidence_features)

        fused_text = self.leaky_relu(self.fc1(torch.cat([attention_claim_text * Hc, attention_claim_text - Hc], 1)))
        fused_img = self.leaky_relu(self.fc1(torch.cat([attention_claim_img * Hc, attention_claim_img - Hc], 1)))

        claim_out = self.fc_claim(Hc)
        claim_out = self.softmax(claim_out)

        # Conv modules
        c1_t = F.relu(self.conv(fused_text.T).T)
        c2_t = F.relu(self.conv2(fused_text.T).T)
        c3_t = F.relu(self.conv3(fused_text.T).T)
        conv_t = torch.cat([c1_t, c2_t, c3_t], 1).to(device)

        c1_i = F.relu(self.conv(fused_img.T).T)
        c2_i = F.relu(self.conv2(fused_img.T).T)
        c3_i = F.relu(self.conv2(fused_img.T).T)
        conv_i = torch.cat([c1_i, c2_i, c3_i], 1).to(device)

        combine = torch.cat([conv_t, conv_i], 1).to(device)
        combine = self.pool(combine)
        claim_evidence_out = self.fc_evidence(combine)
        claim_evidence_out = self.dropout(claim_evidence_out)
        claim_evidence_out = self.softmax(claim_evidence_out)
        # end conv module

        # Full
        out = torch.mean(torch.stack([claim_out, claim_evidence_out], 0), 0).to(device)

        return out, Lb
    

def one_hot(a, num_classes):
    v = np.zeros(num_classes, dtype=int)
    v[a] = 1
    return v


def encode_one_sample(sample, images_db):
    claim = sample['claim']
    text_evidence = sample['context']
    image_evidence = images_db.loc[(images_db.id == sample['image_id'])]['image'].values
    label = sample['correct_answer']

    label2idx = {
        'True': 3,
        'Partially True': 2,
        'False': 1,
        'Not Verifiable': 0
    }

    encoded_sample = {}
    encoded_sample["claim"] = claim
    encoded_sample["label"] = torch.tensor(one_hot(label2idx[label], 4), dtype=float)
    encoded_sample['text_evidence'] = [text_evidence]
    encoded_sample['image_evidence'] = image_evidence.tolist()

    return encoded_sample

class MCVEClaimVerificationDataset(torch.utils.data.Dataset):
    def __init__(self, claim_verification_data, images_db):
        self._data = claim_verification_data
        # self._processor = processor

        self._encoded = []
        for d in self._data:
            self._encoded.append(encode_one_sample(d, images_db))

    def __len__(self):
        return len(self._encoded)

    def __getitem__(self, idx):
        return self._encoded[idx]

    def to_list(self):
        return self._encoded


def make_batch(train_data, batch_size=128, shuffle=True):
    claim_labels = []
    claim_features = []

    if shuffle:
        train_data = train_data.to_list() if not isinstance(train_data, list) else train_data
        random.shuffle(train_data)

    for d in train_data:
        claim_labels.append(d['label'])
        claim_features.append(d)

    num_batches = math.ceil(len(train_data) / batch_size)
    train_features_batch = [claim_features[batch_size * y:batch_size * (y + 1)] for y in range(num_batches)]
    # train_label_batch = [torch.cat(claim_labels[batch_size * y: batch_size * (y + 1)], out=torch.Tensor(len(claim_labels[batch_size * y:batch_size * (y + 1)]), 1, 3).to(device)) for y in range(num_batches)]
    train_label_batch = [claim_labels[batch_size * y: batch_size * (y + 1)] for y in range(num_batches)]

    return train_features_batch, train_label_batch



class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        bce_loss = F.binary_cross_entropy(inputs.squeeze(), targets.float())
        loss = self.alpha * (1 - torch.exp(-bce_loss)) ** self.gamma * bce_loss
        return loss


def train_model(train_data, batch_size, epoch=1, save_checkpoint=True, is_val=False, val_data=None, claim_pt="roberta-base", vision_pt='vit',
                long_pt="longformer", device=None):
    model = MultiModalClassification(device, claim_pt, vision_pt, long_pt)
    # model = MultiModalClassificationLarge(device, claim_pt, vision_pt, long_pt)
    model = model.to(device)
    loss_function = FocalLoss(gamma=2)

    loss_function = loss_function.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=0.0001)
    loss_vals = []

    print('Training.......')
    best_model = model
    best_acc = 0

    chk_dir = "./model_dump/model_verification_{}_{}_{}_{}".format(
        str(claim_pt),
        str(long_pt),
        str(vision_pt),
        str(datetime.datetime.now().strftime("%d-%m_%H-%M"))
    )
    os.makedirs(chk_dir)
    os.makedirs("{}/checkpoint".format(chk_dir))

    X, y = make_batch(train_data, batch_size=batch_size)
    for e in trange(epoch):
        model.train()
        total_loss = 0
        print("Epoch {}:\n ".format(e + 1))

        for i in trange(len(X)):
            optimizer.zero_grad()
            batch_x = X[i]
            score, lb = model(batch_x, y[i])
            loss = loss_function(score.to(device), lb.to(device))
            loss.backward()

            optimizer.step()
            total_loss = total_loss + loss.item()

        loss_vals.append(total_loss)
        print("Loss: {}\n".format(total_loss))

        if is_val and val_data:
            truelb, predlb = predict(val_data, model, batch_size=batch_size)
            mif1 = f1_score(truelb, predlb, average='micro')
            if mif1 > best_acc:
                best_acc = copy.deepcopy(mif1)
                best_model = copy.deepcopy(model)
            print("Macro F1-score: {}\n".format(f1_score(truelb, predlb, average='macro')))
            print("F1-score: {}\n".format(f1_score(truelb, predlb, average='micro')))
        else:
            best_model = copy.deepcopy(model)
        print('===========\n')

        if save_checkpoint:
            torch.save({
                'total_epochs': epoch,
                'current_epoch': e,
                'batch_size': batch_size,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()
            }, "{}/checkpoint/checkpoint_{}.pt".format(
                str(chk_dir),
                str(e))
            )

    torch.save({
            'total_epochs': epoch,
            'current_epoch': e,
            'batch_size': batch_size,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()
    }, "{}/checkpoint/checkpoint_{}.pt".format(
            str(chk_dir),
            str(e))
    )
    torch.save(best_model, '{}/best_model.pt'.format(chk_dir))

    return best_model, loss_vals, claim_pt


def train_resume(train_data, chkpoint, save_checkpoint=True, is_val=False, val_data=None, claim_pt="roberta-base",
                 vision_pt='vit', long_pt="longformer", device=None):
    model = MultiModalClassification(device, claim_pt, vision_pt, long_pt)
    # model = MultiModalClassificationLarge(device, claim_pt, vision_pt, long_pt)
    model.load_state_dict(chkpoint['model_state_dict'])
    model = model.to(device)
    print(model)
    loss_function = FocalLoss(gamma=2)
    loss_function = loss_function.to(device)
    epoch = chkpoint['total_epochs']
    batch_size = chkpoint['batch_size']

    optimizer = optim.AdamW(model.parameters(), lr=0.0001)
    optimizer.load_state_dict(chkpoint['optimizer_state_dict'])
    loss_vals = []

    print('Training.......')
    best_model = model
    best_acc = 0
    X, y, = make_batch(train_data, batch_size=batch_size)

    chk_dir = "model_dump/model_verification_{}_{}_{}_{}".format(
        str(claim_pt),
        str(long_pt),
        str(vision_pt),
        str(datetime.datetime.now().strftime("%d-%m_%H-%M"))
    )
    os.makedirs(chk_dir)
    os.makedirs("{}/checkpoint".format(chk_dir))

    for e in trange(chkpoint['current_epoch'], epoch):
        model.train()
        total_loss = 0
        print("Epoch {}:\n ".format(e + 1))

        for i in trange(len(X)):
            optimizer.zero_grad()
            batch_x = X[i]
            score, lb = model(batch_x, y[i])
            loss = loss_function(score.to(device), lb.to(device))
            loss.backward()

            optimizer.step()
            total_loss = total_loss + loss.item()

        loss_vals.append(total_loss)
        print("Loss: {}\n".format(total_loss))

        if is_val and val_data:
            truelb, predlb = predict(val_data, model, batch_size=batch_size)
            mif1 = f1_score(truelb, predlb, average='micro')
            if mif1 > best_acc:
                best_acc = copy.deepcopy(mif1)
                best_model = copy.deepcopy(model)
            print("Macro F1-score: {}\n".format(f1_score(truelb, predlb, average='macro')))
            print("F1-score: {}\n".format(f1_score(truelb, predlb, average='micro')))
        else:
            best_model = copy.deepcopy(model)
        print('===========\n')

        if save_checkpoint:
            torch.save({
                'total_epochs': epoch,
                'current_epoch': e,
                'batch_size': batch_size,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()
            }, "{}/checkpoint/checkpoint_{}.pt".format(
                str(chk_dir),
                str(e))
            )

    torch.save({
            'total_epochs': epoch,
            'current_epoch': e,
            'batch_size': batch_size,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()
    }, "{}/checkpoint/best_checkpoint_{}.pt".format(
        str(chk_dir),
        str(e))
    )
    torch.save(best_model, '{}/best_model.pt'.format(chk_dir))

    return best_model, loss_vals, claim_pt


def predict(test_data, model, batch_size, device=None):
    model = model.to(device)

    ground_truth = []
    predicts = []

    X, y = make_batch(test_data, batch_size=batch_size, shuffle=False)

    model.eval()
    print('Predict.......')

    for i in trange(len(X)):
        batch_x = X[i]
        batch_y = y[i]

        scores, lb = model(batch_x)
        scores = scores.reshape(-1, 4)

        if not ground_truth:
            ground_truth = [np.argmax(label.tolist(), -1) for label in batch_y]
        else:
            ground_truth.extend([np.argmax(label.tolist(), -1) for label in batch_y])

        if not predicts:
            predicts = [np.argmax(score.tolist(), -1) for score in scores]
        else:
            predicts.extend([np.argmax(score.tolist(), -1) for score in scores])

    return ground_truth, predicts


