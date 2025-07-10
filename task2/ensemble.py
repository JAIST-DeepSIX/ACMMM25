import pandas as pd
import numpy as np
import json
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib as plt

def ensemble(lst_model, data):
    results = []
    for i in range(0, len(data)):
        oh = [0, 0, 0, 0]
        for m in lst_model:
            oh[m[i]] = oh[m[i]] + 1
        results.append(np.argmax(oh, axis=-1).item())
    
    return results

if __name__ == '__main__':
    label2idx = {
        'A': 3,
        'B': 1,
        'C': 2,
        'D': 0
    }

    ground = []
    with open("./results/dev/dev_qwen2.5_vl_summary.json", "r") as f:
        gold = json.load(f)
        for r in gold:
            ground.append(label2idx[r['label']])
    f.close()

    # model1 = []
    # with open("./qwen2.5-14-fc.json") as f:
    #     data = json.load(f)
    #     for d in data:
    #         model1.append(label2idx[d['Pred']])
    # f.close()

    # print("model1")
    # print("Test result micro: {}\n".format(f1_score(ground, model1, average='micro')))
    # print("Test result macro: {}\n".format(f1_score(ground, model1, average='macro')))
    # print(confusion_matrix(ground, model1, labels=[0, 1, 2, 3]))
    # print("--------")
    
    # model2 = []
    # with open("./qwen2.5-14-f30.json") as f:
    #     data = json.load(f)
    #     for d in data:
    #         model2.append(label2idx[d['predict']])
    # f.close()

    # print("model2")
    # print("Test result micro: {}\n".format(f1_score(ground, model2, average='micro')))
    # print("Test result macro: {}\n".format(f1_score(ground, model2, average='macro')))
    # print(confusion_matrix(ground, model2, labels=[0, 1, 2, 3]))
    # print("--------")
    
    # model3 = []
    # with open("./qwen2.5-32.json") as f:
    #     data = json.load(f)
    #     for d in data:
    #         model3.append(label2idx[d['predict']])
    # f.close()

    # print("model3")
    # print("Test result micro: {}\n".format(f1_score(ground, model3, average='micro')))
    # print("Test result macro: {}\n".format(f1_score(ground, model3, average='macro')))
    # print(confusion_matrix(ground, model3, labels=[0, 1, 2, 3]))
    # print("--------")
    
    model3 = []
    with open("./results/dev/qwen14-fc-dev.json", "r") as f:
        data = json.load(f)
        for d in data:
            model3.append(label2idx[d['Pred']])
    f.close()
    print("model3")
    print("Test result micro: {}\n".format(f1_score(ground, model3, average='micro')))
    print("Test result macro: {}\n".format(f1_score(ground, model3, average='macro')))
    print(confusion_matrix(ground, model3, labels=[0, 1, 2, 3]))
    print("--------")
    
    model4 = []
    with open("./results/dev/qwen72-fc-dev.json", "r") as f:
        data = json.load(f)
        for d in data:
            model4.append(label2idx[d['Pred']])
    f.close()

    print("model4")
    print("Test result micro: {}\n".format(f1_score(ground, model4, average='micro')))
    print("Test result macro: {}\n".format(f1_score(ground, model4, average='macro')))
    print(confusion_matrix(ground, model4, labels=[0, 1, 2, 3]))
    print("--------")

    model5 = []
    with open("./results/dev/qwen2.5VL-32-dev.json", "r") as f:
        data = json.load(f)
        for d in data:
            model5.append(label2idx[d['predict']])
    f.close()

    print("model5")
    print("Test result micro: {}\n".format(f1_score(ground, model5, average='micro')))
    print("Test result macro: {}\n".format(f1_score(ground, model5, average='macro')))
    print(confusion_matrix(ground, model5, labels=[0, 1, 2, 3]))
    print("--------")

    model6 = pd.read_csv("./results/dev/MCVE_dev.csv")['predict']
    print("model6")
    print("Test result micro: {}\n".format(f1_score(ground, model6, average='micro')))
    print("Test result macro: {}\n".format(f1_score(ground, model6, average='macro')))
    print(confusion_matrix(ground, model6, labels=[0, 1, 2, 3]))
    print("--------")

    model7 = []
    with open("./results/dev/qwen2.5VL-72-dev.json", "r") as f:
        data = json.load(f)
        for d in data:
            model7.append(label2idx[d['predict']])
    f.close()
    
    print("===========")

    results = ensemble([model3, model4, model5, model6, model7], gold)
    # results = ensemble([model5, model7], gold)
    print("Test result micro: {}\n".format(f1_score(ground, results, average='micro')))
    print("Test result macro: {}\n".format(f1_score(ground, results, average='macro')))
    print(confusion_matrix(ground, results, labels=[0, 1, 2, 3]))

    # with open("/home/sonlt/drive/data/acmm25/fact_checking/json/test.json") as f:
    #     test_raw = json.load(f)
    # f.close()

    # assert len(test_raw) == len(results)
    idx2label = {
        3: 'A',
        1: 'B',
        2: 'C',
        0: 'D'
    }
    # for i in range(0, len(test_raw)):
    #     test_raw[i]['predict_label'] = idx2label[results[i]]
    
    # with open('./submission.json', 'w', encoding='utf-8') as f:
    #     json.dump(test_raw, f, ensure_ascii=False, indent=4)
    # f.close()
    new_ground = [idx2label[o] for o in ground]
    new_result = [idx2label[o] for o in results]
    
    cm = confusion_matrix(new_ground, new_result, labels=['A', 'B', 'C', 'D'])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['A', 'B', 'C', 'D'])
    
    disp.plot(cmap=plt.cm.Blues).figure_.savefig('confusion_matrix.png')
