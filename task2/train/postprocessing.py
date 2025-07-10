from sklearn.metrics import classification_report
import evaluate
import numpy as np

def make_report(y_pre, y_true, save_dir):
    report = classification_report(y_true=y_true, y_pred=y_pre, digits=4)
    print(report)
    with open(save_dir + "/note.txt", "w", encoding="utf-8") as f:
        f.write(report)

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    accuracy = evaluate.load("accuracy")
    return accuracy.compute(predictions=predictions, references=labels)