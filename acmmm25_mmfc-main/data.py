import glob
import pandas as pd
import json 

def read_image_path_only(path):
    ext = ['jpg', 'jpeg', 'png']

    files = []
    images = []
    names = []
    for e in ext:
        for img in glob.glob(path + "/*." + e):
            files.append(img)

    for f in files:
        names.append(f.split('/')[-1])
        images.append(f)

    return pd.DataFrame({
        'id': names,
        'image': images
    })

def read_data(path):
    with open(path + "/train.json", "r") as f:
        train = json.load(f)
    f.close()

    with open(path + "/val.json", "r") as f:
        dev = json.load(f)
    f.close()

    with open(path + "/test.json", "r") as f:
        test = json.load(f)
    f.close()

    return train, dev, test


if __name__ == '__main__':
    data = read_image_path_only("/home/sonlt/drive/data/acmm25/fact_checking/image")

    train, dev, test = read_data("/home/sonlt/drive/data/acmm25/fact_checking/json")
    print(len(train))
    print(len(dev))
    print(len(test))
