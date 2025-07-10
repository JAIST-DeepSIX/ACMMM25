import json

def load_json(json_path):
    """
        Read data json file
        Output a list of samples, each sample is a dict
    """
    with open(json_path, "r") as f:
        data = json.load(f)

    return data