import pickle
import yaml
import os
import nltk
nltk.download('stopwords')


with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

def save_artifact(artifact, name):
    os.makedirs(config["ARTIFACT_PATH"], exist_ok=True)
    path = os.path.join(config["ARTIFACT_PATH"], name)
    with open(path, "wb") as file:
        pickle.dump(artifact, file)
    return path

def load_artifact(path):
    with open(path, 'rb') as f:
        artifact = pickle.load(f)
    return artifact