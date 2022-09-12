import pickle
import os
import nltk
nltk.download('stopwords')


ARTIFACT_PATH = "artifacts"

def save_artifact(artifact, name):
    os.makedirs(ARTIFACT_PATH, exist_ok=True)
    path = os.path.join(ARTIFACT_PATH, name)
    with open(path, "wb") as file:
        pickle.dump(artifact, file)
    return path

def load_artifact(path):
    with open(path, 'rb') as f:
        artifact = pickle.load(f)
    return artifact