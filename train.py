import argparse
import numpy as np
import pandas as pd
import re
from datetime import datetime
import os
import pickle
import yaml

import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from datetime import timedelta

from hyperopt import STATUS_OK, Trials, fmin, hp, tpe
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

from utils import config, save_artifact

ARTIFACT_PATH = config['ARTIFACT_PATH']
EXP_NAME = config['EXP_NAME']
MODEL_NAME = config['MODEL_NAME']

import mlflow
from mlflow.tracking import MlflowClient
from mlflow.entities import ViewType

mlflow.set_tracking_uri("http://localhost:6677")
mlflow.set_experiment(EXP_NAME)

from prefect import flow, task, get_run_logger
from prefect.task_runners import SequentialTaskRunner
from prefect.deployments import Deployment
from prefect.orion.schemas.schedules import IntervalSchedule



def load_data(dataset):
    news_dataset = dataset.fillna('')
    news_dataset['content'] = news_dataset['author']+' '+news_dataset['title']
    return news_dataset

def stemming(content, port_stem):
    stemmed_content = re.sub('[^a-zA-Z]',' ',content)
    stemmed_content = stemmed_content.lower()
    stemmed_content = stemmed_content.split()
    stemmed_content = [port_stem.stem(word) for word in stemmed_content if not word in stopwords.words('english')]
    stemmed_content = ' '.join(stemmed_content)
    return stemmed_content


def preprocess(dataset):
    port_stem = PorterStemmer()
    dataset['content'] = dataset['content'].apply(stemming, args=(port_stem, ))
    X_data = dataset['content'].values
    Y_labels = dataset['label'].values
    vectorizer = TfidfVectorizer()  
    X_data = vectorizer.fit_transform(X_data)

    ## save artifacts locally ##
    save_artifact(port_stem, "PortStemmer.pickle")
    save_artifact(vectorizer, "vectorizer.pickle")

    return X_data, Y_labels

def run(train_path: str, k_fold:int):
    news_dataset = pd.read_csv(train_path)
    news_dataset = load_data(news_dataset)
    X_data, Y_labels = preprocess(news_dataset)
    kf = StratifiedKFold(n_splits=k_fold)

    for train_index, test_index in kf.split(X_data, Y_labels):
        X_train, X_test = X_data[train_index], X_data[test_index]
        Y_train, Y_test = Y_labels[train_index], Y_labels[test_index]
        
        # training and logging function
        @task
        def objective(params):
            logger = get_run_logger()
            with mlflow.start_run() as run:
                model = LogisticRegression(**params)

                mlflow.log_params(params)

                model.fit(X_train, Y_train)
                X_test_prediction = model.predict(X_test)
                test_data_accuracy = accuracy_score(X_test_prediction, Y_test)
                test_data_f1 = f1_score(X_test_prediction, Y_test)

                mlflow.log_metric("accuracy", test_data_accuracy)
                mlflow.log_metric("f1_score", test_data_f1)
                mlflow.log_artifact(save_artifact(model, "model.pickle"))
                logger.info(f'F1 score {test_data_f1}')

            # to choose maximum f1
            return {'loss': -1*test_data_f1, "status": STATUS_OK}
        
        # Hyperparameter search
        search_space = {
            'C': hp.lognormal('LR_C', 0, 1.0),
            'solver': hp.choice('solver', ['liblinear', 'lbfgs']),
            'random_state': 100
        }

        rstate = np.random.default_rng(42)  # for reproducible results

        fmin(
            fn=objective,
            space=search_space,
            algo=tpe.suggest,
            max_evals=2,
            trials=Trials(),
            rstate=rstate,
            verbose = True
        )

@task
def register_best_model():
    logger = get_run_logger()
    client = MlflowClient()
    experiment = client.get_experiment_by_name(EXP_NAME)
    best_run = client.search_runs(
        experiment_ids=experiment.experiment_id,
        run_view_type=ViewType.ACTIVE_ONLY,
        max_results=1,
        order_by=["metrics.f1_score DESC"])[0]

    # register the best model
    logger.info(best_run.info)
    mlflow.register_model( model_uri = f"runs:/{best_run.info.run_id}/model", name = MODEL_NAME)

    latest_versions = client.get_latest_versions(name=MODEL_NAME)
    model_version = len(latest_versions)
    # Move the old model to the "Staging"
    new_stage = "Staging"
    client.transition_model_version_stage(
        name=MODEL_NAME,
        version=model_version,
        stage=new_stage,
        archive_existing_versions=False
    )

    date = datetime.today().date()
    client.update_model_version(
        name=MODEL_NAME,
        version=model_version,
        description=f"The model version {model_version} was transitioned to {new_stage} on {date}"
    )

    # save the model locally
    mlflow.artifacts.download_artifacts(
        artifact_uri=f"runs:/{best_run.info.run_id}/model.pickle", dst_path=ARTIFACT_PATH
    )

@flow 
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train_path",
        default="./data/train.csv",
        help="the location where you've downloaded train.csv file. Read data/read_me.txt"
    )
    parser.add_argument(
        "--k_folds",
        default=3,
        help="the number of different folds to train data"
    )
    args = parser.parse_args()
    run(args.train_path, args.k_folds)
    register_best_model()

# deployment = Deployment.build_from_flow(
#     flow=main,
#     name="model_training",
#     schedule=IntervalSchedule(interval=timedelta(minutes=5)),
#     work_queue_name="ml"
# )
# deployment.apply()

if __name__ == "__main__":
    main()