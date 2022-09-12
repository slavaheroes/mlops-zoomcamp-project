# MLOps project

Hi all! I would like to present my project for [MLOps course](https://github.com/DataTalksClub/mlops-zoomcamp) organized by DataTalksClub.

If you are reviewer, please look at the [checklist](#checklist-for-peer-review) for convenience. 
  

## Project description

  

In this project, I used [Kaggle Fake News data](https://www.kaggle.com/competitions/fake-news/overview) to classify an article whether it's fake given the title, author and text.

I built a small web service using Flask that identifies fake news.

### Training process

**Preprocessing:**

Firstly, we must clean data like by removing words with little meaning like prepositions. For that, I utilized the stop words list in [nltk](https://www.nltk.org/) library. Also, I used [stemming](https://www.nltk.org/howto/stem.html) to reduce word to its root.

  

Since we have a text data, we must represent a document as a numerical data. To do so, I used [TfidVectorizer in sklearn](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html) library because it converts a text into a matrix of tf-idf features.

  

**Model and training:**

For training I selected **Logistic Regression** as method to classify the data. In this project, I used a model which is already implemented in sklearn.

I trained the model using different hyperparameters and KFold, and logged the metrics to **MLFlow**. Best model was selected according to the F1 score.

For basic orchestration I used **Prefect 2.3.1**, but it is not fully deployed.

  

### Deployment and Monitoring

  

I built a simple web service on Flask that listens on port 9000. It is containerized by **Docker** and deployed locally.

It supports two simple routes:

 1. **"/"**: On *"GET"* request, you can access a form where you can write your own test case. If you press "Submit" button on this form,  *"POST"* request will be sent to the same route, where model will make a prediction using preprocessors and model. Prediction consists of predicted label, and probability of it. Also, data and prediction will be stored locally for monitoring.

 2.  **"/monitor"**: This page is designed as a simple monitoring service using **evidently**. After gathering enough data, the dashboard with DataDriftTab will be created to analyze the prediction labels and their probability. The report will be saved as ***a html*** file, then can be viewed on *"GET"* request.

  

## Reproducibility

1. Download a data from [kaggle](https://www.kaggle.com/competitions/fake-news/data) or [google drive](https://drive.google.com/drive/folders/1m_LH5qdr68ML-t83M2BIdZ7zVs4qSvcK?usp=sharing), then place **train.csv** and **test.csv** into `data/` folder.
2. Make sure you have **pipenv** installed. Then, on the project folder, enter `pipenv install` and `pipenv shell`  to install packages in *Pipfile* and activate the working environment.
3. Run MLFlow and Prefect on different terminal windows using commands below, and make sure they won't be shut down during the training. 
> Please use the commands below for correct reproducibility because mlflow is not operating on the usual 5000th port but on 6677th.

     make mlflow
     make prefect
	 
4. Enter `make train` command. You might open **mlflow** and **prefect** on localhost to view training logs, parameteres, etc. 
	> Reminder: MLFlow is operating on 6677 port of localhost.

5. At this point, we want to deploy our model. Type `make deploy` . It will build a docker image and run docker container that will operate the **Flask** server described above.
6. After that, I want to simulate traffic by sending test cases from **`data/test.csv`**  by one to the server. To do so, type `make simulate`
7. Now you can check `localhost:9000` and `localhost:9000/monitor` to see the simple form for POST request and **evidently report** respectively.


## Checklist for peer review

 - Cloud is not used. All things run locally.
 - For experiment tracking and registry, I used **MLFlow** and for each experiment I log metrics, model artifact. After train runs finish, best run with the highest F1 score is registered, and it's weight is saved locally in artifacts folder. Also, previous model is moved to "Staging". For the details check **train.py**
 - I used basic **Prefect** flows and tasks for model orchestration, but it  is not deployed.
 - For deployment, I used **Flask** server and **Docker**.  It uses locally saved models, and writes predictions in the same folder. For details, check the **Dockerfile** and **deploy_model.py**
 - For monitoring, I used **evidently** to analyze data drift. It has a basic functionality, so it just calculates metrics and saves report in html format so we can see it in the browser. For details, take a look at  **deploy_model.py** 
 - [Reproducibility](#reproducibility). If you have some problems with the instructions, contact me on [telegram](https://t.me/slavaheroes) or on slack of DataTalksClub (@Slava Shen).
- Best practices:
	- [x] Unit tests
	- [ ] Integration tests
	- [x] Linter or Code formatter
	- [x] Makefile
	- [x] Pre-commit hooks
	- [ ] CI/CD


### Updated on September 12, 2022.
