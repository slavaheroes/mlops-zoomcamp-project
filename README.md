# MLOps project
Hi all! I would like to present my project for [MLOps course](https://github.com/DataTalksClub/mlops-zoomcamp) organized by DataTalksClub.

## Project description

In this project, I used [Kaggle Fake News data](https://www.kaggle.com/competitions/fake-news/overview) to classify an article whether it's fake given the title, author and text. 
I built a small web service using Flask that identifies fake news. 
### Training process
**Preprocessing:**
Firstly, we must clean data like by removing words with little meaning like prepositions. For that, I utilized the stop words list in [nltk](https://www.nltk.org/) library. Also, I used [stemming](https://www.nltk.org/howto/stem.html) to reduce word to its root. 

Since we have a text data, we must represent a document as a numerical data. To do so, I used [TfidVectorizer in sklearn](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html) library because it converts a text into a matrix of tf-idf features. 

**Model and training:**
For training I selected Logistic Regression as method to classify the data. In this project, I used a model which is already implemented in sklearn. 
I trained the model using different hyperparameters and KFold, and logged the metrics to *MLFlow*. Best model was selected according to the F1 score.

### Deployment

I built a simple web service on Flask. ...

## Reproducibility

1. Download a data from [kaggle](https://www.kaggle.com/competitions/fake-news/data) or [google drive](https://drive.google.com/drive/folders/1m_LH5qdr68ML-t83M2BIdZ7zVs4qSvcK?usp=sharing), then place **train.csv** and **test.csv** into `data/` folder. 
