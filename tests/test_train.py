import pandas as pd
from nltk.stem.porter import PorterStemmer
import train

def test_load_data():
    df = pd.DataFrame([{
        "author": "David Streitfeld",
        "title": "Specter of Trump Loosens Tongues, if Not Purse Strings, in Silicon Valley - The New York Times",
        "text": "no text"
    }])

    output = train.load_data(df)
    expected_output = "David Streitfeld Specter of Trump Loosens Tongues, if Not Purse Strings, in Silicon Valley - The New York Times"
    assert output.iloc[0]['content'] == expected_output