from nltk.stem.porter import PorterStemmer

import deploy_model


def test_stemming():
    stemmer = PorterStemmer()
    content = "David Streitfeld Specter of Trump Loosens Tongues, if Not Purse Strings, in Silicon Valley - The New York Times"
    output = deploy_model.stemming(content, stemmer)
    expected_output = "david streitfeld specter trump loosen tongu purs string silicon valley new york time"
    assert output == expected_output


def test_make_prediction():
    test_data = {
        "author": "David Streitfeld",
        "title": "Specter of Trump Loosens Tongues, if Not Purse Strings, in Silicon Valley - The New York Times",
        "text": "no text",
    }

    response = deploy_model.make_prediction(test_data)
    # check presence of keys
    assert "prediction" in response
    assert "probability" in response
