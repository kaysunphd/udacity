import json
import plotly
import pandas as pd
import re

import nltk
nltk.download('stopwords')

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
from sklearn.externals import joblib
from sqlalchemy import create_engine

from collections import Counter


app = Flask(__name__)

url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'

def tokenize(text):
    """
    Tokenize text by
    1) finding and replacing all urls with "urlplaceholder"
    2) convert to lower case, substitution special characters with " ", tokenize
    3) lemmatize, lower case, strip
    4) append if not stop word in english
    
    Args:
    text: String of text.
    
    Return:
    clean_tokens: List of cleaned tokens.
    """
    
    # find and replace all urls with "urlplaceholder"
    detected_urls = re.findall(url_regex, text)
    
    for url in detected_urls:
        text = text.replace(url, "urlplaceholder")
    
    # convert to lower case, substitution special characters with " ", tokenize
    tokens = word_tokenize(re.sub(r"[^a-zA-Z0-9]", " ", text.lower()))
    
    # initalize lemmatizer
    lemmatizer = WordNetLemmatizer()
    
    # list of stop words in english
    stop_words = stopwords.words("english")

    # lemmatize, lower case, strip, and append if not stop word
    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        if clean_tok not in stop_words:
            clean_tokens.append(clean_tok)

    return clean_tokens

# load data
engine = create_engine('sqlite:///data/messages.db')
df = pd.read_sql_table('messages', engine)

# load model
model = joblib.load("models/classifier.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # extract data needed for visuals
    # calculate distribution of message genres
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)
    
    # calculate distribution of message categories
    categories_mean = df.iloc[:, -36:].mean().sort_values(ascending=False)
    categories_names = list(categories_mean.index)
    
    # calculate top words in messages
    # based on https://stackoverflow.com/questions/20510768/count-frequency-of-words-in-a-list-and-sort-by-frequency
    all_texts = ' '.join(df['message'])
    all_tokens = tokenize(all_texts)
    all_tokens_counts = len(all_tokens)
    tokens_counts = Counter(all_tokens).most_common()
    
    top_words = []
    top_words_count = []
    for token in tokens_counts:
        top_words.append(token[0])
        top_words_count.append(token[1] / all_tokens_counts * 100)
    
    # create visuals
    graphs = [
        # distribution of message genres
        {
            'data': [
                Bar(
                    x=categories_names,
                    y=genre_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        },
        # distribution of median message categories
        {
            'data': [
                Bar(
                    x=categories_names,
                    y=categories_mean
                )
            ],

            'layout': {
                'title': 'Distribution of Median Message Categories',
                'yaxis': {
                    'title': "Mean"
                },
                'xaxis': {
                    'tickangle':45,
                    'tickfont':{'size':10}
                }
            }
        },
        # distribution of frequently used words
        {
            'data': [
                Bar(
                    x=top_words[:10],
                    y=top_words_count[:10]
                )
            ],

            'layout': {
                'title': 'Distribution of 10 Most Frequently Used Words in Messages',
                'yaxis': {
                    'title': "Percentage"
                },
                'xaxis': {
                    'title': "Words"
                }
            }
        }
    ]
    
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '') 

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()