import json
import plotly
import pandas as pd
import re
import joblib
import numpy as np
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request
from plotly.graph_objs import Bar

from sqlalchemy import create_engine

# regex to support the X_transform function
url_re = '[a-zA-Z]*http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+[a-zA-Z]*'
email_re = '[a-zA-Z_0-9-+]+@(.?[a-zA-Z_0-9-]+\.[a-zA-Z_0-9-]+)+$'
url = re.compile(url_re)
eml = re.compile(email_re)

app = Flask(__name__)

def X_Transform(X):
    '''
    
    transform the message to 4 features columns
    
    inputs:
    
    X : a message
       
    outputs:

    array of containing and 4 columns of :-

    message : message text without numbers , emails and url
    number : 0 or 1 if the text has a number
    email : 0 or 1 if the text has an email
    url : 0 or 1 if the text has an url

    
    '''

    Msg = np.zeros((1,4),dtype=object)
    Msg[0,0] = X

    if url.search(X):
        Msg[0,3] = 1   
        text = url.sub(" ",X)
        Msg[0,0] = text    
    if eml.search(X):
        Msg[0,2] = 1   
        text = eml.sub(" ",X)
        Msg[0,0] = text        
    if re.search(r'[0-9]+',X):
        Msg[0,1] = 1   
        text = re.sub(r'[a-zA-Z]*[0-9]+[a-zA-Z]*'," ",X)
        Msg[0,0] = text             
            
    return Msg     

def tokenize(text):
    '''
    
    applying NLP process on the text
    
    inputs:
    
    text: String
       
    outputs:

    list of cleaned text
    
    '''
    # nothing but the alphapets

    text = text.lower()
    text = re.sub(r"[^a-z]+"," ",text)
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    # for loop to stem the word

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).strip()
        if tok == clean_tok :
           clean_tok = lemmatizer.lemmatize(tok,'v').strip()
        if tok == clean_tok :
           clean_tok = lemmatizer.lemmatize(tok,'a').strip()
        if tok == clean_tok :
           clean_tok = lemmatizer.lemmatize(tok,'r').strip()
        if tok == clean_tok :
           clean_tok = lemmatizer.lemmatize(tok,'s').strip()
        clean_tokens.append(clean_tok)
        
    clean_tokens = [w for w in clean_tokens if (len(w) > 2)]
    return clean_tokens

# load data
engine = create_engine('sqlite:///data/DisasterResponse.db')
df = pd.read_sql_table('message', engine)

# load model
model = joblib.load("./models/classifier.pkl")

# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')

def index():
    # create visuals
    # first graph - count of every category
    fire = df.loc[df['fire']==1,'fire'].shape[0]
    water = df.loc[df['water']==1,'water'].shape[0]
    storm = df.loc[df['storm']==1,'storm'].shape[0]
    FWS = ['Fire','Water','Storm']
    CountFWS = [fire,water,storm]
    #second graph
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)
    # create visuals
    graphs = [
        {
            'data': [
                Bar(
                    x=FWS,
                    y=CountFWS,
                    marker_color= ['#F79C4C','#17ABE5','#BBBA95']
                )
            ],

            'layout': {
                'title': 'Distribution of Fire , Water and Strom Categories',
                'yaxis': {'title': 'Count'},
                'xaxis': {'title': 'Category'}
            }
        },{
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {
                    'title': 'Count'
                },
                'xaxis': {
                    'title': 'Genre'
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
    query = X_Transform(query)
    classification_labels = model.predict(query)[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query[0,0],
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()