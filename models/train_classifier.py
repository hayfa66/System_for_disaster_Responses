import sys
import re
import pickle

import nltk
nltk.download(['punkt', 'wordnet','stopwords','omw-1.4'])
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

import pandas as pd
import numpy as np 

from sqlalchemy import create_engine

from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import classification_report

#---------------------
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.compose import ColumnTransformer

# SUPPORT X_transform function 
url_re = '[a-zA-Z]*http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+[a-zA-Z]*'
email_re = '[a-zA-Z_0-9-+]+@(.?[a-zA-Z_0-9-]+\.[a-zA-Z_0-9-]+)+$'
url = re.compile(url_re)
eml = re.compile(email_re)
# SUPPORT Toknize function 
stop = stopwords.words("english")
stop.append('us')
stop.append('pass')
stop.append('could')
stop.append('may')
stop.append('maybe')
stop.remove('in')
stop.remove('here')  




def X_Transform(X):
    '''
    
    transform the feature column to 4 features columns
    
    inputs:
    
    X : one column of the message
       
    outputs:

    2d array of containing and 4 columns of :-

    message : message text without numbers , emails and url
    number : 0 or 1 if the text has a number
    email : 0 or 1 if the text has an email
    url : 0 or 1 if the text has an url

    
    '''
    Array = np.zeros((len(X),4),dtype=object)
    for i,x in enumerate(X) :
        Array[i,0] = x
        if url.search(x):
            Array[i,3] = 1   
            text = url.sub(" ",x)
            Array[i,0] = text    
        if eml.search(x):
            Array[i,2] = 1   
            text = eml.sub(" ",x)
            Array[i,0] = text        
        if re.search(r'[0-9]+',x):
            Array[i,1] = 1   
            text = re.sub(r'[a-zA-Z]*[0-9]+[a-zA-Z]*'," ",x)
            Array[i,0] = text     
            
    return Array       

def load_data(database_filepath):
    '''
    Load data
    Load data from sql database and split it into features and categories and
    list of category names

    Input:
    database_filepath : filepath to where the database is

    Output :
    X : messages
    Y : categories values
    category_names : list of category names

    '''
    engine = create_engine('sqlite:///'+database_filepath)
    df = pd.read_sql_table('message', engine)

    category_names = list(df.iloc[:,4:].columns)
    Y = df.iloc[:,4:].values
    X = X_Transform(df['message'].values)

    return X , Y , category_names

def tokenize(text):
    '''
    
    applying NLP process on the text
    
    inputs:
    
    text: String
       
    outputs:

    list of cleaned text
    
    '''
    text = text.lower()
    text = re.sub(r"[^a-z]+"," ",text)
    
    tokens = word_tokenize(text)
    tokens = [w for w in tokens if w not in stop]
    lemmatizer = WordNetLemmatizer()

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
        
    clean_tokens = [w for w in clean_tokens if (len(w)>2) | (w in ['in','at'])]
    return clean_tokens


def build_model():
    '''
    
    Bulding a cv grid search
    
    inputs:

    None
       
    outputs:

    a cv grid search
    
    '''
    ct = ColumnTransformer( 
        [('CatTr', 
        Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer())
        ]),0)],
        remainder='passthrough') 

    pi = Pipeline([ ('cr', ct),  ('MltRFC',MultiOutputClassifier
                 (RandomForestClassifier())) ])

    parameters = {
    'cr__CatTr__vect__min_df': [0.001,0.002],
    'cr__CatTr__tfidf__smooth_idf': [True,False],
    'cr__CatTr__tfidf__use_idf': [True,False]
     }

    cv = GridSearchCV(pi, param_grid=parameters, cv=2, verbose=3)

    return cv

def evaluate_model(model, X_test, Y_test, category_names):
    '''
    
    Print the model Properties
    
    inputs:
    
    None
       
    outputs:

    None
    
    '''
    Y_pred = model.predict(X_test)

    for i in range(len(category_names)):
        print(category_names[i],'\n', classification_report(Y_test[:,i], 
        Y_pred[:,i]))
    
    accuracy=(Y_pred == Y_test).mean()
    print("Accuracy :-\n", accuracy.mean()*100 ,"%\n------")
    print(model.best_params_)
    
    
def save_model(model, model_filepath):
    '''
    Save Model
    Save The trained model as a pkl file

    Input:
    model : model to be saved
    model_filepath : string a filepath to where the model would be saved

    Output :
    None

    '''
    pickle.dump(model.best_estimator_, open(model_filepath, 'wb'))


def main():
    if len(sys.argv) == 3:

        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        print(model.best_params_)

        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)
  
        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')
  

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()