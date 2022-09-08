import sys
import pandas as pd
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    messages = pd.read_csv(messages_filepath,dtype=str)
    categories = pd.read_csv(categories_filepath,dtype=str)
    return pd.concat([messages,categories],axis=1)


def clean_data(df):
    categories = df.categories.str.split(pat=";", expand=True)
    n = list(categories.iloc[0,:].str.split(pat="-",expand=True)[0])
    categories.columns = n
    for column in categories:
        categories[column] = categories[column].str[-1]
        categories[column] = categories[column].astype(int)
    df = df.drop(["categories"],axis=1)
    df = pd.concat([df,categories],axis=1)
    df=df.iloc[:,~df.columns.duplicated()]
    df=df.iloc[list(~df.duplicated()),:]
    df.loc[df['related']>1,'related'] = 1
    df = df.iloc[list(~df.duplicated()),:]
    df.drop(df[df.iloc[:,4:].mean(axis=1)==0].index,axis=0,inplace=True)  
    df.drop(df[df.iloc[:,4:].mean(axis=1)==1].index,axis=0,inplace=True)

    return df

def save_data(df, database_filename):
    engine = create_engine("sqlite:///"+database_filename)
    df.to_sql('message', engine, index=False, if_exists='replace')


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()
