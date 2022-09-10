import sys
import pandas as pd
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    '''
    Load data
    Load data from csv files and merge to single pandas dataframe

    Input:
    messages_filepath  : filepath to message csv file
    categories_filepath : filepath to categories csv file

    Output :
    df : dataframe , merged dataframe 

    '''

    messages = pd.read_csv(messages_filepath,dtype=str)
    categories = pd.read_csv(categories_filepath,dtype=str)

    return pd.concat([messages,categories],axis=1)


def clean_data(df):
    '''
    
    Function to clean the data
    Input:

    df : dataframe , data to be cleaned
    Output :

    df : dataframe , cleaned data

    '''
    # split the categories into dataframe
    # expand = True meaning it will divide the data
    # to 36 column
    categories = df.categories.str.split(pat=";", expand=True)
    # split the binary values .
    n = list(categories.iloc[0,:].str.split(pat="-",expand=True)[0])
    categories.columns = n

    # loop to assign every value to its correspanding category
    for column in categories:
        categories[column] = categories[column].str[-1]
        categories[column] = categories[column].astype(int)

    # delete the uncleaned category to assign new columns
    df = df.drop(["categories"],axis=1)
    df = pd.concat([df,categories],axis=1)
   
    # delete duplicated columnsand rows
    df=df.iloc[:,~df.columns.duplicated()]
    df=df.iloc[list(~df.duplicated()),:]
    # convert the value 2 to 1
    df.loc[df['related']>1,'related'] = 1
    df = df.iloc[list(~df.duplicated()),:]
    # drom the rows that has all values 1 or 0
    df.drop(df[df.iloc[:,4:].mean(axis=1)==0].index,axis=0,inplace=True)  
    df.drop(df[df.iloc[:,4:].mean(axis=1)==1].index,axis=0,inplace=True)

    return df

def save_data(df, database_filename):
    '''
    Save data
    Save dataframe to a sql database

    Input:
    dd  : dataframe to be saved
    database_filename : string name of the new database

    Output :
    None

    '''
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
