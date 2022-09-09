# System for Disaster Responses

## 1. Installation

it works on python3
, and needs this libraries
- numpy
- pandas
- plotly
- flask
- nltk
- sklearn (=> 1.1.2)
  - it has some new features like ColumnTransformer
- sqlalchemy
- re
- pickle
--------------
## 2. Project Motivation

this is a project for a udacity course it is to make an app on web that can predict a message category regarding the disaster field
and shows a certain information about the dataset provided

*The interface of the web app*

![Screenshot Web1 ](https://user-images.githubusercontent.com/110268610/189388738-30018296-a0ff-44e7-befb-912bbb46c43d.png)

*The interface for Classifying a message*

![Screenshot Web2  ](https://user-images.githubusercontent.com/110268610/189388509-bfd1883a-f789-4a99-a59b-8086acf6ed9d.png)

---------

## 3. How to interact with the project

You need to install all the libraries and run the files
In the terminal :-

- process_data.py
- train_classifier.py
- run.py

Respectively

And click on the url that will be given to you in the terminal.

---------
## 4.File Descriptions

App

- templets
  - go.html : html code.
  - master.html : html code.
- run.py : code to run the app.

Data

- ETL Pipeline prepration.ipynb : notebook that led me to write the proccess data file.
- disaster_messages.csv : raw data contains messages.
- disaster_categories.csv : raw data contains categories.
- process_data.py : file to process the uncleaned data.
- DisasterResponse.db : the processed data.

Models

- train_classifier.py : file to save the classifier model.
- ML Pipeline prepration : guide to write train_classifier.py file.

Screenshot Web1 .png : a screenshot of the web

Screenshot Web2 .png : a screenshot of the web

-----

## 5. Licensing, Authors, Acknowledgements

thanks to udacity for providing the complementry codes .
