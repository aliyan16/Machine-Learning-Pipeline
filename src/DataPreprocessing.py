import os
import logging
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
import string
import nltk
nltk.download('stopwords')
nltk.download('punkt')


#logs name directory created
logDirectory='logs'
os.makedirs(logDirectory,exist_ok=True)

#logger object initialized
logger=logging.getLogger('DataPreprocessing') #logger with name dataIngestion
logger.setLevel('DEBUG')

#hander
#type1
consoleHandler=logging.StreamHandler()
consoleHandler.setLevel('DEBUG')
#type2
logFilePath=os.path.join(logDirectory,'DataPreprocessing.log')
fileHandler=logging.FileHandler(logFilePath)
fileHandler.setLevel('DEBUG')
#formatter
formatter=logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
consoleHandler.setFormatter(formatter)
fileHandler.setFormatter(formatter)
#logger object set
logger.addHandler(consoleHandler)
logger.addHandler(fileHandler)

def transformText(text):
    '''transforms the text'''
    ps=PorterStemmer()
    text=text.lower()
    text=nltk.word_tokenize(text)
    y=[]
    for i in text:
        if i.isalnum():
            y.append(i)
    text=y[:]
    y.clear()
    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)
    text=y[:]
    y.clear()
    for i in text:
        y.append(ps.stem(i))
    return ' '.join(y)

def preprocessData(df,textColumn='text',targetColumn='target'):
    '''preprocess dataframe my encoding target column and remove duplicates plus transform text'''
    try:
        encoder=LabelEncoder()
        df['target']=encoder.fit_transform(df['target'])
        df=df.drop_duplicates(keep='first')
        df['transformedText']=df['text'].apply(transformText)
        logger.debug('Text preprocess completed')
        return df
    except Exception as e:
        logger.error('Unexpected error ',e)
        raise

def main():
    try:
        trainData=pd.read_csv('./data/raw/trainData.csv')
        testData=pd.read_csv('./data/raw/testData.csv')
        trainProcessedData=preprocessData(trainData,'text','target')
        testProcessedData=preprocessData(testData,'text','target')
        dataPath=os.path.join('./data','interim')
        os.makedirs(dataPath,exist_ok=True)
        trainProcessedData.to_csv(os.path.join(dataPath,'trainProcessed.csv'),index=False)
        testProcessedData.to_csv(os.path.join(dataPath,'testProcessed.csv'),index=False)
        logger.debug('Processed Data saved')
    except Exception as e:
        logger.error('unexpected error ',e)
        raise

if __name__=='__main__':
    main()

