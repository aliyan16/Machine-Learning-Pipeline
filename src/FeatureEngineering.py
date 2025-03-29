import pandas as pd
import os
from sklearn.feature_extraction.text import TfidfVectorizer
import logging
import yaml


#logs name directory created
logDirectory='logs'
os.makedirs(logDirectory,exist_ok=True)

#logger object initialized
logger=logging.getLogger('Feature Engineering') #logger with name dataIngestion
logger.setLevel('DEBUG')

#hander
#type1
consoleHandler=logging.StreamHandler()
consoleHandler.setLevel('DEBUG')
#type2
logFilePath=os.path.join(logDirectory,'FeatureEngineering.log')
fileHandler=logging.FileHandler(logFilePath)
fileHandler.setLevel('DEBUG')
#formatter
formatter=logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
consoleHandler.setFormatter(formatter)
fileHandler.setFormatter(formatter)
#logger object set
logger.addHandler(consoleHandler)
logger.addHandler(fileHandler)

def loadParams(paramsPath:str)->dict:
    """load parameters from a Yaml file"""
    try:
        with open(paramsPath,'r') as file:
            params=yaml.safe_load(file)
        logger.debug('parameters retrieved from %s',paramsPath)
        return params
    except Exception as e:
        logger.error('Unexpected Error',e)
        raise

def loadData(filePath:str)->pd.DataFrame:
    '''loads data from csv file'''
    try:
        df=pd.read_csv(filePath)
        df.fillna('',inplace=True)
        logger.debug('Data loaded from interim')
        return df
    except Exception as e:
        logger.error('Unexpected error ',e)
        raise

def applyTfidf(trainData:pd.DataFrame,testData:pd.DataFrame,maxFeatures:int)->tuple:
    '''apply tfidf to data'''
    try:
        vectorizer=TfidfVectorizer(max_features=maxFeatures)
        xtrain=trainData['text'].values
        ytrain=trainData['target'].values
        xtest=testData['text'].values
        ytest=testData['target'].values
        xtrainV=vectorizer.fit_transform(xtrain)
        xtestV=vectorizer.transform(xtest)
        trainDf=pd.DataFrame(xtrainV.toarray())
        trainDf['label']=ytrain
        testDf=pd.DataFrame(xtestV.toarray())
        testDf['label']=ytest
        logger.debug('data transformed')
        return trainDf,testDf
    except Exception as e:
        logger.error('unexpected error ',e)
        raise

def saveData(df:pd.DataFrame,filePath:str)->None:
    '''save transformed data'''
    try:
        os.makedirs(os.path.dirname(filePath),exist_ok=True)
        df.to_csv(filePath,index=False)
        logger.debug('data saved')
    except Exception as e:
        logger.error('unexpected error ',e)
        raise
def main():
    try:
        params=loadParams(paramsPath='params.yaml')
        maxFeatures=params['featureEngineering']['max_features']
        # maxFeatures=500
        trainData=loadData('./data/interim/trainProcessed.csv')
        testData=loadData('./data/interim/testProcessed.csv')
        trainData,testData=applyTfidf(trainData,testData,maxFeatures)
        saveData(trainData,os.path.join('./data','processed','trainTfid.csv'))
        saveData(testData,os.path.join('./data','processed','testTfid.csv'))
    except Exception as e:
        logger.error('unexpected error',e)
        raise

if __name__=='__main__':
    main()

