import os
import pandas as pd
import numpy as np
import pickle
import logging
from sklearn.ensemble import RandomForestClassifier
import yaml

#logs name directory created
logDirectory='logs'
os.makedirs(logDirectory,exist_ok=True)

#logger object initialized
logger=logging.getLogger('Model Building') #logger with name dataIngestion
logger.setLevel('DEBUG')

#hander
#type1
consoleHandler=logging.StreamHandler()
consoleHandler.setLevel('DEBUG')
#type2
logFilePath=os.path.join(logDirectory,'Modelbuilding.log')
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

def loadData(filePath:str)-> pd.DataFrame:
    '''loads data from csv file'''
    try:
        df=pd.read_csv(filePath)
        logger.debug('data loaded successfully')
        return df
    except Exception as e:
        logger.error(e)
        raise

def trainModel(xtrain:np.ndarray,ytrain:np.ndarray,params:dict)->RandomForestClassifier:
    '''train random forest model'''
    try:
        if xtrain.shape[0]!=ytrain.shape[0]:
            raise ValueError('number of samples in xtrain and ytrain must be same')
        logger.debug('initializing randomForest model with params ',params)
        clf=RandomForestClassifier(n_estimators=params['n_estimators'],random_state=params['random_state'])
        clf.fit(xtrain,ytrain)
        logger.debug('model training completed')
        return clf
    except Exception as e:
        logger.error(e)
        raise
def saveModel(model,filePath:str)->None:
    '''save the trained model'''
    try:
        os.makedirs(os.path.dirname(filePath),exist_ok=True)
        with open(filePath,'wb') as file:
            pickle.dump(model,file)
        logger.debug('model saved to ',filePath)
    except Exception as e:
        logger.error(e)
        raise

def main():
    try:
        params=loadParams(paramsPath='params.yaml')['modelBuilding']
        # params={'n_est':50,'randomState':42}
        trainData=loadData('./data/processed/trainTfid.csv')
        xtrain=trainData.iloc[:,:-1].values
        ytrain=trainData.iloc[:,-1].values
        clf=trainModel(xtrain=xtrain,ytrain=ytrain,params=params)
        modelPath='models/model.pkl'
        saveModel(clf,modelPath)
    except Exception as e:
        logger.error(e)
        raise

if __name__=='__main__':
    main()