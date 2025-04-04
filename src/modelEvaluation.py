import os
import numpy as np
import pandas as pd
import pickle
import json
from sklearn.metrics import accuracy_score,precision_score,recall_score,roc_auc_score
import logging
import dvclive
import yaml
from dvclive import Live


#logs name directory created
logDirectory='logs'
os.makedirs(logDirectory,exist_ok=True)

#logger object initialized
logger=logging.getLogger('Model Evaluation') #logger with name dataIngestion
logger.setLevel('DEBUG')

#hander
#type1
consoleHandler=logging.StreamHandler()
consoleHandler.setLevel('DEBUG')
#type2
logFilePath=os.path.join(logDirectory,'ModelEvaluation.log')
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
    '''load the parameters from Yaml file'''
    try:
        with open(paramsPath,'r') as file:
            params=yaml.safe_load(file)
            logger.debug('Parameters loaded for DVClive')
            return params
    except Exception as e:
        logger.error('Unexpected error',e)
        raise


def loadModel(filePath:str):
    '''loads the model'''
    try:
        with open(filePath,'rb') as file:
            model=pickle.load(file)
        logger.debug('Model loaded successfully')
        return model
    except Exception as e:
        logger.error(e)
        raise

def loadData(filePath:str)->pd.DataFrame:
    '''loads the data from csv file'''
    try:
        df=pd.read_csv(filePath)
        logger.debug('Data loaded successfully')
        return df
    except Exception as e:
        logger.error(e)
        raise

def EvaluateModel(clf,xtest:np.ndarray,ytest:np.ndarray)->dict:
    '''Evaluate model and return evaluation metrics'''
    try:
        ypred=clf.predict(xtest)
        ypredProb=clf.predict_proba(xtest)[:,1]
        accuracy=accuracy_score(ypred,ytest)
        precision=precision_score(ypred,ytest)
        recall=recall_score(ypred,ytest)
        auc=roc_auc_score(ytest,ypredProb)
        metricsDict={
            'accuracy':accuracy,
            'precision':precision,
            'recall':recall,
            'auc':auc
        }
        logger.debug('Sucessfully computed metrices')
        return metricsDict
    except Exception as e:
        logger.error(e)
        raise

def saveMetrices(metrices:dict,filePath:str)->None:
    '''saves the metrices data'''
    try:
        os.makedirs(os.path.dirname(filePath),exist_ok=True)
        with open(filePath,'w') as file:
            json.dump(metrices,file,indent=4)
        logger.debug('Data saved to %s',filePath)
    except Exception as e:
        logger.error(e)
        raise

def main():
    try:
        params=loadParams(paramsPath='params.yaml')
        clf=loadModel('./models/model.pkl')
        testData=loadData('./data/processed/testTfid.csv')
        xtest=testData.iloc[:,:-1].values
        ytest=testData.iloc[:,-1].values
        metrices=EvaluateModel(clf,xtest,ytest)
        #initiating dvclive code
        with Live(save_dvc_exp=True) as live:
            live.log_metric('accuracy',metrices['accuracy'])
            live.log_metric('precision',metrices['precision'])
            live.log_metric('recall',metrices['recall'])
            live.log_metric('auc',metrices['auc'])
            live.log_params(params)
        saveMetrices(metrices,'reports/metrics.json')
    except Exception as e:
        logger.error(e)
        raise


if __name__=='__main__':
    main()