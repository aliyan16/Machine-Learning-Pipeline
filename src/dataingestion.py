import pandas as pd
import os
from sklearn.model_selection import train_test_split
#builtin python lib logging
import logging
#importing yaml for params
import yaml

#logs name directory created
logDirectory='logs'
os.makedirs(logDirectory,exist_ok=True)

#logger object initialized
logger=logging.getLogger('DataIngestion') #logger with name dataIngestion
logger.setLevel('DEBUG')

#hander
#type1
consoleHandler=logging.StreamHandler()
consoleHandler.setLevel('DEBUG')
#type2
logFilePath=os.path.join(logDirectory,'Dataingestion.log')
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

def loadData(dataUrl:str)->pd.DataFrame:
    '''load data from csv file'''
    try:
        df=pd.read_csv(dataUrl)
        logger.debug(f'Data Loaded from {dataUrl}')
        return df
    except Exception as e:
        logger.error('Unexpected error ',e)
        raise

def preprocessData(df:pd.DataFrame)->pd.DataFrame:
    '''preprocess data'''
    try:
        df.drop(columns=['Unnamed: 2','Unnamed: 3','Unnamed: 4'],inplace=True)
        df.rename(columns={'v1':'target','v2':'text'},inplace=True)
        logger.debug('Data preprocess completed')
        return df
    except Exception as e:
        logger.error('Unexpected error ',e)
        raise

def saveData(trainData: pd.DataFrame, testData: pd.DataFrame, dataPath: str) -> None:
    '''save train and test data'''
    try:
        rawDataPath = os.path.abspath(os.path.join(dataPath,'raw'))
        os.makedirs(rawDataPath, exist_ok=True)  # Ensure directory exists
        trainData.to_csv(os.path.join(rawDataPath,'trainData.csv'), index=False)
        testData.to_csv(os.path.join(rawDataPath,'testData.csv'), index=False)
        logger.debug(f'train and test data saved to {rawDataPath}')
    except Exception as e:
        logger.error('Unexpected error ', e)  # Improved logging
        raise



def main():
    
    try:
        # test_size=0.2
        params=loadParams(paramsPath='params.yaml')
        test_size=params['dataingestion']['test_size']
        dataPath='https://raw.githubusercontent.com/aliyan16/Datasets/refs/heads/main/spam.csv'
        df=loadData(dataUrl=dataPath)
        finalDf=preprocessData(df=df)
        trainData,testData=train_test_split(finalDf,test_size=test_size,random_state=42)
        saveData(trainData=trainData,testData=testData,dataPath='./data')
        logger.debug(f'dataingestion executed')
    except Exception as e:
        logger.error('Unexpected error ',e)
        print(e)

if __name__=='__main__':
    main()