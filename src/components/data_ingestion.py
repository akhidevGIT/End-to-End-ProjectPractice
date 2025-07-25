import os
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd

from sklearn.model_selection import train_test_split
from dataclasses import dataclass


from src.components.data_transformation import DataTransformation, DataTransformationConfig

from src.components.model_trainer import ModelTrainerConfig, ModelTrainer

@dataclass 
# @dataclass saves you from writing __init__, __repr__, etc.
# Perfect for config or model classes that just hold data.
# Improves readability, reduces boilerplate.
class DataIngestionConfig:
    '''
    This class keeps the inputs required for the data ingestion component. Like the paths for storing the data.
    '''
    train_data_path: str = os.path.join('artifacts', "train.csv")
    test_data_path: str = os.path.join('artifacts', "test.csv")
    raw_data_path: str = os.path.join('artifacts', "raw.csv")

class DataIngestion:

    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion component")
        try:
            df = pd.read_csv('notebook/data/stud.csv') #read data into dataframe
            logging.info('Read the dataset as dataframe')

            os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path), exist_ok= True) #Create artifacts directory
            
            df.to_csv(self.ingestion_config.raw_data_path, index= False, header= True) #store raw data (dataframe) into raw.csv in artifacts folder
            
            logging.info("Train test split initiated")

            train_set, test_set = train_test_split(df, test_size= 0.2, random_state= 42) #split raw data into train and test datasets 

            train_set.to_csv(self.ingestion_config.train_data_path, index = False, header = True) #store train data in train.csv
            test_set.to_csv(self.ingestion_config.test_data_path, index = False, header = True) #store test data in test.csv

            logging.info("Data Ingestion completed!!")

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
        except Exception as e:
            raise CustomException(e, sys)
        

if __name__ == "__main__":
    obj = DataIngestion()
    train_path, test_path = obj.initiate_data_ingestion()


    data_transformation = DataTransformation()

    train_arr, test_arr, _ = data_transformation.initiate_data_transformation(train_path, test_path)
    

    model_trainer = ModelTrainer()
    print(model_trainer.initiate_model_trainer(train_arr, test_arr))



    