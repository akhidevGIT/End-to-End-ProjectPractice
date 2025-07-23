import sys
from src.logger import logging
from src.exception import CustomException
from src.utils import save_object

from dataclasses import dataclass
import os

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder

@dataclass
class DataTransformationConfig:
    preprocessor_obj_path = os.path.join('artifacts', 'preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):
        '''
        This function is responsible for data transformation
        '''
        try:
            numeric_columns = ['reading_score', 'writing_score']
            categorical_columns = ['gender', 'race_ethnicity', 'parental_level_of_education', 'lunch', 'test_preparation_course']

            num_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler())
                ]
            )

            cat_pipeline = Pipeline(
                steps= [
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("one_hot_encoder", OneHotEncoder())
                    #("scaler", StandardScaler())
                ]
            )

            preprocessor = ColumnTransformer(
                [
                    ("num_pipeline", num_pipeline, numeric_columns),
                    ("cat_pipeline", cat_pipeline, categorical_columns)
                ]
            )

            return preprocessor

        except Exception as e:
            raise CustomException(e, sys)
        
    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Read train and test data completed")

            logging.info("Obtaining preprocessing object")

            preprocessor_obj = self.get_data_transformer_object()
            
            target_column_name = ['math_score']
            train_df_X = train_df.drop(columns=target_column_name, axis= 1)
            test_df_X = test_df.drop(columns=target_column_name, axis=1)

            train_df_y = train_df[target_column_name]
            test_df_y = test_df[target_column_name]

            logging.info("Applying preprocessing on train and test input features")

            train_df_X_arr = preprocessor_obj.fit_transform(train_df_X)
            test_df_X_arr = preprocessor_obj.transform(test_df_X)

            train_arr = np.c_[train_df_X_arr, np.array(train_df_y)]
            test_arr = np.c_[test_df_X_arr, np.array(test_df_y)]
            

            logging.info("train preprocessing and test preprocessing done")

            save_object(
                file_path= self.data_transformation_config.preprocessor_obj_path,
                obj= preprocessor_obj)
            
            logging.info("Saving preprocessor object completed!")

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_path
            )



        except Exception as e:
            raise CustomException(e, sys)






