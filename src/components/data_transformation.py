import sys 
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer

from sklearn.impute import SimpleImputer

from sklearn.pipeline import Pipeline

from sklearn.preprocessing import OneHotEncoder, StandardScaler


from src.exception import CustomException
from src.logger import logging


class DataTransformationConfig:
    preprocessor_obj_file = os.path.join("artificats", "preprocessor.pkl")

class DataTransformationn:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

        # this function responsible for data transformation

    def get_data_tranformer_obj(self):
        # all pickle files responsovble for categorical to numerical 

        try:
            numerical_columns = ["writing_score", "reading_score"]
            categorical_columns = [
                "gender",
                "race_ethinicity",
                "parental_level_of_education",
                "lunch",
                "test_preparation_course"
            ]

            numerical_pipeline = Pipeline(
                steps=[
                    # Haninding mising values
                    ("imputer",SimpleImputer(strategy="median"))
                    # Scaling
                    ("Scaler", StandardScaler())
                ]
            )

            categorical_pipeline = Pipeline(
                steps =[
                    # ("Imputer", SimpleImputer(strategy="most_frequent"))
                    ("Imputer", SimpleImputer(strategy="most_frequent"))
                    ("one_hot_enocoer", OneHotEncoder())
                    ("Scaler", StandardScaler())



                ]
            )
            logging.info("Numerical coulms standard scalling completeed")

            logging.info("cATEGORICAL ENOCIDING COMPLETED")  

            preprocessor = ColumnTransformer(
                [
                    ("numerical_piple", numerical_pipeline, numerical_columns)
                    ("Catagorical Pipleline", categorical_pipeline, categorical_columns)
                ]
            )


            return preprocessor

        except Exception as e:
            raise CustomException(e,sys)

