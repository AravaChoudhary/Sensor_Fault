import sys 
import os
import numpy as np
import pandas as pd
from pymongo import MongoClient
from zipfile import Path
from dataclasses import dataclass

from src.constant import *
from src.exception import CustomException
from src.logger import logging
from src.utils.main_utils import MainUtils


@dataclass
class DataIngestionConfig:
    artifact_folder: str = os.path.join(artifact_folder)


class DataIngestion:
    def __init__(self):
        self.data_ingestion_config = DataIngestionConfig()
        self.utils = MainUtils()

    def export_collection_as_dataframe(self,collection_name,db_name):
        
        try:
            mongo_client = MongoClient(MONGO_DB_URL)
            collection = mongo_client[db_name][collection_name]

            df = pd.DataFrame(list(collection.find()))
            if "_id" in df.columns.to_list():
                df = df.drop(columns = ['_id'] , axis = 1)
            
            df.replace({'na' : np.nan} , inplace = True)

            return df
        
        except Exception as e:
            raise CustomException(e,sys)
        

    
    def export_data_into_feature_store_file_path(self) -> pd.DataFrame:

        try:
            logging.info(f"Exporting Data from MongoDB")
            raw_file_path = self.data_ingestion_config.artifact_folder

            os.makedirs(raw_file_path,exist_ok = True)

            sensor_data = self.export_collection_as_dataframe(
                collection_name = "waferfault",
                db_name = "sensordb"
            )

            logging.info(f"Saving the exported Data into feature store file path : {raw_file_path}")

            feature_store_file_path = os.path.join(raw_file_path,'wafer_fault.csv')

            sensor_data.to_csv(feature_store_file_path , index = False)

            return feature_store_file_path
        
        except Exception as e:
            raise CustomException(e,sys)
        


    def intitate_data_ingestion(self) -> Path:
        
        logging.info(f"Entered intitate_data_ingestion() method of DataIntegration class")
        
        try:
            feature_store_file_path = self.export_data_into_feature_store_file_path()

            logging.info(f"Got the Data from MongoDB")
            logging.info(f"Exited intitate_data_ingestion() method of DataIngestion class")

            return feature_store_file_path

        except Exception as e:
            raise CustomException(e,sys)