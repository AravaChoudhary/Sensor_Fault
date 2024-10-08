from pymongo.mongo_client import MongoClient
import pandas as pd
import json

# url
uri = "mongodb+srv://Aranika:Nainika1234$@cluster0.d2qw0.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"

# creating a new client and connecting to server
client = MongoClient(uri)

# DataBase name and collection name
DATABASE_NAME = 'sensordb'
COLLECTION_NAME = 'waferfault'

df = pd.read_csv('/Users/aravachoudhary/Desktop/sensor/notebooks/wafer_23012020_041211.csv')
df = df.drop('Unnamed: 0',axis = 1)

json_record = list(json.loads(df.T.to_json()).values())

client[DATABASE_NAME][COLLECTION_NAME].insert_many(json_record)
