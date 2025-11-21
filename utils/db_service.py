from dotenv import load_dotenv
import os
from pymongo import MongoClient

class MongoDBManager:
    def __init__(self):
        load_dotenv()
        uri = os.getenv('MONGO_URI')
        db_name = os.getenv('MONGO_DB_NAME')
        self.client = MongoClient(uri)
        self.db = self.client[db_name]
 
    def get_collection(self, collection_name):
        return self.db[collection_name]
    
