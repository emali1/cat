from joblib import dump, load
from sklearn.pipeline import make_pipeline
from xgboost import XGBRegressor
import numpy as np
import pandas as pd


class Predictor:

    def __init__(self,modelFile,transformerFile):
        self.model = load(modelFile)
        self.transformer = load(transformerFile)


    @staticmethod
    def __processdata(df):
        #remove  column 
        df = df.drop(['last_vet_visit'],axis=1)    
        return df



    #making prediction 
    def prediction(self,input_df):
        return self.model.predict(self.transformer.transform(self.__processdata(input_df)))

        

    

