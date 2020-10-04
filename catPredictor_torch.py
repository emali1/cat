from joblib import dump, load
from sklearn.pipeline import make_pipeline
import torch
from torchModel import FeedForward
import numpy as np
import pandas as pd

''' A class to perform prediction using pytorch trained model 
    This depends on 
    1. torchmodel file, saved trained pytorch model 
    2. torchmodel.py file, a Pytorch Model class 
    3. transformer_torch.joblib, a saved column transformer object as part of data process

'''
class PredictorTorch:

    def __init__(self,modelFile,transformerFile):
        self.model = torch.load(modelFile)
        self.transformer = load(transformerFile)


    @staticmethod
    def __processdata(df):
        #remove  column 
        df = df.drop(['last_vet_visit'],axis=1)    
        return df



    #making prediction 
    def prediction(self,input_df):
        return self.model.predict(torch.tensor(self.transformer.transform(input_df.drop(['last_vet_visit'],axis=1)),dtype=torch.float32))

        

    

