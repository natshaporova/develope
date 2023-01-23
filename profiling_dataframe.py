import os
import csv
import profile
from  math import sqrt
import numpy as np 
import pandas as pd


#read data from the source files 
data_folder="/home/natalia/Downloads/iu_written_task/"

data_train=os.path.join(data_folder,"train.csv")
data_test=os.path.join(data_folder,"test.csv")
data_ideal=os.path.join(data_folder,"ideal.csv")

train_data=pd.read_csv(data_train)
ideal_data=pd.read_csv(data_ideal)
test_data=pd.read_csv(data_test)


class IdealF:
    '''
    class to find ideal functions for train functions

    Attributes
    trainColumn - one train function
    idealDataFrame - dataframe of ideal functions
    '''
    def __init__(self,trainColumn,idealDataFrame):
        self.trainColumn=trainColumn
        self.idealDataFrame=idealDataFrame
        
    def least_squares(self,trainF,idealF):
        '''
        function to get least squares of one train function column
        and one ideal function column
        '''
        return (((abs(idealF)- abs(trainF))**2).sum())/400   

    def find(self):
        '''
        function to find fitted ideal function for one train function
        '''
        indx=0
        minvl=self.least_squares(self.trainColumn,self.idealDataFrame.iloc[:,0])
        for i in range(0,self.idealDataFrame.shape[1]):
            tmp=self.least_squares(self.trainColumn,self.idealDataFrame.iloc[:,i])
        
            if tmp < minvl :
                minvl=tmp
                indx=i
                   
        return indx+1

    def getMaxD(self,trainColumn,idealColumn):
        '''
        function to get max deviation between train function and 
        ideal one
        '''
        return (abs( abs(idealColumn)- abs(trainColumn) )).max()

class MapPoint:
    '''
    class to find mapped points for fitted ideal functions
    '''
    def __init__(self,point_XY,ideal_Y,max_deviation,number_of_idealF,koef):
        self.point_XY=point_XY
        self.ideal_Y=ideal_Y
        self.max_deviation=max_deviation
        self.nIF=number_of_idealF
        self.koef=koef


    def get_mapped_list(self):
        result_points=list()
        deviation_ls= abs(self.point_XY.iloc[:,1] - self.ideal_Y)
        for i in range(0,deviation_ls.shape[0]):
            if deviation_ls.iloc[i] <= self.max_deviation*sqrt(2)*self.koef:
                result_points.append((i,self.point_XY.iloc[i,0],self.point_XY.iloc[i,1],deviation_ls[i],self.nIF))
        return result_points


def get_fit_func(train_data, ideal_data):
    '''
    function get all train data and ideal data and
    return list with next info
    (train function, ideal function, max deviation)
    '''
    lstTrainIdealMdev=list()
    for c in range(1,5):
        f=IdealF(train_data.iloc[:,c],ideal_data.iloc[:,1:51])
        rownum=f.find()
        
        lstTrainIdealMdev.append((c,rownum,\
            f.getMaxD(train_data.iloc[:,c],ideal_data.iloc[:,rownum])))
    return lstTrainIdealMdev

lstTrainIdealM=get_fit_func(train_data, ideal_data)

#merge dataframes to get more suitable dataframe to next step
merged_by_test_points=pd.merge(test_data,ideal_data, how ='inner', on =['x'])


#find mapped points list 
for i,x in enumerate(lstTrainIdealM):
    mapping=MapPoint(merged_by_test_points.iloc[:,0:2],\
       merged_by_test_points.iloc[:,x[1]+1],x[2],x[1]+1,0.8)
    result_points=mapping.get_mapped_list()
