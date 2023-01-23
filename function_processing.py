import os
import csv
from  math import sqrt

#imports to work with sqlacademy
from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base
from sqlalchemy.orm import Session

from sqlalchemy import MetaData
from sqlalchemy.orm import mapper

from sqlalchemy import Column, Float, Integer,String

#import libraries to work with datasets
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt

#import libraries to draw function 
from bokeh.io import output_file
from bokeh.plotting import figure, show
from bokeh.layouts import gridplot

#prepare paths for source files
data_folder="/home/natalia/Downloads/iu_written_task/"

data_train=os.path.join(data_folder,"train.csv")
data_test=os.path.join(data_folder,"test.csv")
data_ideal=os.path.join(data_folder,"ideal.csv")

#sqlite engine creation
database_file="/home/natalia/python_tasks/written_task/database.db"
engine = create_engine(f'sqlite:///{database_file}', echo=True)

# allows us to create classes that include
# directives to describe the actual database 
# table they will be mapped to
Base = declarative_base()


class Mapped_Points(Base):
    ''' table to add Mapped points to the database
    '''
    __tablename__ = 'mapped_test_points'

    id = Column(Integer, primary_key=True)
    x_test = Column(Float)
    y_test = Column(Float)
    y_ideal = Column(String)
    test_ideal_devtn = Column(Float)
    
#create table 
Base.metadata.create_all(engine)

#additional defined exceptions
class DataFrameIsEmpty(Exception):
  ''' class to raise exception about the dataframe emptiness  
      which involved into the arithmetic operation

      Attributes
      name : the name of the dataframe
      arithm : operation 
  '''
  def __init__(self,name,arithm):
    self.name=name
    self.arithm=arithm
    self.message=f'Emptiness of {self.name}\
    in the {self.arithm} operation'
    super().__init__(self.message)

class DataFramesIncorrectDim(Exception):
  '''class to raise exception about the dataframes incorrect 
     dimensions for the operations

     Attributes
     dim1 : dimension of dataframe1
     dim2 : dimension of dataframe2
  '''
  def __init__(self,dim1,dim2,arithm):
    self.dim1=dim1
    self.dim2=dim2
    self.arithm=arithm
    self.message=f'incorrect dimensions dataframe1 {self.dim1} \
    dataframe2 {self.dim2}  in {self.arithm} operation'
    super().__init__(self.message)

class DataFrameIncorrectType(Exception):
  '''class to raise exception about dataframe column (serias)  incorrect type

     Attributes
     dfname: dataframe from which the column will be used    
  '''
  def __init__(self,dfname):
    self.dfname = dfname
    self.message=f'incorrect type of column in dataframe {self.dfname}'
    super().__init__(self.message)


class IdealF:
    '''class implement possibility to find fitted 
       ideal function for given train
       function

       Attributes
       trainColumn - train function [column from the train functions dataset] 
       idealDataFrame - dataframe with ideal functions
    '''
    def __init__(self,trainColumn,idealDataFrame):
        self.trainColumn=trainColumn
        self.idealDataFrame=idealDataFrame

   
    def least_squares(self,trainF,idealF):
        ''' function to culculate least squares
            Attributes
            trainF - column from train dataset
            idealF - column from ideal dataset
        '''
        if idealF.empty:
            raise DataFrameIsEmpty("ideal functions dataframe","least squares")
        if trainF.empty:
            raise DataFrameIsEmpty("train functions dataframe","least squares")
        if idealF.shape != trainF.shape:
            raise DataFramesIncorrectDim(idealF.shape,trainF.shape,"least squares")
        if idealF.dtype != 'float64':
            raise DataFrameIncorrectType("ideal")
        if trainF.dtype != 'float64':
            raise DataFrameIncorrectType("train")

        return (((abs(idealF)- abs(trainF))**2).sum())/trainF.shape[0]

    
    def find(self):
        '''
        function to culculate deviation between to points like a distance
        '''
        if self.trainColumn.empty:
            raise DataFrameIsEmpty("train column dataframe","find fitted function")
        if self.idealDataFrame.empty:
            raise DataFrameIsEmpty("ideal functions dataframe","find fitted function")

        if self.trainColumn.shape[0] != self.idealDataFrame.shape[0]:
            raise DataFramesIncorrectDim(self.trainColumn.shape[0],\
                self.idealDataFrame.shape[0],"find fitted function")

        indx=0
        
        for i in range(0,self.idealDataFrame.shape[1]):
            try:
                tmp=self.least_squares(self.trainColumn,self.idealDataFrame.iloc[:,i])
                if i==0:
                    minvl=tmp
            except (DataFrameIsEmpty,DataFramesIncorrectDim,DataFrameIncorrectType):
                raise
            else:
                if tmp < minvl :
                    minvl=tmp
                    indx=i        

        return indx+1

  
    def getMaxD(self,trainColumn,idealColumn):
        '''function to calculate maximum deviation
           between train and ideal function 
        '''
        if trainColumn.empty:
            raise DataFrameIsEmpty("train functions dataframe","get max deviation")
        if idealColumn.empty:
            raise DataFrameIsEmpty("ideal functions dataframe","get max deviation")
        if trainColumn.shape != idealColumn.shape:
            raise DataFramesIncorrectDim(trainColumn.shape,idealColumn.shape,"get max deviation")
        if trainColumn.dtype != 'float64':
            raise DataFrameIncorrectType("ideal")
        if idealColumn.dtype != 'float64':
            raise DataFrameIncorrectType("train")
        return (abs( abs(idealColumn)- abs(trainColumn) )).max()



def get_fit_func(train_data, ideal_data):
    ''' function to get list of fitted functions
        every item in the list has (number_of_train_fnc,number_of_ideal_fnc,max_deviation) 
    '''
    lstTrainIdealMdev=list()
    for c in range(1,train_data.shape[1]):
        try:

            f=IdealF(train_data.iloc[:,c],ideal_data.iloc[:,1:ideal_data.shape[1]])   
            rownum=f.find()
            lstTrainIdealMdev.append((c,rownum,\
                f.getMaxD(train_data.iloc[:,c],ideal_data.iloc[:,rownum])))
        except (DataFrameIsEmpty,DataFramesIncorrectDim,DataFrameIncorrectType):
            raise
    return lstTrainIdealMdev

def get_Mapped_points(merged_by_test_points,lstTrainIdealM):
    ''' function to find mapped test points to ideal functions

        Attributes
        merged_by_test_points - dataset with x-y values of ideal functions  
        which correspond by x for x-y test points values 

        lstTrainIdealM - list of (number_of_train_fnc,number_of_ideal_fnc,max_deviation) 

        Return 
        mapped_pnts - list of points to build graphs
        mapped_dict - dictionary with values to fill the database table

    '''
    mapped_pnts=list()
    mapped_dict=[]

    if merged_by_test_points.empty:
        raise DataFrameIsEmpty("merged points","get mapped")
    if not lstTrainIdealM:
        raise DataFrameIsEmpty("lstTrainIdealM", "get mapped")

    for j in range(1,merged_by_test_points.shape[0]):
        print(f'point{j } --{merged_by_test_points.iloc[j,0]},{merged_by_test_points.iloc[j,1]}')
        min=None
        indx=0
        for i,x in enumerate(lstTrainIdealM):
            dev=abs(merged_by_test_points.iloc[j,1]-merged_by_test_points.iloc[j,x[1]+1])
            if dev <= lstTrainIdealM[i][2]*sqrt(2):
                if min == None or dev<min:
                    min=dev
                    indx=i
        if min != None: 
            mapped_pnts.append((merged_by_test_points.iloc[j,0],\
                    merged_by_test_points.iloc[j,1],min,lstTrainIdealM[indx][1]))
             
            mapped_dict.append(Mapped_Points(x_test=merged_by_test_points.iloc[j,0],\
                y_test=merged_by_test_points.iloc[j,1],y_ideal=min,\
                    test_ideal_devtn=lstTrainIdealM[indx][1]))

    return mapped_pnts,mapped_dict

def main():
    #read data from the source files
    try:
        train_data=pd.read_csv(data_train)
        ideal_data=pd.read_csv(data_ideal)
        test_data=pd.read_csv(data_test)
    except FileNotFoundError:
        print(f'Error in the path to file with data')
    else:
        #write train data set to the database 
        train_data.to_sql("train",engine)

        #write ideal data set to the database
        ideal_data.to_sql("ideal",engine)

        #get list of fitted ideal functions
        lstTrainIdealM=list()
        try:
            lstTrainIdealM=get_fit_func(train_data, ideal_data)
        except (DataFrameIsEmpty,DataFramesIncorrectDim,DataFrameIncorrectType) as exc:
            print(repr(exc.message))

        else:
            # prepare merged dataframe
            merged_by_test_points=pd.merge(test_data,ideal_data, how ='inner', on =['x'])
            try:
                mapped_pntsr,mapped_dictr=get_Mapped_points(merged_by_test_points,lstTrainIdealM)
            except DataFrameIsEmpty as exc:
                print(repr(exc.message))
            else:
                map_points_dataset=pd.DataFrame(mapped_pntsr)

                #draw graphs for functions
                graphs=list()
                for x in lstTrainIdealM:

                    xx = train_data['x']
                    #get train y values
                    ytrain = train_data[f'y{x[0]}']
                    #get ideal y values
                    yideal = ideal_data[f'y{x[1]}']
                    #get map points 
                    mapped = map_points_dataset[map_points_dataset[3]==x[1]]
                    x_pnt = mapped[0]
                    y_pnt = mapped[1]
                    #get deviation for the points 
                    y_dvn = mapped[2]

                    fig = figure()
                    fig.line(xx, ytrain, line_width=2,color="yellow",legend_label="train y")
                    fig.line(xx, yideal, line_width=2,color="orange",legend_label="ideal y")
                    fig.square(x_pnt, y_pnt, fill_color="black", size=10,legend_label="test point")
                    fig.circle(x_pnt,y_dvn,color="green",size=6,legend_label="test point deviation")

                    graphs.append(fig)

                grid=gridplot(children=graphs,ncols=2)
                show(grid)

            with Session(engine) as session:
    
                session.add_all(mapped_dictr)
                session.commit()
if __name__=='__main__':
    main()