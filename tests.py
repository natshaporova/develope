import unittest

import numpy as np 
import pandas as pd

from function_processing import DataFrameIsEmpty
from function_processing import DataFramesIncorrectDim
from function_processing import DataFrameIncorrectType
from function_processing import IdealF
from function_processing import get_fit_func


class UnitTestForIdealR(unittest.TestCase):
    '''
    class for testing IdealF methods
    '''
    def setUp(self):
        self.df_empty=pd.DataFrame([])
        self.df_one_column=pd.DataFrame([[1],[2]])
        self.df_three_columns=pd.DataFrame(np.array([[1.0,0.2,'Nata'],\
                [3.0,4.5,'Lev'],[0.9,9.8,'Maiia']]),columns=['a','b','c'])
        self.df_y1=pd.DataFrame(np.array([[-20.0,9.765587],[-19.9,10.035174],[-19.8,9.799387]]))
        self.df_yy=pd.DataFrame(np.array([[-20.0,-0.9129453,5.408082],[-19.9,-0.8676441,5.4971857],[-19.8,-0.81367373,5.5813217]]),\
            columns=['x','y11','y22'])

    def test_1(self):
        '''
        testcase to verify raising exception from\
            least squares function in case train functions dataframe\
                is empty
        '''
        idealFclass=IdealF(self.df_empty,self.df_one_column)
        with self.assertRaises(DataFrameIsEmpty):
            idealFclass.least_squares(self.df_empty,self.df_one_column)

    def test_2(self):
        '''
        testcase to verify raising exception from\
            find (or least squares) function in case ideal functions dataframe\
                is empty            
        '''
        idealFclass=IdealF(self.df_one_column,self.df_empty)
        with self.assertRaises(DataFrameIsEmpty):
            idealFclass.find()

    def test_3(self):
        '''
        testcase to verify raising exception from\
            find function in case incorrect dimentions\
                of dataframes      
        '''
        idealFclass=IdealF(self.df_one_column,self.df_three_columns)
        with self.assertRaises(DataFramesIncorrectDim):
            idealFclass.find()
    
    def test_4(self):
        '''
        testcase to verify raising exception from\
            least squares function in case incorrect type\
                of train dataframe      
        '''
        idealFclass=IdealF(self.df_three_columns['c'],self.df_three_columns['a'])
        with self.assertRaises(DataFrameIncorrectType):
            idealFclass.least_squares(self.df_three_columns['c'],\
                self.df_three_columns['a'])
    
    def test_5(self):
        '''
        testcase to verify raising exception from\
            least squares function in case incorrect type\
                of ideal dataframe      
        '''
        idealFclass=IdealF(self.df_three_columns['a'],self.df_three_columns['c'])
        with self.assertRaises(DataFrameIncorrectType):
            idealFclass.least_squares(self.df_three_columns['a'],\
                self.df_three_columns['c'])


    def test_6(self):
        '''
        testcase to verify raising exception from\
            get Max deviation function in case train functions dataframe\
                is empty
        getMaxD(self,trainColumn,idealColumn)
        '''
        idealFclass=IdealF(self.df_empty,self.df_one_column)
        with self.assertRaises(DataFrameIsEmpty):
            idealFclass.getMaxD(self.df_empty,self.df_one_column)

    def test_7(self):
        '''
        testcase to verify raising exception from\
            get Max deviation function in case ideal functions dataframe\
                is empty            
        '''
        idealFclass=IdealF(self.df_one_column,self.df_empty)
        with self.assertRaises(DataFrameIsEmpty):
            idealFclass.getMaxD(self.df_one_column,self.df_empty)

    def test_8(self):
        '''
        testcase to verify raising exception from\
            get Max Deviation function in case incorrect dimentions\
                of dataframes      
        '''
        idealFclass=IdealF(self.df_one_column,self.df_three_columns)
        with self.assertRaises(DataFramesIncorrectDim):
            idealFclass.getMaxD(self.df_one_column,self.df_three_columns)
    
    def test_9(self):
        '''
        testcase to verify raising exception from\
            get Max Deviation function in case incorrect type\
                of train dataframe      
        '''
        idealFclass=IdealF(self.df_three_columns['c'],self.df_three_columns['a'])
        with self.assertRaises(DataFrameIncorrectType):
            idealFclass.getMaxD(self.df_three_columns['c'],\
                self.df_three_columns['a'])
    
    def test_10(self):
        '''
        testcase to verify raising exception from\
            get Max Deviation function in case incorrect type\
                of ideal dataframe      
        '''
        idealFclass=IdealF(self.df_three_columns['a'],self.df_three_columns['c'])
        with self.assertRaises(DataFrameIncorrectType):
            idealFclass.getMaxD(self.df_three_columns['a'],\
                self.df_three_columns['c'])
    
    def test_11(self):
        '''
        testcase to verfy that least squares functions \
            calculation works correct
        '''
        testResult=get_fit_func(self.df_y1,self.df_yy)
        self.assertEqual(testResult,[(1, 2, 4.537988299999999)])


if __name__ == '__main__':
    # begin the unittest.main()
    unittest.main()