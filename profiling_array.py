import array as arr
import os
import csv

from  math import sqrt

data_folder="/home/natalia/Downloads/iu_written_task/"

data_train=os.path.join(data_folder,"train.csv")
data_test=os.path.join(data_folder,"test.csv")
data_ideal=os.path.join(data_folder,"ideal.csv")

def from_csv_to_dictionary(csv_file,array_length):
    '''
    function for read data from source files

    Attributes
    csv_file - soursce file 
    array_length - array length to avoid resource consuming for appending

    return dictionary  

    '''
    
    dict_from_file=dict()
    number_of_keys=0

    with open(csv_file) as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',')
        for i,row in enumerate(spamreader):
            if i==0:
                number_of_keys=len(row)
                for k in range(number_of_keys):
                    dict_from_file[row[k]]=arr.array('d',[0.0]*array_length) 
            else:
                for j in range(number_of_keys):
                    dict_from_file[f'{list(dict_from_file)[j]}'][i-1]=float(row[j])

    return  dict_from_file


trainfnt=from_csv_to_dictionary(data_train,400)
testfnt=from_csv_to_dictionary(data_test,100)
idealfnt=from_csv_to_dictionary(data_ideal,400)

class UtilsArray:
    '''
    class to implement useful functions to work with arrays
    '''

    def __init__(self):
        pass

    def sum_squares_residuals(self,array1,array2):
        '''
        function to find least squares of two arrays
        '''
        nm_elements=len(array1)
        if nm_elements!=len(array2):
            return None
        else:
            sum=0
            for j in range(nm_elements):
                sum+=(array1[j]-array2[j])**2
            return sum/nm_elements

class IdealF(UtilsArray):
    '''
    class to find fitted ideal function
    '''
    def __init__(self,trainColumn,idealDataFrame):
        self.trainColumn=trainColumn
        self.idealDataFrame=idealDataFrame
        
   
    def find(self):
        '''
        function to find fitted ideal function
        it uses class attributes
        trainColumn : array of one train function (y values)
        idealDataFrame
        '''
        indx=0
        minvl=super().sum_squares_residuals(self.trainColumn,self.idealDataFrame[list(self.idealDataFrame)[0]])
        for i in range(1,len(self.idealDataFrame)):
            tmp=super().sum_squares_residuals(self.trainColumn,self.idealDataFrame[list(self.idealDataFrame)[i]])
            if tmp < minvl :
                minvl=tmp
                indx=i        
  
        return indx+1

   
    def getMaxD(self,trainColumn,idealColumn):
        '''
        function to calculate Max deviation between one train
        function and one ideal function 
        '''
        if len(trainColumn)!=len(idealColumn):
            return None
        else:
            maxval=0
            for j in range(len(trainColumn)):
                result = abs( abs(idealColumn[j]) - abs(trainColumn[j]) )
                if result >maxval:
                    maxval= result   
        return maxval

def to_find_ideal(trn_exptX,idl_exptX):
    '''
    function to find for each train function its ideal function
    '''
    resultarr=list() 
    for i in range(len(trn_exptX)):
        findpnt=IdealF(trn_exptX[list(trn_exptX)[i]],idl_exptX)
        numberIdeal=findpnt.find()
        resultarr.append((i,numberIdeal,findpnt.getMaxD(trn_exptX[list(trn_exptX)[i]],idl_exptX[list(idl_exptX)[numberIdeal-1]])))
    return resultarr


#except x - values in the arrays
trn_exptX={k:v for k, v in trainfnt.items() if k != 'x'}
idl_exptX={k:v for k, v in idealfnt.items() if k != 'x'}

#get train - ideal functions array with max deviation for it
resultarr1=to_find_ideal(trn_exptX,idl_exptX)

#actions to get more suiteable array for the next steps
numbers_of_ideal_funct=arr.array('i')
for x in resultarr1:
    numbers_of_ideal_funct.append(x[1])

all_needed_x_values={}
for i in range(len(numbers_of_ideal_funct)):
    all_needed_x_values[f'y{numbers_of_ideal_funct[i]}']=arr.array('d')

usefull_ideal_func={k:v for k, v in idealfnt.items()\
     if k == 'x' or (int(k[1:]) in numbers_of_ideal_funct)}

for z in range(len(testfnt['x'])):
  indx=usefull_ideal_func['x'].index(testfnt['x'][z])
  for g in range(len(numbers_of_ideal_funct)):
        idealfnc=numbers_of_ideal_funct[g]
        all_needed_x_values[f'y{idealfnc}'].append(usefull_ideal_func[f'y{idealfnc}'][indx])


def map_point(test_point_dict,fitted_funct,deviation):
    '''
    function to map test points for  one ideal function
    '''
    result_points=list()
    for i in range(len(test_point_dict['x'])):
        dev_res=abs(test_point_dict['y'][i]-fitted_funct[i])
        if dev_res<=deviation*sqrt(2):
            result_points.append((i,test_point_dict['x'],test_point_dict['y'],dev_res))
    return result_points

def main_map_point(test_point_dict,fitted_funct_dict,fitted_funct_dev):
    '''
    function to get mapped test points for each ideal function
    '''

    for c in range(len(fitted_funct_dict)):
    
        result_points=list()
        result_points=map_point(test_point_dict,fitted_funct_dict[f'y{fitted_funct_dev[c][1]}'],fitted_funct_dev[c][2])
        return result_points



result_p= main_map_point(testfnt,all_needed_x_values,resultarr1)


