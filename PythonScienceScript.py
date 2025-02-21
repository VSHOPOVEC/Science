import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
import math as mt
from scipy.special import erf
plt.rcParams['xtick.labelsize'] = 30
plt.rcParams['ytick.labelsize'] = 30
name_of_files = ["F27_data_arr.dat","F30_data_arr.dat","F32_data_arr.dat", "F35_data_arr.dat","F37_data_arr.dat","F40_data_arr.dat","F42_data_arr.dat","F45_data_arr.dat","F47_data_arr.dat","F50_data_arr.dat","F110_data_arr.dat","F120_data_arr.dat","F130_data_arr.dat","F140_data_arr.dat","F150_data_arr.dat","F160_data_arr.dat"]
AmountOfArr = 3
StaticConstNorm = 10

def LoadFileData(name_of_file):
    try:
        with open(name_of_file) as file:
           input_file = [name_of_file,[],[]]
           array_of_strings = file.readlines()
           for line in array_of_strings:
               split_line = line.split()
               if split_line[1] != "NaN" and split_line[0] != "NaN" and split_line[2] != "NaN":
                   input_file[1].append(float(split_line[0])) #time
                   input_file[2].append(float(split_line[2])) #point
        return input_file
    except ValueError:
        print("Can't open your file")

def LoadFileDataWithMap(name_of_file):
        with open(name_of_file) as file:
           mydict = {}
           array_of_strings = file.readlines()
           for line in array_of_strings:
               split_line = line.split()
               if split_line[1] != "NaN" and split_line[0] != "NaN" and split_line[2] != "NaN" and split_line[0] not in mydict:
                   results = []
                   mydict[split_line[0]] = results
                   results.append(float(split_line[2]))
               elif split_line[1] != "NaN" and split_line[0] != "NaN" and split_line[2] != "NaN":
                   mydict[split_line[0]].append(float(split_line[2]))
        time_list = list(mydict.keys())
        value_list = list(mydict.values())
        for ind in range(len(time_list)):
            time_list[ind] = float(time_list[ind])
            value_list[ind] = np.array(list(value_list[ind]))
        time_list = np.array(time_list)
        sorted_list = list(zip(time_list, value_list))
        sorted_combined = sorted(sorted_list)
        sorted_time, sorted_ch_wait = zip(*sorted_combined)   
        return np.array(sorted_time),sorted_ch_wait
    
def ToSortPoints(thiknessFromfile): #Переписанная функция
    lenTimeArr = len(thiknessFromfile) - 1 #Изменение индексации
    listThiknessResult = list(np.concatenate([thiknessFromfile[ind] for ind in range(index, AmountOfArr + index) ]).tolist() for index in range(0, lenTimeArr - AmountOfArr))
    return listThiknessResult

def ToSortTime(timeFromfile):
    lenTimeArr = len(timeFromfile) - 1
    tempTimeResult = [np.mean(timeFromfile[0 + ind: AmountOfArr + ind]) for ind in range(0,lenTimeArr - AmountOfArr)]
    timeResult = np.array([float(x) for x in tempTimeResult])
    return timeResult

def ConfidenceInterval(listThiknessResult):
    sq = np.array([ np.std(item) for item in listThiknessResult])
    num = np.array([len(item) for item in listThiknessResult])
    sq_num = np.sqrt(num)
    coeff = np.array([get_value(item) for item in num])
    temp_result = list(coeff*sq/(sq_num))
    result = np.array([float(item) for item in temp_result])
    return result

def DataNormalize(subListThikness):
    maxVal = float(np.max(subListThikness))
    minVal = float(np.min(subListThikness))
    NormailizeList = np.array([(item - minVal) / (maxVal - minVal)*StaticConstNorm for item in subListThikness])
    return NormailizeList, [minVal, maxVal]

def ErfFunc(x, sigma, nu):
    Epsilon = 0.00000000001
    return 1/2 + 1/2 * erf(np.log(x + Epsilon) - nu)/(sigma*np.sqrt(2))
#Epsilon is a constant need to us,                                                                     
#just because in another case u will get error log(0),                                                                    
#because min value in array transform into 0.

ErfFunc_vectorized = np.vectorize(ErfFunc)

def ErfFuncFit(data_x, data_y, minVal, maxVal):
    params, covariance = curve_fit(ErfFunc, data_x, data_y)
    sigmaFit, nuFit = params
    sigma, nuFit = float(sigmaFit), float(nuFit)
    checkWait =  (mt.exp(nuFit + (sigmaFit**2)/2)/StaticConstNorm)*(maxVal - minVal) + minVal
    disper = ((mt.exp(sigmaFit**2) - 1)*mt.exp(2*nuFit + sigmaFit**2))/StaticConstNorm*(maxVal - minVal) + minVal
    return checkWait, disper, sigma, nuFit

def get_value(r_input):
    data = {
        5: 2.78, 6: 2.57, 7: 2.45, 8: 2.37, 9: 2.31, 10: 2.26, 
        11: 2.23, 12: 2.20, 13: 2.18, 14: 2.16, 15: 2.15, 
        16: 2.13, 17: 2.12, 18: 2.11, 19: 2.10, 20: 2.093, 
        25: 2.064, 30: 2.045, 35: 2.032, 40: 2.023, 45: 2.016, 
        50: 2.009, 60: 2.001, 70: 1.996, 80: 1.991, 90: 1.987, 
        100: 1.984, 120: 1.980
    }
    
    if r_input <= 20:
        return data.get(r_input, 2.78)  # Вернем значение, либо 2.78, если r_input меньше 5

    # Для значений больше 20 ищем ближайшее ключевое значение
    keys = sorted(data.keys())
    closest_key = min(keys, key=lambda x: abs(x - r_input))
    return data[closest_key]

def To_get_Funk_of_probablitity(normalizePoints, val_normalizePoints):
    bins = 100
    minVal, maxVal = val_normalizePoints[0], val_normalizePoints[1]
    edges = np.linspace(0, StaticConstNorm, num = bins)
    numPointsIntoInerval = np.histogram(normalizePoints,bins = edges)[0]
    totalSumOfPoints = np.sum(numPointsIntoInerval)
    probabilityToBeInInterval = numPointsIntoInerval/totalSumOfPoints
    y_FunkOfProbability = np.cumsum(probabilityToBeInInterval)
    x_FunkOfProbability = edges[:-1]
    checkWait, disper, sigma, nu = ErfFuncFit(x_FunkOfProbability, y_FunkOfProbability, minVal, maxVal)
    return [y_FunkOfProbability, x_FunkOfProbability, checkWait, disper, sigma, nu]

def Processing_the_Result(index_of_file, limit_of_thikness, limit_of_time):#Индекс файла в массиве файловых имен

   file_index = index_of_file
   name, time_c, point_c = LoadFileData(name_of_files[file_index])
   
   time, points = LoadFileDataWithMap(name_of_files[file_index])#to loud file and to separate time and thikness
   points = ToSortPoints(points)# to combine {AmountOfArr} points arrayes  into one \\ points = thikness
   
   
   point_mask = [np.array([ limit_of_thikness > temp_item for temp_item in item]) for item in points]
   points = [np.array(arr)[mask_temp] for arr, mask_temp in zip(points,point_mask)]#Возвращает массив типа Bool для сартировки 
   times = ToSortTime(time)# to combine {AmountOfArr} time arrayes  into one 
   time_mask = np.array([ limit_of_time > time for time in times])#Возвращает массив типа Bool для сартировки массива времен
   times = times[time_mask]
   
   normalizePoints = [DataNormalize(item)[0] for item in points]#Value of points
   val_normalizePoints = [DataNormalize(item)[1] for item in points]#Value of max and min in each array
   funk_of_probablitity_res_for_ALL_arrayes = [To_get_Funk_of_probablitity(normal, val_normal) for normal, val_normal in zip(normalizePoints,val_normalizePoints)]
   
   checkWait = [item[2] for item in funk_of_probablitity_res_for_ALL_arrayes]#checkWait 
   checkWait = np.array(checkWait)[time_mask]#Привожу массив матОжиданий к одному размеру со временем
   
   return times, checkWait, time_c, point_c, name
def main():
    
   time, checkWait, time_c, point_c, name = Processing_the_Result(5,2,3) 
   
   plt.figure(figsize = (30,30))
   plt.subplot(2,1,1)
   plt.grid() 
   plt.title(str(name), fontsize = 30)
   plt.scatter(time_c, point_c)
   plt.plot(time,checkWait)
   plt.scatter(time,checkWait)
   
   plt.subplot(2,1,2)
   plt.plot(time,checkWait)
   plt.scatter(time,checkWait)
   
main()