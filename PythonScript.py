import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
from scipy.optimize import curve_fit
import math as mt
from scipy.special import erf
plt.rcParams['xtick.labelsize'] = 30
plt.rcParams['ytick.labelsize'] = 30
name_of_files = ["F110_data_arr.dat","F120_data_arr.dat","F130_data_arr.dat","F140_data_arr.dat","F150_data_arr.dat","F160_data_arr.dat"]
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
def Check_wait(sigma,nu, coeff):
    mat_ojid =  mt.exp(nu + (sigma**2)/2)*coeff
    disper = (mt.exp(sigma**2) - 1)*mt.exp(2*nu + sigma**2)*coeff
    return mat_ojid, disper
def log_function(x, a, b):
    return a * np.log(x) + b
def log_normal(x,sigma,nu):
    return (1 / (x * sigma * np.sqrt(2 * np.pi))) * np.exp(-((np.log(x) - nu)**2) / (2 * sigma**2))
def fit_log_normal(data_x, data_y):
    params, covariance = curve_fit(log_normal, data_x, data_y)
    sigma_fit, nu_fit = params
    return sigma_fit, nu_fit
def erf_func(x,sigma,nu):
    return 1/2 + 1/2 * erf((np.log(x) - nu)/(sigma*mt.sqrt(2)))
def erf_func_normal(data_x, data_y):
    params, covariance = curve_fit(erf_func, data_x, data_y)
    sigma_fit, nu_fit = params
    return sigma_fit, nu_fit
def LoadFileData(name_of_file):
    try:
        with open(name_of_file) as file:
           input_file = [name_of_file,[],[]]
           array_of_strings = file.readlines()
           for line in array_of_strings:
               split_line = line.split()
               if split_line[1] != "NaN" and split_line[0] != "NaN" and split_line[2] != "NaN":
                   input_file[1].append(float(split_line[0]))
                   input_file[2].append(float(split_line[2]))
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
        for ind in range(len(time_list)):
            time_list[ind] = float(time_list[ind])
        value_list = list(mydict.values())
        return time_list,value_list
def Distribution(file):
    len_of_array = len(file[0]) - 1 
    result = [ np.array(file[1][ind-1]+ file[1][ind]+file[1][ind+1]) for ind in range(1,len_of_array - 1) ]
    return result 
def Norma(data):
    for ind in range(len(data)):
        coeff = np.max(data[ind])
        k = (100/coeff)
        data[ind] = k*data[ind]
    return data, 1/k
def Distribution_without(file):
    len_of_array = len(file[0]) - 1
    result = []
    for ind in range(1, len_of_array - 1):
        current_sum = file[1][ind-1]+file[1][ind]+file[1][ind+1]
        max_val = max(current_sum)
        min_val = min(current_sum)
        current_sum.remove(max_val)
        current_sum.remove(min_val)
        result.append(current_sum)
    return result
def Time(file):
    len_of_array = len(file[0]) - 1
    result = [ (file[0][ind-1]+ file[0][ind]+file[0][ind+1])/3 for ind in range(1,len_of_array - 1) ]
    return result 
def Checkmate_waiting(data):
    Checkmate_waiting_list = []
    for unit in data:
        count = np.array(list((Counter(unit).values())))
        count_sum = np.sum(count)
        count = count/count_sum
        check_array = np.sum(count*unit)
        Checkmate_waiting_list.append(float(check_array))
    return Checkmate_waiting_list
def Mean(data):
    mean = [float(np.mean(item)) for item in data]
    return mean
def Sq_average(data):
    sq_average = np.array([np.std(item) for item in data])
    return sq_average
def Function_of_density(res_in, bins):
    for unit in res_in:
        edges = np.linspace(min(unit), max(unit), num = bins)
        freq = np.histogram(unit,bins = edges)
        freq_sum = np.sum(freq[0])
        freq_prev = (freq[0]/freq_sum)
        thikness =  edges[:-1]
        sigma_fit, nu_fit = fit_log_normal(thikness, freq_prev)
        plt.figure(figsize = (30,20))
        plt.title(str(freq_sum) , fontsize = 40)
        plt.plot(thikness,freq_prev)
        plt.plot(thikness, list(map(lambda x: log_normal(x,sigma_fit,nu_fit), thikness)))
        plt.grid()
def Function_of_dis_prob(data, ind, bins, status):
    unit = data[ind]
    edges = np.linspace(min(unit), max(unit), num = bins)
    freq = np.histogram(unit,bins = edges)
    freq_sum = np.sum(freq[0])
    freq_prev = (freq[0]/freq_sum)
    thikness =  edges[:-1]
    freq_n = np.cumsum(freq_prev)
    sigma_fit, nu_fit = erf_func_normal(thikness, freq_n)
    if(status == True):
       plt.figure(figsize = (30,20))
       plt.title(str(freq_sum) , fontsize = 40)
       plt.plot(thikness,freq_n)
       plt.plot(thikness, list(map(lambda x: erf_func(x,sigma_fit,nu_fit), thikness)))
       plt.grid()
    return sigma_fit,nu_fit
def Function_of_dis_prob1(data, ind, bins, status):
    unit = data[ind]
    edges = np.linspace(min(unit), max(unit), num = bins)
    freq = np.histogram(unit,bins = edges)
    freq_sum = np.sum(freq[0])
    freq_prev = (freq[0]/freq_sum)
    thikness =  Norma(edges[:-1])
    freq_n = np.cumsum(freq_prev)
    sigma_fit, nu_fit = erf_func_normal(thikness, freq_n)
    if(status == True):
       plt.figure(figsize = (30,20))
       plt.title(str(freq_sum) , fontsize = 40)
       plt.plot(thikness,freq_n)
       plt.plot(thikness, list(map(lambda x: erf_func(x,sigma_fit,nu_fit), thikness)))
       plt.grid()
    return sigma_fit,nu_fit
def CheckAndSqAverage(Norma, status):
    Check = []
    Disper = []
    for ind in range(len(Norma[0])):
       sigma, nu = Function_of_dis_prob(Norma[0],ind,20, status)
       mat, disper = Check_wait(sigma,nu,Norma[1])
       Disper.append(float(disper))
       Check.append(float(mat))
    return Check, Disper
def Confidence_interval(data):
    sq = np.array(Sq_average(data))
    num = np.array([len(item) for item in data])
    sq_num = np.sqrt(num)
    coeff = np.array([get_value(item) for item in num])
    return list(coeff*sq/(sq_num))
def Plot_for_ind(ind):
   file_cloud = LoadFileData(name_of_files[ind])
   file = LoadFileDataWithMap(name_of_files[ind])
   res = Distribution(file)
   res_wh = Distribution_without(file)
   time = Time(file)
   end = max(time)
   mean_wh = Mean(res_wh)
   check_wh = Checkmate_waiting(res_wh)
   mean_wh = Mean(res_wh)
   mean = Mean(res)
   check = Checkmate_waiting(res)
   mean = Mean(res)
   Conf = Confidence_interval(res)
   plt.figure( figsize = (30,20))
   plt.plot(time,check, label = "График Мат.ожидания")
   plt.errorbar(time,check, yerr= Conf, color = "red", fmt=' ',capsize= 10)
   plt.scatter(time,check_wh, marker = 'v', s = 1500, color= "orange", label = "Мат.ожидание без наиб и наим")
   plt.scatter(time,mean_wh, marker = '.', s = 1500, color= "blue", label = "среднее без наиб и наим")
   plt.scatter(time,mean, marker = 'x', s = 1500, color= "red", label = "среднее")
   plt.scatter(file_cloud[1],file_cloud[2])
   plt.xlabel("time", fontsize = 30)
   plt.ylabel("thikness",fontsize = 30)
   plt.title(str(name_of_files[ind]),fontsize = 40)
   mat, disper = CheckAndSqAverage(Norma(Distribution(file)), False)
   plt.scatter(time, mat, s = 500,label = "Мат.ожидание из функц.распр")
   arr, time_ = Array_average(mat,check, time)
   coeffs = Appr_liner(time_, arr)
   poly_func = np.poly1d(coeffs)
   arr1  = poly_func(np.arange(0,end,0.00005))
   plt.plot(np.arange(0,end,0.00005), arr1)
   plt.legend(fontsize = 30)
   plt.grid()
def Plot2():
   for ind in range(len(name_of_files)):
      file_cloud = LoadFileData(name_of_files[ind])
      file = LoadFileDataWithMap(name_of_files[ind])
      res = Distribution(file)
      res_wh = Distribution_without(file)
      time = Time(file)
      mean_wh = Mean(res_wh)
      check_wh = Checkmate_waiting(res_wh)
      mean_wh = Mean(res_wh)
      mean = Mean(res)
      check = Checkmate_waiting(res)
      mean = Mean(res)
      Conf = Confidence_interval(res)
      plt.figure( figsize = (30,20))
      plt.plot(time,check, label = "График Мат.ожидания")
      plt.errorbar(time,check, yerr= Conf, color = "red", fmt=' ',capsize= 10)
      plt.scatter(time,check_wh, marker = 'v', s = 1500, color= "orange", label = "Мат.ожидание без наиб и наим")
      plt.scatter(time,mean_wh, marker = '.', s = 1500, color= "blue", label = "среднее без наиб и наим")
      plt.scatter(time,mean, marker = 'x', s = 1500, color= "red", label = "среднее")
      plt.scatter(file_cloud[1],file_cloud[2])
      plt.xlabel("time", fontsize = 30)
      plt.ylabel("thikness",fontsize = 30)
      plt.title(str(name_of_files[ind]),fontsize = 40)
      mat, disper = CheckAndSqAverage(Norma(Distribution(file)),False)
      mat = mat[:-1]
      plt.scatter(time[:-1], mat, s = 500,label = "Мат.ожидание из функц.распр")
      plt.legend(fontsize = 30)
      plt.grid()
def Appr_liner(data_x, data_y):
    coeff = np.polyfit(data_x, data_y, deg = 1)
    return coeff
def Array_average(data_y1_coord, data_y2_coord, data_x_coord):
    if len(data_y1_coord) > len(data_y2_coord):
       k = len(data_y1_coord) -  len(data_y2_coord)
       data_y1_coord = data_y1_coord[:(-1)*k]
       result = [(data_y1_coord[ind] + data_y2_coord[ind])/2 for ind in range(len(data_y1_coord))]
       data_x_coord =  data_x_coord[:(-1)*k]
       return result,data_x_coord
    elif len(data_y2_coord) > len(data_y1_coord):
       k = len(data_y2_coord) - len(data_y1_coord)
       data_y2_coord = data_y2_coord[:(-1)*k]
       result = [(data_y1_coord[ind] + data_y2_coord[ind])/2 for ind in range(len(data_y1_coord))]
       data_x_coord = data_x_coord[:(-1)*k]
       return result,data_x_coord
    else:
       result = [(data_y1_coord[ind] + data_y2_coord[ind])/2 for ind in range(len(data_y1_coord))]
       return result,data_x_coord
file = LoadFileDataWithMap(name_of_files[0])
res = Distribution(file)
Plot2()