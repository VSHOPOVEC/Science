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
def Check_wait(sigma,nu, ind, coeff):
    mat_ojid =  mt.exp(nu + (sigma**2)/2)*coeff[ind]
    disper = (mt.exp(sigma**2) - 1)*mt.exp(2*nu + sigma**2)*coeff[ind]
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
    result = [ np.array(file[1][ind-1]+ file[1][ind]+file[1][ind+1] + file[1][ind+1]) for ind in range(1,len_of_array - 2) ]
    return result 
def Norma(data):
    koeff = []
    for ind in range(len(data)):
        coeff = np.max(data[ind])
        k = (100/coeff)
        data[ind] = k*data[ind]
        koeff.append(1/k)
    return data, np.array(koeff)
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
def CheckAndSqAverage(Norma, status):
    Check = []
    Disper = []
    for ind in range(len(Norma[0])):
       sigma, nu = Function_of_dis_prob(Norma[0],ind,100, status)
       mat, disper = Check_wait(sigma,nu,ind, Norma[1])
       Disper.append(float(disper))
       Check.append(float(mat))
    return Check, Disper
def Confidence_interval(data):
    sq = np.array(Sq_average(data))
    num = np.array([len(item) for item in data])
    sq_num = np.sqrt(num)
    coeff = np.array([get_value(item) for item in num])
    return list(coeff*sq/(sq_num))
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
def Selection(data_points, data_time, thikness_limit, min_amount_points_in_arr):
    general_res = []
    general_time  = []
    for item, time in zip(data_points, data_time):
        if len(item) >= min_amount_points_in_arr:
           res = []
           for point in item:
               if point <= thikness_limit:
                   res.append(float(point))
           general_res.append(np.array(res))
           general_time.append(time)
    return general_res, general_time

# ind = 5
# plt.figure(figsize = (30,20))
# file, file_cloud = LoadFileDataWithMap(name_of_files[ind]),LoadFileData(name_of_files[ind])
# res, time = Distribution(file),Time(file)
# new_point, new_time = Selection(res, time, 0.8e-5, 30)
# new_point = Norma(new_point)
# point  = CheckAndSqAverage(new_point, False)[0]
# a, b = Appr_liner(new_time[:-1], point[:-1])
# time_for_liner =  np.arange(0,max(new_time),0.000001)
# plt.plot(time_for_liner, list(map(lambda x: a*x + b ,time_for_liner)))
# plt.scatter(new_time, point,s = 1500, color= "blue", label = "мат.ожид из функц.распред" )
# plt.scatter(file_cloud[1], file_cloud[2])
# mean = Mean(res)
# check = Checkmate_waiting(res)
# mean = Mean(res)
# Conf = Confidence_interval(res)
# # plt.plot(time,check, label = "График Мат.ожидания")
# # plt.errorbar(time,check, yerr= Conf, color = "red", fmt=' ',capsize= 10)
# plt.scatter(time,mean, marker = 'x', s = 1500, color= "red", label = "среднее")
# plt.scatter(file_cloud[1],file_cloud[2])
# plt.xlabel("time", fontsize = 30)
# plt.ylabel("thikness",fontsize = 30)
# plt.title(str(name_of_files[ind]),fontsize = 40)
# plt.legend(fontsize = 30)
# plt.grid()
# print(b)

plt.figure(figsize = (20,10))
plt.plot([110,120,130,140,150,160],[7.615e-6, 4.31923e-6,1.697099e-6,1.10e-6,1.2657e-6,1.2e-6])
plt.title("thikness(power)", fontsize = 30)
plt.grid()


# def Distribution_without(file):
#     len_of_array = len(file[0]) - 1
#     result = []
#     for ind in range(1, len_of_array - 1):
#         current_sum = file[1][ind-1]+file[1][ind]+file[1][ind+1]
#         max_val = max(current_sum)
#         min_val = min(current_sum)
#         current_sum.remove(max_val)
#         current_sum.remove(min_val)
#         result.append(current_sum)
#     return result