import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from collections import Counter
from scipy.optimize import curve_fit
import copy
from scipy.integrate import quad

plt.rcParams['xtick.labelsize'] = 30
plt.rcParams['ytick.labelsize'] = 30

name_of_files = ["F110_data_arr.dat","F120_data_arr.dat","F130_data_arr.dat","F140_data_arr.dat","F150_data_arr.dat","F160_data_arr.dat"]

def log_function(x, a, b):
    return a * np.log(x) + b

def log_normal(x,sigma,nu):
    return (1/(x*sigma*(2*np.pi)**1/2)) * np.exp((-1)*((np.log(x) - nu)**2)/(2*sigma**2))

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

def LoadFolderData(address_of_folder):
    LoadedFoldersData = []
    for file in Path(address_of_folder).rglob("*"):
        result = LoadFileData(str(file))
        LoadedFoldersData.append(result)
    return LoadedFoldersData 

def LoadFolderDataWithMap(address_of_folder):
    LoadedFoldersData = []
    for file in Path(address_of_folder).rglob("*"):
        result = LoadFileDataWithMap(str(file))
        LoadedFoldersData.append(result)
    return LoadedFoldersData 

def Distribution_of_the_condition_with_window(file, numbers, divider):
    time_max, time_min = max(file[0]), min(file[0])
    step = (time_max - time_min)/numbers
    frames = time_min/divider
    result = []
    for ind in range(numbers):
       thikness_list = [[],[time_min - frames + ind*step ,time_min + ind*step + frames]]
       for index, moment in enumerate(file[0]):
          if time_min - frames + ind*step <= moment and moment <= time_min + ind*step + frames:
              for unit in file[1][index]:  
                  thikness_list[0].append(unit)
       result.append(thikness_list)
    return result

def Checkmate_waiting(result_in):
    result = copy.deepcopy(result_in)
    Time = []
    Checkmate_waiting_list = []
    for unit in result:
        unit[1] = float(np.sum(unit[1])/2)
        count = np.array(list((Counter(unit[0]).values())))
        count_sum = np.sum(count)
        count = count/count_sum
        check_array = count*unit[0]
        Time.append(unit[1])
        Checkmate_waiting_list.append(np.sum(check_array))
    return Time, Checkmate_waiting_list

def Sq_average(result_in):
    result = copy.deepcopy(result_in)
    Time = []
    Sq_average = []
    for unit in result:
        unit[0] = float(np.std(np.array(unit[0])))
        unit[1] = float(np.sum(unit[1])/2)
        Time.append(unit[1])
        Sq_average.append(unit[0])
    return Time, Sq_average

def Function_of_distribution_of_prob(res_in, bins):
    for unit in res_in:
        edges = np.linspace(min(unit[0]), max(unit[0]), num = bins)
        freq = np.histogram(unit[0],bins = edges)
        freq_sum = np.sum(freq[0])
        freq_n = freq[0]/freq_sum
        plt.figure(figsize = (30,20))
        prob = np.cumsum(freq_n)
        params, _ = curve_fit(log_function, edges[:-1], prob, maxfev=10000)
        a, b = params
        print(prob)
        plt.title(str((-min(unit[0]) + max(unit[0]))/bins) + " " +str(freq_sum) , fontsize = 40)
        plt.plot(edges[:-1], prob)
        plt.plot(edges[:-1], list(map(lambda x: log_function(x,a,b), edges[:-1])))
        plt.grid()

def Function_of_distribution_of_def_prob(res_in, bins):
    for unit in res_in:
        edges = np.linspace(min(unit[0]), max(unit[0]), num = bins)
        freq = np.histogram(unit[0],bins = edges)
        freq_sum = np.sum(freq[0])
        freq_n = freq[0]/freq_sum
        params, _ = curve_fit(log_normal, edges[:-1], freq_n, maxfev=10000)
        sigma,nu = params
        plt.figure(figsize = (30,20))
        plt.title(str((-min(unit[0]) + max(unit[0]))/bins) + " " +str(freq_sum) , fontsize = 40)
        plt.step(edges[:-1],freq_n)
        plt.plot(edges[:-1], list(map(lambda x: log_normal(x,sigma,nu),edges[:-1] )))
        plt.grid()

def PLOT():
    for ind in range(len(name_of_files)):
       file_cloud = LoadFileData(name_of_files[ind])
       file = LoadFileDataWithMap(name_of_files[ind])
       res = Distribution_of_the_condition_with_window(file,10,2)
       name, time_cl, plot_cl = file_cloud
       time, plot = Checkmate_waiting(res)
       time_sq, Sq_average_ = Sq_average(res)
       a,b = np.polyfit(time, plot, deg = 1)
       plt.figure(figsize = (50,40))
       plt.plot(time,list(map(lambda x: a*x + b, time)), label= "y = " + str(a) + "*x + " + str(b))
       plt.scatter(time, plot, marker = "x", s = 1000)
       plt.legend(fontsize=30)
       plt.scatter(time_cl, plot_cl)
       plt.xlabel("time", fontsize = 40)
       plt.ylabel("thikness", fontsize = 40)
       plt.title(name, fontsize = 50)


ind = 0
file_cloud = LoadFileData(name_of_files[ind])
file = LoadFileDataWithMap(name_of_files[ind])
res = Distribution_of_the_condition_with_window(file,10,2)
math_oj = Checkmate_waiting(res)
sq_average = Sq_average(res)
# Function_of_distribution_of_def_prob(res,100)
Function_of_distribution_of_def_prob(res, 100)

# def f(result_in):
#     result = copy.deepcopy(result_in)
#     for unit in result:
#         unit[1] = float(np.sum(unit[1])/2)
#         count = np.array(list((Counter(unit[0]).values())))
#         count_sum = np.sum(count)
#         count = count/count_sum
#         unit.append(list(count))
#     return result


# def log_normal_dist(sq_average_result,checkmate_waiting_result):
#     result = []
#     for ind in range(len(sq_average_result[0])):
#         plt.figure(figsize = (30,20))
#         time = np.array(sq_average_result[0])
#         time = time*10000
#         print(time)
#         print(sq_average_result[1][ind]**2, checkmate_waiting_result[1][ind])
#         res = list(map(lambda x: log_normal(x,sq_average_result[1][ind]**2, checkmate_waiting_result[1][ind]),time))
#         plt.plot(time,res)
#     return result










# def to_make_plot(plt_file,checkmate_waiting_list,error_file,file, name):
#    coeff = np.polyfit(time_list, thikness_list, deg = 1)
#    plt.figure(figsize = (40,30))
#    plt.xscale('log')
#    plt.yscale('log')
#    plt.scatter(checkmate_waiting_list[1],checkmate_waiting_list[0])
#    plt.scatter(file[1],file[2])
#    plt.plot(time_list,list(map(lambda x: coeff[0]*x + coeff[1],time_list)))
#    plt.errorbar(time_list, thikness_list, yerr=error_file, fmt='o', ecolor= 'red', capsize=5)
#    plt.scatter(time_list, thikness_list, marker='x', s=1000)
#    plt.title("thikness(time)" + name,fontsize = 60)
#    plt.xticks(fontsize = 40)
#    plt.yticks(fontsize = 40)
#    plt.plot()
#    plt.ylabel("thikness",fontsize = 50)
#    plt.xlabel("time",fontsize = 50)
#    plt.grid()