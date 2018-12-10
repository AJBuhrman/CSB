import numpy as np
import math as m
import os
import random

def GetData(folder_name):
    wd = os.getcwd()
    fnames = os.listdir(wd + folder_name)
    big_array = np.vstack([np.loadtxt(wd + folder_name + f, delimiter=' ') for f in fnames if os.path.isfile(wd + folder_name + f)])
    
    np.random.shuffle(big_array)

    input_matrix = big_array[:, :-2]
    output_matrix = big_array[:, -2:]

    input_avg = input_matrix.mean(axis=0)
    input_stddev = input_matrix.std(axis=0)

    output_avg = output_matrix.mean(axis=0)
    output_stddev = output_matrix.std(axis=0)

    rows_to_strip = []
    for i, x in enumerate(input_matrix):
        if not (all(input_avg - 3*input_stddev <= x) and all(x <= input_avg + 3*input_stddev)):
            rows_to_strip.append(i)

    rows_to_strip = []
    for i, x in enumerate(output_matrix):
        if not (all(output_avg - 3*output_stddev <= x) and all(x <= output_avg + 3*output_stddev)):
            if not i in rows_to_strip: rows_to_strip.append(i)

    input_stripped = np.delete(input_matrix, rows_to_strip, axis=0)
    input_min_std = [0,0,0,-m.pi,0,-m.pi,0,0,0,0,-m.pi,0,0,0,0,0,-m.pi,0,0,0]
    input_max_std = [1,0,0,m.pi,25000,m.pi,25000,1,1,2*m.pi,m.pi,25000,0,0,1,2*m.pi,m.pi,25000,0,0]

    input_min = np.array([input_stripped.min(axis=0), input_min_std]).min(axis=0)
    input_max = np.array([input_stripped.max(axis=0),input_max_std]).max(axis=0)

    output_stripped = np.delete(output_matrix, rows_to_strip, axis=0)
    output_min = np.array([output_stripped.min(axis=0), [-0.31, 0]]).min(axis=0)
    output_max = np.array([output_stripped.max(axis=0), [0.31, 200]]).max(axis=0)

    input_norm = np.round((input_stripped - input_min) / (input_max - input_min),2)
    output_norm = np.round((output_stripped - output_min) / (output_max - output_min),2)

    return (input_norm, output_norm, input_min, input_max)
