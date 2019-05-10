import sklearn
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import glob

file_list = glob.glob("R*.npz")


def file_list_to_positions_list(file_list):
    positions_list = []
    for file_name in file_list:
        file_names_without_npz = file_name.split('.npz')
        file_names_without_npz.remove('')
        for name in file_names_without_npz:
            file_name_without_npz_and_p = name.split('R')
            file_name_without_npz_and_p.remove('')
            floated_sublist = []

            for position in file_name_without_npz_and_p:
                floated_sublist.append(float(position))
            positions_list.append(floated_sublist)

    return positions_list


def npz_to_data_arr(file_list):
    data_arr = []
    for file in file_list:
        data_arr.append((np.load(file))['arr_0'])
    return data_arr


data_arr = npz_to_data_arr(file_list)
positions_list = file_list_to_positions_list(file_list)

#min_max_scaler = MinMaxScaler()

#for i in range(len(data_arr)):
#    data_arr[i] = min_max_scaler.fit_transform(data_arr[i])

def merge_data(data_arr):
    merged_data = []
    for i in range(len(data_arr)):
        sub_data= []
        for j in range (len(data_arr[i])):
            sub_data.append(data_arr[i][j])
        merged_data.append(sub_data)
    return merged_data

X_train = np.array(data_arr).reshape(100,4096,200)
y_train = np.array(positions_list)

X_test = np.array(data_arr[90:len(data_arr)]).reshape(10,4096,200)
y_test = np.array(positions_list[90:len(data_arr)])
