from keras.models import load_model

model = load_model('tryKeras100by100WithPreprocess')
#model = load_model('firstTryWith30thousandImages')
model.compile(optimizer = 'adam', loss = 'mean_squared_error', metrics = ['accuracy'])

from keras.models import Sequential
from keras.layers import Dense,Flatten,LSTM,Conv1D,Conv2D, MaxPool1D
from keras.optimizers import Adam,SGD
import numpy as np
from numpy.random import seed
from sklearn.preprocessing import MinMaxScaler,StandardScaler, RobustScaler,Normalizer
import glob

seed(42)

joint_0_coef = 6.3
joint_1_coef = 20
joint_2_coef = 2.9
joint_3_coef = 6.3
joint_4_coef = 6.3
joint_5_coef = 1

joint_coef_arr = [joint_0_coef,joint_1_coef,joint_2_coef,joint_3_coef,joint_4_coef,joint_5_coef]

def file_list_to_positions_list_from_outer_folder(file_list):
    positions_list = []
    for file_name in file_list:
        file_names_without_path = file_name.split('/home/pc/Downloads/Robot_Data/')
        file_names_without_path.remove('')
        file_names_without_npz = file_names_without_path[0].split('.npz')
        file_names_without_npz.remove('')
        for name in file_names_without_npz:
            file_name_without_npz_and_p = name.split('A')
            file_name_without_npz_and_p.remove('')
            floated_sublist = []
            for position in file_name_without_npz_and_p:

                floated_sublist.append(float(position))
            positions_list.append(floated_sublist[0:5])

    return positions_list


def npz_to_data_arr(file_list):
    data_arr = []
    for file in file_list:
        loaded_file = (np.load(file))['arr_0']
        last_position_data = loaded_file[len(loaded_file)-1]
        last_position_data = last_position_data.reshape(100,100)
        data_arr.append(last_position_data)
    return data_arr

def flatten_data(data_arr):
    flattened = []
    for i in range(len(data_arr)):
        new_arr = data_arr[i]
        flattened.append(np.array(data_arr[i]).flatten())
    return flattened

def scale_data(data_arr):
    scaler = RobustScaler()
    scaled = []
    for i in range(len(data_arr)):
        new_arr = data_arr[i]
        scaled.append(scaler.fit_transform(np.array(data_arr[i])))
    return scaled

def normalize_output_to_minus_1_to_1(y):
    normalized_output = []
    for i in range(len(y)):
        normalized_output.append([])
        for j in range(len(y[i])):
            normalized_output[i].append(y[i][j]/joint_coef_arr[j])
    return normalized_output;

def return_min_for_joint(j, positions_list):
    joint_i_arr = []
    for i in range(len(positions_list)):
        joint_i_arr.append(positions_list[i][j])
    return np.amin(np.array(joint_i_arr))

def return_max_for_joint(j, positions_list):
    joint_i_arr = []
    for i in range(len(positions_list)):
        joint_i_arr.append(positions_list[i][j])
    return np.amax(np.array(joint_i_arr))

def normalize_output(y):
    normalized_output = []
    for i in range(len(y)):
        normalized_output.append([])
        for j in range(len(y[i])):
            min = return_min_for_joint(j,y)

            max = return_max_for_joint(j,y)

            normalized_output[i].append((y[i][j]-min)/(max-min))
    return normalized_output;

def denormalize_output(y,positions_list):
    denormalized_output = []
    for i in range(len(y)):
        denormalized_output.append([])
        for j in range(len(y[i])):
            min = return_min_for_joint(j,positions_list)
            max = return_max_for_joint(j,positions_list)
            denormalized_output[i].append(((y[i][j])*(max-min))+min)
    return np.array(denormalized_output);

def write_out(file_name,depth_array):
    file_name = (file_name)
    with open(file_name, "wb") as myfile:
        np.savez(file_name, depth_array, delimiter=',')


file_list_for_outer_folder = glob.glob("/home/pc/Downloads/Robot_Data/A*.npz") #1.0000052452087402 is black.
print("File list loaded.")

print("Creating data array")
data_arr = np.array(npz_to_data_arr(file_list_for_outer_folder[24300:50000]))
print(data_arr.shape)
print("Data array created.")

print("Scaling data array.")
data_arr = np.array( scale_data(data_arr))
print("Data is scaled.")
print("Printing shape of scaled data")
print(data_arr.shape)

print("Extracting positions array.")
positions_list = file_list_to_positions_list_from_outer_folder(file_list_for_outer_folder[85000:110000])
print("Positions array extracted.")
file_list_for_outer_folder = None

print("Predicting results.")
test_data_prediction = model.predict(data_arr)

print("Denormalizing results.")
test_data_prediction = denormalize_output(test_data_prediction,positions_list)

print("Writing out the results.")
file_name = "test_results_of_test_data" + str(len(test_data_prediction))
test_data_and_predictions = [positions_list,test_data_prediction]
write_out(file_name, test_data_and_predictions)






