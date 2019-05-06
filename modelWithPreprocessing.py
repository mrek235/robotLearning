from keras.models import Sequential
from keras.layers import Dense,Flatten,LSTM,Conv1D,Conv2D, MaxPool1D
from keras.optimizers import Adam,SGD
import numpy as np
from numpy.random import seed
from sklearn.preprocessing import MinMaxScaler,StandardScaler, RobustScaler
import glob

seed(42)


model = Sequential()

model.add(Dense(16, activation = 'linear'))

model.add(Dense(8, activation = 'linear'))
model.add(Flatten())
model.add(Dense(4, activation='linear'))
model.compile(loss="mean_squared_error", optimizer=Adam())

file_list_for_outer_folder = glob.glob("/home/pc/Downloads/Robot_Data/A*.npz") #1.0000052452087402 is black.
print("File list loaded.")


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
            positions_list.append(floated_sublist[0:4])

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

print("Creating data array")
data_arr = np.array(npz_to_data_arr(file_list_for_outer_folder[0:30000]))
print(data_arr.shape)
print("Data array created.")

print("Scaling data array.")
data_arr = np.array( scale_data(data_arr))
print("Data is scaled.")
print("Printing shape of scaled data")
print(data_arr.shape)

print("Extracting positions array.")
positions_list = np.array(file_list_to_positions_list_from_outer_folder((file_list_for_outer_folder[0:30000])))
print("Positions array extracted.")


print("Train and test data are prepared.")

ninetieth_percentile = int(len(data_arr)*0.9)
X_train = data_arr[0:ninetieth_percentile]
y_train = np.array(positions_list[0:ninetieth_percentile])

X_test = data_arr[ninetieth_percentile:len(data_arr)]
y_test = np.array(positions_list[ninetieth_percentile:len(data_arr)])


print("Starting training.")
model.fit(X_train,y_train, epochs=100, validation_split=0.1)
X_train = None
y_train = None


print("This is the evaluation of model based on test data")
print(model.evaluate(X_test,y_test))
model.save("tryKeras100by100WithPreprocess")


print(model.predict(X_test[0:2]))
print(y_test[0:2])
