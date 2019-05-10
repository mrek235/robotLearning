from keras.models import Sequential
from keras.layers import Dense,Flatten,LSTM,Conv1D,Conv2D, MaxPool1D
from keras.optimizers import Adam,SGD
import numpy as np
from numpy.random import seed
from sklearn.preprocessing import MinMaxScaler
import glob

seed(42)

model = Sequential()

timesteps = 30
data_dim = 4096
num_classes = 6

model = Sequential()
#model.add(Conv2D(64,(3,3), strides = (1,1),  activation= 'tanh'))
#model.add(Conv2D(64,(3,3), strides = (1,1),  activation= 'tanh'))

#model.add(Conv1D(8192,1,  activation= 'linear', input_shape=(100,100)))
#model.add(Conv1D(4096,1,  activation= 'linear' ))

#model.add(Conv1D(2048,1,  activation= 'linear' ))
#model.add(Conv1D(1024,1,  activation= 'linear' ))
#model.add(Conv1D(800,1,  activation= 'linear' ))

#model.add(Conv1D(300,1,  activation= 'linear' ))

#model.add(Conv1D(128,1,  activation= 'linear'))

#model.add(Conv1D(64,1,  activation= 'linear'))
#model.add(Conv1D(16,1, activation= 'linear'))
#model.add(Conv1D(8,1, activation= 'linear'))

#model.add(Dense(32000, activation = 'linear'))
#model.add(Dense(16000, activation= 'tanh'))
#model.add(Dense(8200, activation = 'tanh'))
#model.add(Dense(4096, activation = 'linear'))
#model.add(Dense(2048, activation = 'linear'))
#model.add(Dense(1024, activation = 'linear'))
#model.add(Dense(256, activation= 'linear'))
#model.add(Dense(64, activation = 'linear', input_shape=(100,100)))
#model.add(Dense(32, activation = 'linear'))
#model.add(Dense(16, activation = 'linear'))

#model.add(Dense(1032, activation = 'linear'))  # return a single vector of dimension 32
#model.add(Dense(128, activation = 'linear'))
model.add(Dense(16, activation = 'linear'))

#model.add(Dense(64, activation = 'linear'))
model.add(Dense(8, activation = 'linear'))
model.add(Flatten())
model.add(Dense(5, activation='linear'))
model.compile(loss="mean_squared_error", optimizer=Adam())

#file_list_30_samples = glob.glob("P*.npz")
#file_list_200_samples = glob.glob("R*.npz")
file_list_for_outer_folder = glob.glob("/home/pc/Downloads/Robot_Data/A*.npz") #1.0000052452087402 is black.
print("File list loaded.")
#file_list = glob.glob("A*.npz")

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

def file_list_to_positions_list(file_list):
    positions_list = []
    for file_name in file_list:
        file_names_without_npz = file_name.split('.npz')
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

print("Creating data array")
data_arr = np.array(npz_to_data_arr(file_list_for_outer_folder[0:20000]))
print("Data array created.")

print("Extracting positions array.")
positions_list = np.array(file_list_to_positions_list_from_outer_folder((file_list_for_outer_folder[0:20000])))
print("Positions array extracted.")

def flatten_data(data_arr):
    flattened = []
    for i in range(len(data_arr)):
        new_arr = data_arr[i]
        flattened.append(np.array(data_arr[i]).flatten())
    return flattened



#flattened_data = np.array(flatten_data(data_arr))
#data_arr = None
flattened_data = data_arr

print("Train and test data are prepared.")

ninetieth_percentile = int(len(flattened_data)*0.9)
X_train = flattened_data[0:ninetieth_percentile]
y_train = np.array(positions_list[0:ninetieth_percentile])

#X_test = flattened_data[ninetieth_percentile:len(flattened_data)]
#ly_test = np.array(positions_list[ninetieth_percentile:len(flattened_data)])


print("Starting training.")
model.fit(X_train,y_train, epochs=100, validation_split=0.1)
X_train = None
y_train = None


print("This is the evaluation of model based on test data")
#print(model.evaluate(X_test,y_test))
model.save("firstTryWith30thousandImages")
#print(X_train.shape)
#print(y_train.shape)