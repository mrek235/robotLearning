from keras.models import Sequential
from keras.layers import Dense,Flatten,LSTM
from keras.optimizers import Adam
from keras.preprocessing.sequence import TimeseriesGenerator
import numpy as np
import glob


model = Sequential()

timesteps = 30
data_dim = 4096
num_classes = 6

model = Sequential()
model.add(Dense(1024, activation= 'sigmoid', input_shape =(200,4096))) # returns a sequence of vectors of dimension 32
model.add(Dense(512, activation = 'sigmoid'))  # returns a sequence of vectors of dimension 32
model.add(Dense(256, activation = 'sigmoid'))
model.add(Dense(128, activation = 'sigmoid'))  # return a single vector of dimension 32
model.add(Flatten())
model.add(Dense(6, activation='softmax'))


model.compile(loss="mean_squared_error", optimizer=Adam())

file_list = glob.glob("*.npz")


def file_list_to_positions_list(file_list):
    positions_list = []
    for file_name in file_list:
        file_names_without_npz = file_name.split('.npz')
        file_names_without_npz.remove('')
        for name in file_names_without_npz:
            file_name_without_npz_and_p = name.split("R")
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


def merge_data(data_arr):
    merged_data = []
    for i in range(len(data_arr)):
        sub_data= []
        for j in range (len(data_arr[i])):
            sub_data.append(data_arr[i][j])
        merged_data.append(sub_data)
    return merged_data

X_train = np.array(data_arr[0:(0.9(len(data_arr)))])
y_train = np.array(positions_list[0:(0.9(len(data_arr)))])

X_test = np.array(data_arr[(0.9(len(data_arr))):len(data_arr)])
y_test = np.array(positions_list[(0.9(len(data_arr))):len(data_arr)])

model.fit(X_train,y_train, batch_size=4)


pred = model.predict(X_test)
print("X predictions:")
print(pred[0:3])

print("Reals")
print(y_test[0:3])

#score = model.evaluate(X_test, y_test, batch_size=128)
#model.save("firstTryWith30Buffers4Positions")
#print(X_train.shape)
#print(y_train.shape)