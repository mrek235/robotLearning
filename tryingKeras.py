from keras.models import Sequential
from keras.layers import Dense
import numpy as np
import glob


model = Sequential()

model.add(Dense(units=64,activation = 'relu',input_dim = 4096))
model.add(Dense(units=10,activation = 'softmax'))

#model.compile(loss= 'categorical_crossentropy',
 #              optimize = 'sgd',
 #             metrics = 'accuracy')

file_list = glob.glob("*.npz")
#print(file_list)


def file_list_to_positions_list(file_list):
    positions_list = []
    for file_name in file_list:
        file_names_without_npz = file_name.split('.npz')
        file_names_without_npz.remove('')
        for name in file_names_without_npz:
            file_name_without_npz_and_p = name.split('P')
            file_name_without_npz_and_p.remove('')
            floated_sublist = []

            for position in file_name_without_npz_and_p:
                floated_sublist.append(float(position))
            positions_list.append(floated_sublist)

    print(positions_list)
    return positions_list

file_list_to_positions_list(file_list)