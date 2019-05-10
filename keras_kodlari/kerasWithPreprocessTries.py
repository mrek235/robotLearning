from keras.models import Sequential
from keras.layers import Dense,Flatten,LSTM,Conv1D,Conv2D, MaxPool1D
from keras.optimizers import Adam,SGD
import numpy as np
from numpy.random import seed
from sklearn.preprocessing import MinMaxScaler,StandardScaler, RobustScaler
import glob
from sklearn.model_selection import train_test_split

seed(42)


model = Sequential()

#model.add(Dense(64,activation='relu'))
#model.add(Dense(32, activation = 'relu'))
model.add(Dense(16, activation = 'relu'))

model.add(Dense(8, activation = 'sigmoid'))
model.add(Dense(4,activation ='sigmoid'))
model.add(Flatten())
model.add(Dense(5, activation='sigmoid'))
model.compile(loss="mean_squared_error", optimizer=Adam() )


joint_0_coef = 6.3
joint_1_coef = 20
joint_2_coef = 2.9
joint_3_coef = 6.3
joint_4_coef = 6.3
joint_5_coef = 1

joint_coef_arr = [joint_0_coef,joint_1_coef,joint_2_coef,joint_3_coef,joint_4_coef,joint_5_coef]

#file_list_for_outer_folder = glob.glob("/home/pc/Downloads/Robot_Data/A*.npz") #1.0000052452087402 is black.
file_list_for_outer_folder = glob.glob("C*.npz")
print("File list loaded.")


def file_list_to_positions_list_from_outer_folder(file_list):
    positions_list = []
    for file_name in file_list:
        file_names_without_path = file_name.split('/home/pc/Downlo_without_path[0]ads/Robot_Data/')
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
        print(file_names_without_npz)
        for name in file_names_without_npz:
            print(name)
            file_name_without_npz_and_p = name.split('C')
            print(file_name_without_npz_and_p)
            file_name_without_npz_and_p.remove('')
            print(file_name_without_npz_and_p)
            floated_sublist = []
            for position in file_name_without_npz_and_p:
                floated_sublist.append(float(position))
            positions_list.append(floated_sublist[0:5])
    print(positions_list)
    return positions_list



def npz_to_data_arr(file_list):
    data_arr = []
    for file in file_list:
        loaded_file = (np.load(file))['arr_0']
        last_position_data = loaded_file[len(loaded_file)-1]
        last_position_data = last_position_data.reshape(128,128)
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


def deprecated_normalize_output(y):
    normalized_output = []
    for i in range(len(y)):
        normalized_output.append([])
        for j in range(len(y[i])):
            normalized_output[i].append(((y[i][j]/joint_coef_arr[j])+1)/2)
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


print("Creating data array")
data_arr = np.array(npz_to_data_arr(file_list_for_outer_folder[0:40000]))
print(data_arr.shape)
print("Data array created.")

print("Scaling data array.")
data_arr = np.array( scale_data(data_arr))
print("Data is scaled.")
print("Printing shape of scaled data")
print(data_arr.shape)

print("Extracting positions array.")
positions_list = np.array(file_list_to_positions_list((file_list_for_outer_folder[0:40000])))
print("Positions array extracted.")

print("Normalizing array.")
normalized_positions_list = np.array(normalize_output(positions_list))
print("Positions array normalized to unit difference.")


X_train, X_test, y_train, y_test = train_test_split(data_arr, normalized_positions_list, test_size=0.2, random_state=42)

print("Train and test data are prepared.")

#ninetieth_percentile = int(len(data_arr)*0.9)
#X_train = data_arr[0:ninetieth_percentile]
#y_train = np.array(normalized_positions_list[0:ninetieth_percentile])

#X_test = data_arr[ninetieth_percentile:len(data_arr)]
#y_test = np.array(normalized_positions_list[ninetieth_percentile:len(data_arr)])
data_arr = None
#positions_list = None

print("Starting training.")
model.fit(X_train,y_train, epochs=100, validation_split=0.1)
X_train = None
y_train = None


print("This is the evaluation of model based on test data")
print(model.evaluate(X_test,y_test))
model.save("tryKeras128by128WithPreprocess40KData")

