import vrep
import numpy as np
import sys
import time
import random
import glob



dontChangePosition = -500;

# Connection
vrep.simxFinish(-1)
client_id = vrep.simxStart('127.0.0.1', 19997, True, True, 5000, 5)

if client_id != -1:
    print ("Connected to remote API Server")
else:
    print ("Connection not successful")
    sys.exit("Could not connect")

jointHandles=[]
joint_count = 6

for i in range(joint_count):
    jointName = 'UR5_joint'+str(i+1)
    jointHandles.append(vrep.simxGetObjectHandle(client_id, jointName, vrep.simx_opmode_blocking)[1])
    err_code = vrep.simxSetJointTargetPosition(client_id, jointHandles[i], 0, vrep.simx_opmode_streaming)

error_code_kinect_depth, kinect_depth = vrep.simxGetObjectHandle(client_id, 'kinect_depth', vrep.simx_opmode_oneshot_wait)
err_ur5,ur5_handle = vrep.simxGetObjectHandle(client_id, 'UR5', vrep.simx_opmode_blocking);

def resetRobotWithArray():
    give_positions([0,0,0,0,0,0])

def random_position_array_generator():
    return [random.uniform(-6.3,6.3),random.uniform(-20,20), random.uniform(-2.9,2.9),random.uniform(-6.3,6.3),
            random.uniform(-6.3,6.3),random.randrange(0,1)]

def restricted_random_position_array_generator():
    return [random.uniform(-3.14, 3.14), random.uniform(-5, 5), random.uniform(-1.5, 1.5), random.uniform(-3.14, 3.14),
            random.uniform(-3.14, 3.14), 0]


#joint[0] is scary: it winds basically(a tour is around 6.3), if you give it 200 it will go for 200 times(i think?)
#joint[1] works with everything - i even tried 50- , masallah
#joint[2] target position should be less than 2.9
#joint[3] is also scary: a tour is around 6.3
#joint[4]: around 6.3 again
#joint[5]: tip of the head, it isn't important

def give_positions(position_array):
    for i in range (0,6):
        if position_array[i] != dontChangePosition:
            error_code = vrep.simxSetJointTargetPosition(client_id, jointHandles[i], position_array[i], vrep.simx_opmode_blocking)

def getJointPositions():
    pos = []
    for i in range(0,6):
        pos.append((vrep.simxGetJointPosition(client_id, jointHandles[i], operationMode=vrep.simx_opmode_oneshot_wait))[1])
    return pos

def joints_in_position(position_array):
    current_positions = getJointPositions()
    delta = 0.2
    all_in_position = False
    for i in range(len(current_positions)):
        if(abs((current_positions[i] - position_array[i])) <= 0.2):
            #print("joint")
            #print(i)
            #print("is in position")
            all_in_position = True
        else:
            all_in_position = False
    return all_in_position


def create_file_name(position_array,sample_size):
    file_name = ""
    for i in range(joint_count):
        if sample_size <= 3:
            file_name += "C" + str(position_array[i]) # 128x128
        elif sample_size <= 5:
            file_name += "A" + str(position_array[i])  # A is for 100x100 images with full blackness
        elif sample_size <= 25:
            file_name += "G" + str(position_array[i]) #G is for 100x100 images
        elif(sample_size<= 30):
            file_name += "S" + str(position_array[i]) #"S" is for 128x128 images, "P" is for 64x64
        elif (sample_size <= 200):
            file_name += "R" + str(position_array[i])
    return file_name


def write_out(file_name,depth_array):
    file_name = (file_name)
    with open(file_name, "wb") as myfile:
        np.savez(file_name, depth_array, delimiter=',')


def record_kinect_depth(position_array,sample_size):
    resetRobotWithArray()
    give_positions(position_array)
    err, resolution, kinect_buff = vrep.simxGetVisionSensorDepthBuffer(client_id, kinect_depth,
                                                                       vrep.simx_opmode_oneshot_wait)
    kinect_buffer = []

    while joints_in_position(position_array) == False:
        print("why am i here?")

    if (joints_in_position(position_array)):
        err, resolution, kinect_buff = vrep.simxGetVisionSensorDepthBuffer(client_id, kinect_depth,
                                                                           vrep.simx_opmode_oneshot_wait)
        err, resolution, kinect_buff = vrep.simxGetVisionSensorDepthBuffer(client_id, kinect_depth,
                                                                           vrep.simx_opmode_oneshot_wait)
        kinect_buffer.append(kinect_buff)

    file_name = create_file_name(position_array,sample_size)

    if len(kinect_buffer) != 0 :
        write_out(file_name, kinect_buffer)
        vrep.simxGetVisionSensorDepthBuffer(client_id, kinect_depth, vrep.simx_opmode_discontinue)

def random_position_data_generator(data_count,sample_size):
    for i in range(data_count):
        resetRobotWithArray()
        time.sleep(1)
        record_kinect_depth(restricted_random_position_array_generator(),sample_size)


for i in range(3000):
    print(i)
    resetRobotWithArray()
    random_position_data_generator(25,3)


#testing keras models

#file_path_for_test_results = glob.glob("test_results_of_test_data25700.npz")[0]
#data_arr = (np.load(file_path_for_test_results))['arr_0']
#predictions = data_arr[0]
#reals = data_arr[1]


def predictions_test(predictions,reals):
    for i in range(len(predictions)):
        print(i)
        prediction = np.append(predictions[i],0)
        real = np.append(reals[i],0)

        resetRobotWithArray()
        time.sleep(5)
        give_positions(prediction)
        time.sleep(5)
        resetRobotWithArray()
        time.sleep(3)
        give_positions(real)
        time.sleep(5)

#predictions_test(predictions,reals)
