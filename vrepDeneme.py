import vrep
import numpy as np
import sys
import time
import random
import thread

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


def create_file_name(position_array):
    file_name = ""
    for i in range(joint_count):
        file_name += "P" + str(position_array[i])
    return file_name


def write_out(file_name,depth_array):
    file_name = (file_name)
    with open(file_name, "wb") as myfile:
        np.savez(file_name, depth_array, delimiter=',')


def record_kinect_depth(position_array):
    resetRobotWithArray()
    give_positions(position_array)
    err, resolution, kinect = vrep.simxGetVisionSensorDepthBuffer(client_id, kinect_depth,
                                                                       vrep.simx_opmode_oneshot_wait)
    kinect_buffer = []
    for i in range(30):
        err, resolution, kinect_buff = vrep.simxGetVisionSensorDepthBuffer(client_id, kinect_depth,
                                                                           vrep.simx_opmode_oneshot_wait)
        if len(kinect_buff) != 0:
            kinect_buffer.append(kinect_buff)


    file_name = create_file_name(position_array)

    if len(kinect_buffer) != 0 :
        write_out(file_name, kinect_buffer)
        vrep.simxGetVisionSensorDepthBuffer(client_id, kinect_depth, vrep.simx_opmode_discontinue)

def random_position_data_generator(data_count):
    for i in range(data_count):
        resetRobotWithArray()
        time.sleep(5)
        record_kinect_depth(random_position_array_generator())


for i in range(5):
    resetRobotWithArray()
    random_position_data_generator(1)


