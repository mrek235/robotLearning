import vrep
import numpy as np
import sys

dontChangePosition = -500;

# Connection
vrep.simxFinish(-1)
clientID = vrep.simxStart('127.0.0.1', 19997, True, True, 5000, 5)

if clientID != -1:
    print ("Connected to remote API Server")
else:
    print ("Connection not successful")
    sys.exit("Could not connect")

jointHandles=[]
err_code = ''
for i in range(6):
     jointName = 'UR5_joint'+str(i+1)
     err_code,jointHandles.append(vrep.simxGetObjectHandle(clientID, jointName, vrep.simx_opmode_blocking)[1])
     err_code2 = vrep.simxSetJointTargetPosition(clientID, jointHandles[i], 0, vrep.simx_opmode_streaming)

errorCodeKinectDepth, kinectDepth = vrep.simxGetObjectHandle(clientID, 'kinect_depth', vrep.simx_opmode_oneshot_wait)
err,ur5_handle = vrep.simxGetObjectHandle(clientID, 'UR5', vrep.simx_opmode_blocking);


def resetRobot():

    err_code = vrep.simxSetJointTargetPosition(clientID, jointHandles[0],0, vrep.simx_opmode_blocking)
    err_code = vrep.simxSetJointTargetPosition(clientID, jointHandles[1], 0, vrep.simx_opmode_blocking)
    err_code = vrep.simxSetJointTargetPosition(clientID, jointHandles[3], 0, vrep.simx_opmode_blocking)
    err_code = vrep.simxSetJointTargetPosition(clientID, jointHandles[4], 0, vrep.simx_opmode_blocking)
    err,resolution,kinectBuff = vrep.simxGetVisionSensorDepthBuffer(clientID,kinectDepth,vrep.simx_opmode_streaming)
    #appliedMinusFourToThirdJointDepth.append(kinectBuff)

    #joint[0] is scary: it winds basically(a tour is around 6.3), if you give it 200 it will go for 200 times(i think?)
    #joint[1] works with everything - i even tried 50- , masallah
    #joint[2] target position should be less than 2.9
    #joint[3] is also scary: a tour is around 6.3
    #joint[4]: around 6.3 again
    #joint[5]: tip of the head, it isn't important


def givePositions(positionArr):
    for i in range (0,6):
        if( positionArr[i] != dontChangePosition):
            err_code = vrep.simxSetJointTargetPosition(clientID, jointHandles[i], positionArr[i], vrep.simx_opmode_blocking)


def writeOut(fileName,array):
    fileName = (fileName)
    with open(fileName, "wb") as myfile:
        depthArr = array;
        np.savez(fileName, depthArr, delimiter=',')

