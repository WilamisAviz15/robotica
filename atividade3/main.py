import time
import numpy as np
import transformationMatrix
import util
from plot import showAllPlots
from zmqRemoteApi import RemoteAPIClient

SECONDS = 0.05
count = 0
manipulability = []
effector1 = []
effector2 = []
effector3 = []
Θ1 = []
Θ2 = []
Θ3 = []
Θ4 = []
Θ5 = []
joints = []
thetas = []

client = RemoteAPIClient()
sim = client.getObject('sim')
dummy_handle = sim.getObject('/Dummy')
position = sim.getObjectPosition(dummy_handle, -1)
orientation = sim.getObjectQuaternion(dummy_handle, -1)
(alpha, beta, gamma) = sim.getObjectOrientation(dummy_handle, -1)
(yaw, pitch, roll) = sim.alphaBetaGammaToYawPitchRoll(alpha, beta, gamma)
np.set_printoptions(suppress=True, precision=4)
X_d = np.array([[position[0]], [position[1]], [position[2]], [roll], [pitch], [yaw]])

joints.append(sim.getObject('/theta1'))
joints.append(sim.getObject('/theta2'))
joints.append(sim.getObject('/theta3'))
joints.append(sim.getObject('/theta4'))
joints.append(sim.getObject('/theta5'))
thetas.append(sim.getJointPosition(joints[0]))
thetas.append(sim.getJointPosition(joints[1]))
thetas.append(sim.getJointPosition(joints[2]))
thetas.append(sim.getJointPosition(joints[3]))

q = util.setArray(thetas[0], thetas[1], thetas[2], thetas[3], sim.getJointPosition(joints[4]))
q_cloned = q
TransformationMatrix = util.kine(transformationMatrix.M0_1(q[0]),transformationMatrix.M1_2(q[1]), transformationMatrix.M2_3(q[2]),  transformationMatrix.M3_4(q[3]) , transformationMatrix.M4_5(q[4]))
R = TransformationMatrix[0:3, 0:3]

for i in range(3):
    for j in range(3):
        if (np.abs(R[i][j]) < 0.01):
            R[i][j] = 0.0

alpha = np.arctan2(R[1][0], R[0][0])
beta = np.arctan2(-R[2][0], np.sqrt( (R[2][1]**2) + (R[2][2]**2)))
gamma = np.arctan2(R[2][1], R[2][2])
X_m = np.vstack([TransformationMatrix[0:3, -1].reshape(3,1), np.array([alpha, beta, gamma]).reshape((3, 1))])
effector = sim.getObject('/end_effector_visual')

while (count < 15):
    jacobian = util.generateJacobian(q)
    try:
        manipulability.append(np.sqrt(np.linalg.det(jacobian@jacobian.T)))
    except:
        pass
    
    (x, y, z)  = sim.getObjectPosition(effector, -1)
    effector1.append(x)
    effector2.append(y)
    effector3.append(z)
    Θ1.append(q[0])
    Θ2.append(q[1])
    Θ3.append(q[2])
    Θ4.append(q[3])
    Θ5.append(q[4])
    difKinematic = (np.transpose(jacobian) @ np.linalg.inv(jacobian @ np.transpose(jacobian) + 0.5**2 * np.eye(6))) @ (X_d - X_m)
    difKinematic = difKinematic.reshape((5, ))
    q += difKinematic * SECONDS
    sim.setJointTargetPosition(joints[0], q[0])
    sim.setJointTargetPosition(joints[1], q[1])
    sim.setJointTargetPosition(joints[2], q[2])
    sim.setJointTargetPosition(joints[3], q[3])
    sim.setJointTargetPosition(joints[4], q[4])

    time.sleep(SECONDS)

    q = util.setArray(sim.getJointPosition(joints[0]), sim.getJointPosition(joints[1]), sim.getJointPosition(joints[2]), sim.getJointPosition(joints[3]), sim.getJointPosition(joints[4]))
    TransformationMatrix = util.kine(transformationMatrix.M0_1(q[0]),transformationMatrix.M1_2(q[1]), transformationMatrix.M2_3(q[2]),  transformationMatrix.M3_4(q[3]) , transformationMatrix.M4_5(q[4]))
    R = TransformationMatrix[0:3, 0:3]

    for i in range(0, 3):
        for j in range(0, 3):
            if (np.abs(R[i][j]) < 0.1):
                R[i][j] = 0.0

    alpha = np.arctan2(R[1][0], R[0][0])
    beta = np.arctan2(-R[2][0], np.sqrt( (R[2][1]**2) + (R[2][2]**2)))
    gamma = np.arctan2(R[2][1], R[2][2])
    
    X_m = np.vstack([TransformationMatrix[0:3, -1].reshape(3,1), np.array([alpha, beta, gamma]).reshape((3, 1))])
    position = sim.getObjectPosition(dummy_handle, -1)
    orientation = sim.getObjectQuaternion(dummy_handle, -1)
    (alpha, beta, gamma) = sim.getObjectOrientation(dummy_handle, -1)
    (yaw, pitch, roll) = sim.alphaBetaGammaToYawPitchRoll(alpha, beta, gamma)
    X_d = np.array([[position[0]], [position[1]], [position[2]], [roll], [pitch], [yaw]])
    count += SECONDS

showAllPlots(q_cloned, q, np.arange(0, 15, SECONDS), manipulability, [effector1, effector2, effector3], [Θ1, Θ2, Θ3, Θ4, Θ5])