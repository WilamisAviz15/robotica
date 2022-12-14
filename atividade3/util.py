import numpy as np
import transformationMatrix

np.set_printoptions(suppress=True, precision=4)

def kine(M0_1, M1_2, M2_3, M3_4, M4_5):
    return M0_1 @ M1_2 @ M2_3 @ M3_4 @ M4_5

def join_matrix(A, B):
    for a, b in zip(A, B):
        yield [*a, *b]

def generateJacobian(q):
    r0_1 = np.array([np.cos(q[0]), 0, -np.sin(q[0]),np.sin(q[0]), 0, np.cos(q[0]),0,-1,0]).reshape(3, 3)
    r1_2 = np.array([np.cos((2*q[1]-np.pi)/2), -np.sin((2*q[1]-np.pi)/2), 0,np.sin((2*q[1]-np.pi)/2), np.cos((2*q[1]-np.pi)/2), 0,0,0, 1]).reshape(3, 3)
    r2_3 = np.array([np.cos((2*q[2]+np.pi)/2), -np.sin((2*q[2]+np.pi)/2), 0,np.sin((2*q[2]+np.pi)/2),  np.cos((2*q[2]+np.pi)/2), 0, 0,0, 1]).reshape(3, 3)
    r3_4 = np.array([np.cos((2*q[3]+np.pi)/2), 0,   np.sin((2*q[3]+np.pi)/2),np.sin((2*q[3]+np.pi)/2), 0, -np.cos((2*q[3]+np.pi)/2),0,  1, 0]).reshape(3, 3)
    r4_5 = np.array([np.cos(q[4]), -np.sin(q[4]), 0, np.sin(q[4]),  np.cos(q[4]), 0,0,0, 1]).reshape(3, 3)

    p0 = np.array([0, 0, 0]).reshape(3, 1)
    p1 = transformationMatrix.M0_1(q[0])[0:3, -1].reshape(3, 1)
    p2 = (transformationMatrix.M0_1(q[0]) @ transformationMatrix.M1_2(q[1]))[0:3, -1].reshape(3, 1)
    p3 = (transformationMatrix.M0_1(q[0]) @ transformationMatrix.M1_2(q[1]) @  transformationMatrix.M2_3(q[2]))[0:3, -1].reshape(3, 1)
    p4 = (transformationMatrix.M0_1(q[0]) @ transformationMatrix.M1_2(q[1]) @ transformationMatrix.M2_3(q[2]) @ transformationMatrix.M3_4(q[3]))[0:3, -1].reshape(3, 1)
    p5 = (transformationMatrix.M0_1(q[0]) @ transformationMatrix.M1_2(q[1]) @ transformationMatrix.M2_3(q[2]) @ transformationMatrix.M3_4(q[3]) @ transformationMatrix.M4_5(q[4]))[0:3, -1].reshape(3, 1)
    
    t0 = np.array([0, 0, 1]).reshape(3, 1)
    t1 = np.dot(r0_1, t0)
    t2 = np.dot(r0_1, r1_2)
    t2 = np.dot(t2, t0)
    t3 = np.dot(r0_1, r1_2)
    t3 = np.dot(t3, r2_3)
    t3 = np.dot(t3, t0)
    t4 = np.dot(r0_1, r1_2)
    t4 = np.dot(t4, r2_3)
    t4 = np.dot(t4, r3_4)
    t4 = np.dot(t4, t0)
    t5 = np.dot(r0_1, r1_2)
    t5 = np.dot(t5, r2_3)
    t5 = np.dot(t5, r3_4)
    t5 = np.dot(t5, r4_5)
    t5 = np.dot(t5, t0)
    x0 = np.cross(t0.T, (p5-p0).T).T
    x1 = np.cross(t1.T, (p5-p1).T).T
    x2 = np.cross(t2.T, (p5-p2).T).T
    x3 = np.cross(t3.T, (p5-p3).T).T
    x4 = np.cross(t4.T, (p5-p4).T).T

    return np.concatenate((list(join_matrix(list(join_matrix(x0, x1)), list(join_matrix(list(join_matrix(x2, x3)), x4)))), list(join_matrix(list(join_matrix(t0, t1)), list(join_matrix(list(join_matrix(t2, t3)), t4))))))


def setArray(Θ1, Θ2, Θ3, Θ4, d4):
    return np.array([Θ1, Θ2, Θ3, Θ4, d4])
