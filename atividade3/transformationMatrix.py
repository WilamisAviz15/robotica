import numpy as np
np.set_printoptions(suppress=True, precision=4)

def M0_1(Θ):
    return np.array(
        [
            np.cos(Θ), 0, -np.sin(Θ), 0,
            np.sin(Θ), 0,  np.cos(Θ), 0,
            0,-1,0,0.11,
            0,0,0,1
        ]
    ).reshape(4, 4)

def M1_2(Θ):
    return np.array(
        [
            np.cos((2*Θ-np.pi)/2), -np.sin((2*Θ-np.pi)/2), 0, 0.125*np.cos((2*Θ-np.pi)/2),
            np.sin((2*Θ-np.pi)/2),  np.cos((2*Θ-np.pi)/2), 0, 0.125*np.sin((2*Θ-np.pi)/2),
            0,0,1,0,
            0,0, 0,1
        ]
    ).reshape(4, 4)

def M2_3(Θ):
    return np.array(
        [
            np.cos((2*Θ+np.pi)/2), -np.sin((2*Θ+np.pi)/2), 0, 0.096*np.cos((2*Θ+np.pi)/2),
            np.sin((2*Θ+np.pi)/2),  np.cos((2*Θ+np.pi)/2), 0, 0.096*np.sin((2*Θ+np.pi)/2),
            0, 0, 1, 0,
            0, 0, 0, 1
        ]
    ).reshape(4, 4)

def M3_4(Θ):
    return np.array(
        [
            np.cos((2*Θ+np.pi)/2), 0,   np.sin((2*Θ+np.pi)/2), -0.0275*np.cos((2*Θ+np.pi)/2),
            np.sin((2*Θ+np.pi)/2), 0,  -np.cos((2*Θ+np.pi)/2), -0.0275*np.sin((2*Θ+np.pi)/2),
            0, 1, 0, 0,
            0, 0, 0, 1
        ]
    ).reshape(4, 4)

def M4_5(Θ):
    return np.array(
        [
            np.cos(Θ), -np.sin(Θ), 0, 0,
            np.sin(Θ),  np.cos(Θ), 0, 0,
            0, 0, 1, 0.065,
            0, 0, 0, 1
        ]
    ).reshape(4, 4)