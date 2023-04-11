import numpy as np
from .Rotations import axisAngleRot

def RandomRotationMatrix1():

    rv = np.random.rand(3)
    pi = np.pi

    R = np.zeros((3,3))
    theta = rv[0] * pi * 2
    phi   = rv[1] * pi * 2
    z     = rv[2] * 2.0

    r  = np.sqrt(z)
    Vx = np.sin(phi) * r
    Vy = np.cos(phi) * r
    Vz = np.sqrt(2.0 - z)

    st = np.sin(theta)
    ct = np.cos(theta)
    Sx = Vx * ct - Vy * st
    Sy = Vx * st + Vy * ct

    R[0,0] = Vx * Sx - ct
    R[0,1] = Vx * Sy - st
    R[0,2] = Vx * Vz

    R[1,0] = Vy * Sx + st
    R[1,1] = Vy * Sy - ct
    R[1,2] = Vy * Vz

    R[2,0] = Vz * Sx
    R[2,1] = Vz * Sy
    R[2,2] = 1.0 - z

    return R

def RandomRotationMatrix2():
    rv = np.random.rand(3)
    pi = np.pi

    phi = rv[0]*pi*2
    theta = np.arccos(2*rv[1] - 1)
    thetaAroundVector = rv[2]*pi*2

    x = np.array([1,0,0])
    y = np.array([np.sin(theta)*np.cos(phi), np.sin(theta)*np.sin(phi), np.cos(theta)])

    angleBetween = np.arccos(np.dot(x,y)/(np.linalg.norm(x)*np.linalg.norm(y)))
    #print angleBetween

    cross = np.cross(x,y)
    #print cross
    cross = cross/np.linalg.norm(cross)

    R1 = axisAngleRot(cross,angleBetween)
    R2 = axisAngleRot(y,thetaAroundVector)
    #print np.linalg.det(R1),np.linalg.det(R2)

    return R2.dot(R1)

def RandomRotationMatrix():
    return RandomRotationMatrix1()

def RandomRotatedUnitVector3d():
    x = np.array([1.0, 0.0, 0.0])
    R = RandomRotationMatrix()
    return R.dot(x)

def randomDirection():
    x = np.random.rand(3,1)*2 - 1
    l = np.linalg.norm(x)

    while abs(l-1)>0.05:
        x = np.random.rand(3,1)*2 - 1
        l = np.linalg.norm(x)

    x = x / l
    return x

def Euler_zxz(alpha, beta, gamma):
    # Proper euler extrinsic coordinates (z-x-z)
    R = np.dot(np.dot(rotk(gamma), roti(beta)), rotk(alpha))

    return R

def InvEuler_zxz(R):
    alpha = np.arccos( R[2,1] / np.sqrt(1-R[2,2]**2))
    beta  = np.arccos( R[2,2])
    gamma = np.arccos(-R[1,2] / np.sqrt(1-R[2,2]**2))

    return alpha, beta, gamma

def roti(a):
    a = a[0]
    cos = np.cos
    sin = np.sin
    Ri = [[1,      0,       0],
          [0, cos(a), -sin(a)],
          [0, sin(a),  cos(a)]]

    return np.array(Ri)


def rotj(b):
    b = b[0]
    cos = np.cos
    sin = np.sin
    Rj = [[ cos(b), 0, sin(b)],
          [      0, 1,      0],
          [-sin(b), 0, cos(b)]]

    return np.array(Rj)

def rotk(c):
    c = c[0]
    cos = np.cos
    sin = np.sin
    Rk = [[cos(c), -sin(c), 0],
          [sin(c),  cos(c), 0],
          [     0,       0, 1]]

    return np.array(Rk)

def randomCoordinateSystem():
    x1 = randomDirection()
    x2 = randomDirection()

    while abs(np.dot(np.transpose(x1),x2)) > 0.1:
        x2 = randomDirection()

    x2 = x2 - (np.dot(np.transpose(x1),x2)) * x1
    x2 = x2/np.linalg.norm(x2)
    x3 = np.cross(x1.T, x2.T).T
    alpha, beta, gamma = InvEuler_zxz(np.stack([x1,x2,x3], axis=1))

    return alpha, beta, gamma, x1, x2, x3

def jonRandomRotation():
    alpha, beta, gamma, x1, x2, x3 = randomCoordinateSystem()
    R = Euler_zxz(alpha, beta, gamma)
    return R

if __name__ == '__main__':
    import time
    t1 = time.time()
    for i in range(10000):
        RandomRotationMatrix1()
    r1_time = time.time() - t1

    t2 = time.time()
    for i in range(10000):
        RandomRotationMatrix2()
    r2_time = time.time() - t2

    print(r1_time, r2_time)
