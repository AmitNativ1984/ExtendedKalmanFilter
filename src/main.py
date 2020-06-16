##
# Main function of the Python program.
#
##

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Fill in Those functions:

def computeRMSE(trueVector, EstimateVector):

    squaredError = 0.0
    for x, y in zip(trueVector, EstimateVector):
        squaredError += np.array(x - y) ** 2

    n = len(trueVector)
    RMSE = 1/n * np.sqrt(squaredError)
    return RMSE

def computeRadarJacobian(Xvector):

    Px, Py, Vx, Vy = np.array(Xvector).squeeze(-1)

    H = np.array([[Px / np.sqrt(Px**2 + Py**2),                     Py / np.sqrt(Px**2 + Py**2),                    0,                                                         0],
                  [-Px / (Px**2 + Py**2),                           Px / (Px**2 + Py**2),                           0,                                                         0],
                  [Py*(Vx*Py - Vy*Px)/(Px**2 + Py**2)**(3/2),       Px*(Vy*Px - Vx*Py)/(Px**2 + Py**2)**(3/2),      Px / np.sqrt(Px**2 + Py**2),     Py / np.sqrt(Px**2 + Py**2)]])

    H = np.asmatrix(H)

    return H

def computeCovMatrix(deltaT, sigma_aX, sigma_aY):

    G = np.array([[deltaT ** 2 / 2.,                 0.],
                    [0.,                 deltaT ** 2. / 2.],
                    [deltaT,                          0.],
                    [0.,                          deltaT]])

    G = np.asmatrix(G)

    Q_ni = np.array([[sigma_aX ** 2.,                0.],
                     [0.,                     sigma_aY]])

    Q_ni = np.asmatrix(Q_ni)


    Q = G * Q_ni * G.transpose()

    return Q

def computeFmatrix(deltaT):

    F = np.matrix([[1.,      0.,  deltaT,          0.],
                   [0.,      1.,      0.,      deltaT],
                   [0.,      0.,      1.,           0.],
                   [0.,      0.,      0.,           1.]])

    return F

def main():
    # we print a heading and make it bigger using HTML formatting
    print("Hellow")
    my_cols = ["A", "B", "C", "D", "E", "f", "g", "h", "i", "j", "k"]
    data = pd.read_csv("./src/data3.txt", names=my_cols, delim_whitespace=True, header=None)
    print(data.head())
    for i in range(10):
        measur = data.iloc[i, :].values
        print(measur[0])


    """
        LIDAR(L) data format:
        sensor_type, x_measured, y_measured, timestamp, x_groundtruth, y_groundtruth, vx_groundtruth, vy_groundtruth, yaw_groundtruth, yawrate_groundtruth. 
        
        Radar(R) data format:
        sensor_type, rho_measured, phi_measured, rhodot_measured, timestamp, x_groundtruth, y_groundtruth, vx_groundtruth, vy_groundtruth, yaw_groundtruth, yawrate_groundtruth. 
    """
    # define matrices:
    deltaT = 0.1
    useRadar = True
    # initial uncertianty matrix. assume 10 for position and 100 for velocities:
    P = np.array([[1,      0.,      0.,      0.],
                  [0.,       1,     0.,      0.],
                  [0.,       0.,      10,    0.],
                  [0.,       0.,      0.,    10]])
    xEstimate = []
    xTrue = []
    # Lidar projection matrix from sensor reading to state vector
    H_Lidar = np.array([[1.,     0.,      0.,      0.],
                        [0.,     1.,      0.,      0.]])

    # lidar noise
    R_lidar = np.array([[0.0225,        0.0],
                        [0.0,        0.0225]])

    # radar noise
    R_radar = np.array([[0.9,       0,          0],
                        [0.0,       0.0009,     0],
                        [0,         0,      0.09]])

    F_matrix = computeFmatrix(deltaT)

    X_state_current = np.array([[0.3],
                                [0.5],
                                [0.],
                                [0.]])
    X_true_current = np.array([[0.3],
                                [0.5],
                                [0.],
                                [0.]])

    U = np.array([[0.],
                  [0.],
                  [0.],
                  [0.]])

    I = np.identity(4)

    firstMeasurment = data.iloc[0, :].values
    timeStamp = firstMeasurment[3]
    # fill in X_true and X_state. Put 0 for the velocities
    plt.figure()
    for index in range(1, len(data)):
        currentMeas = data.iloc[index, :].values

        # compute the current dela t
        if (currentMeas[0] == 'L'):
            deltaT = (currentMeas[3] - timeStamp) / 1000000
            F_matrix = computeFmatrix(deltaT)
            timeStamp = currentMeas[3]

            X_state_current_temp = np.array([[currentMeas[1]],
                                            [currentMeas[2]],
                                            [0],
                                            [0]])

            Lidar_org=plt.scatter(X_state_current_temp[0, 0], X_state_current_temp[1, 0], color='g', marker='^', facecolors='none')

            if X_state_current == []:
                X_state_current = X_state_current_temp

            # perfrom predict
            X_state_current = (F_matrix * X_state_current) + U
            Q = computeCovMatrix(deltaT, sigma_aX=1, sigma_aY=1)
            P = F_matrix * P * F_matrix.transpose() + Q

            # pefrom measurment update
            z = np.array([[currentMeas[1]],
                          [currentMeas[2]]])
            y = z - (H_Lidar * X_state_current)
            S = H_Lidar * P * H_Lidar.transpose() + R_lidar
            K = P * H_Lidar.transpose() * np.linalg.inv(S)
            X_state_current = X_state_current + (K * y)
            P = (I - K * H_Lidar) * P

        if (currentMeas[0] == 'R' and useRadar):

            deltaT = (currentMeas[4] - timeStamp) / 1000000
            timeStamp = currentMeas[4]
            F_matrix = computeFmatrix(deltaT)

            X_true_current = np.array([[currentMeas[5]],
                                       [currentMeas[6]],
                                       [currentMeas[7]],
                                       [currentMeas[8]]])

            rho, phi, rhodot = np.array([currentMeas[1], currentMeas[2], currentMeas[3]])

            X_state_current_temp = np.array([[rho * np.cos(phi)],
                                        [rho * np.sin(phi)],
                                        [0],
                                        [0]])

            radar_org = plt.scatter(X_state_current_temp[0, 0], X_state_current_temp[1, 0], color='pink', marker='^',facecolors='none')


            # perfrom predict

            X_state_current = (F_matrix * X_state_current) + U
            Q = computeCovMatrix(deltaT, sigma_aX=1, sigma_aY=1)
            P = F_matrix * P * F_matrix.transpose() + Q

            # # pefrom measurment update
            jacobian = computeRadarJacobian(X_state_current)

            # convert back to rho, phi, rhdot
            px, py, vx, vy = np.array(X_state_current).squeeze(-1)
            rho = np.sqrt(px**2 + py**2)
            phi = np.arctan2(py, px)
            rhodot =  (px * vx + py * vy) / rho

            z_pred = np.array([[rho],
                                [phi],
                                [rhodot]])



            z = np.array([[currentMeas[1]],
                          [currentMeas[2]],
                          [currentMeas[3]]])
            y = z - z_pred
            while y[1] < np.pi:
                y[1] += 2*np.pi
            while y[1] > np.pi:
                y[1] -= 2*np.pi

            S = jacobian * P * jacobian.transpose() + R_radar
            K = P * jacobian.transpose() * np.linalg.inv(S)
            X_state_current = X_state_current + (K * y)
            P = (I - (K * jacobian)) * P

        kalmanouput = plt.scatter(X_state_current[0, 0], X_state_current[1, 0], color='b', marker='o',facecolors='none')
        truepos = plt.scatter(X_true_current[0, 0], X_true_current[1, 0], color='r', marker='x')
        plt.pause(0.05)
        xEstimate.append(X_state_current)
        xTrue.append(X_true_current)


    rmse = computeRMSE(xEstimate, xTrue)
    print(rmse)
    plt.legend((Lidar_org, radar_org, kalmanouput, truepos),
               ('Lidar Meas', 'Radar Meas', 'Kalman', 'Ground Truth'),
               loc='lower left',
               ncol=2)
    plt.show()

if __name__ == '__main__':
    main()
