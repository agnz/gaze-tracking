import pandas as pd
import numpy as np
from filterpy.kalman import KalmanFilter
import matplotlib.pyplot as plt

data = pd.read_csv('/Users/ash/Downloads/gaze_positions.csv')

x = np.array(data['norm_pos_x'])
y = np.array(data['norm_pos_y'])

def loss(predict, target):
    assert len(predict) == len(target)
    losses = 0
    for _ in range(len(predict)):
        losses = losses + math.hypot(float(predict[_][0])-float(target[_][0]), float(predict[_][1])-float(target[_][1]))
    return losses

def tracker_2d():
    tracker = KalmanFilter(dim_x=4, dim_z=2)
    tracker.x = np.array([0.0, 0.0, 0.0, 0.0])
    tracker.F = np.array([[1., 0., 1., 0.],
                          [0.,1., 0., 1.],
                          [0., 0., 1., 0.],
                          [0., 0., 0., 1.]])
    tracker.H = np.array([[1., 0., 0., 0.],
                          [0., 1., 0., 0.]])
    tracker.R = np.array([[1., 0],
                          [0, 1.]])
    tracker.P = np.eye(4) * 1000.
    tracker.Q = np.array([[0, 0., 0.001, 0.],
                          [0.001, 0., 0.001, 0.],
                          [0., 0., 0., 0.001],
                          [0., 0.001, 0., 0.001]])
    return tracker



KF_2D = tracker_2d()
preds = []
measure = []
z = np.array([x[0], y[0]])

skip = 1000
step = 400

for _ in range(skip, skip+step):
    KF_2D.predict()
    KF_2D.update(z)
    preds.append([KF_2D.x[0], KF_2D.x[1]])
    measure.append([z[0], z[1]])
    z = np.array(np.array([x[_], y[_]]))



plt.subplot(2,1,1)
plt.title('With Kalman Filter')
plt.plot([preds[_][0] for _ in range(len(preds))], [preds[_][1] for _ in range(len(preds))], color='green', marker='.', mfc='none')

plt.subplot(2,1,2)
plt.title('Without Kalman Filter')
plt.plot([measure[_][0] for _ in range(len(preds))], [measure[_][1] for _ in range(len(preds))], color='blue', marker='.' , mfc='none')

plt.show()

#1480