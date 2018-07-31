from simulation import BoundBox, box_visualize, Center_2D, draw_some_box
from tracker import tracker_2dof, tracker_3dof, tracker_4dof
from filterpy.kalman import KalmanFilter
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
import math

def loss(predict, target):
    assert len(predict) == len(target)
    losses = 0
    for _ in range(len(predict)):
        losses = losses + math.hypot(float(predict[_][0])-float(target[_][0]), float(predict[_][1])-float(target[_][1]))
    return losses

'''
Tracking a point moving in 2D
'''

# step = 100
# c = Center_2D(sigma=8, v=(6,2))
#
# KF_2D = tracker_3dof()
# preds = []
# truths = []
# measure = []
# res_x = []
# res_y = []
# count = [_ for _ in range(step)]
# z = np.array([0, 0])
#
# for _ in range(step):
#     KF_2D.predict()
#     KF_2D.update(z)
#     preds.append([KF_2D.x[0], KF_2D.x[1]])
#     truths.append([c.pos[0], c.pos[1]])
#     measure.append([z[0], z[1]])
#     z = np.array(c.read())
#     res_x.append(KF_2D.y[0])
#     res_y.append(KF_2D.y[1])
#
# print('Loss w/ Filter:', loss(truths, preds)/step)
# print('Loss wt/ Filter:', loss(truths, measure)/step)
#
#
# plt.subplot(2,1,1)
# plt.title('With Kalman Filter')
# plt.plot([truths[_][0] for _ in range(len(preds))], [truths[_][1] for _ in range(len(preds))], color='red')
# plt.plot([preds[_][0] for _ in range(len(preds))], [preds[_][1] for _ in range(len(preds))], color='green', marker='.')
#
# plt.subplot(2,1,2)
# plt.title('Without Kalman Filter')
# plt.plot([truths[_][0] for _ in range(len(preds))], [truths[_][1] for _ in range(len(preds))], color='red')
# plt.plot([measure[_][0] for _ in range(len(preds))], [measure[_][1] for _ in range(len(preds))], color='blue', marker='+')
#
# plt.show()

'''
Stabilizing a still box by tracking its center 
'''

# step = 100
# box = BoundBox(sigma=5)
#
# KF_2D = tracker_2d()
# preds = []
# truths = []
# measure = []
# z = np.array([0, 0])
#
# for _ in range(step):
#     KF_2D.predict()
#     KF_2D.update(z)
#     preds.append([KF_2D.x[0], KF_2D.x[1]])
#     truths.append([box.cen[0], box.cen[1]])
#     measure.append([z[0], z[1]])
#     cen_n, _, _, _, _ = box.read()
#     z = cen_n
#
#
#
# print('Loss w/ Filter:', loss(truths, preds)/step)
# print('Loss wt/ Filter:', loss(truths, measure)/step)
#
# plt.subplot(2,1,1)
# plt.xlim(-10, 10)
# plt.ylim(-10, 10)
# plt.title('With Kalman Filter')
# plt.scatter([truths[_][0] for _ in range(len(preds))], [truths[_][1] for _ in range(len(preds))], color='red')
# plt.scatter([preds[_][0] for _ in range(len(preds))], [preds[_][1] for _ in range(len(preds))], color='green', marker='.')
#
# plt.subplot(2,1,2)
# plt.xlim(-10, 10)
# plt.ylim(-10, 10)
# plt.title('Without Kalman Filter')
# plt.scatter([truths[_][0] for _ in range(len(preds))], [truths[_][1] for _ in range(len(preds))], color='red')
# plt.scatter([measure[_][0] for _ in range(len(preds))], [measure[_][1] for _ in range(len(preds))], color='blue', marker='+')
#
# plt.show()

'''
Stabilizing a still box by tracking its corners
'''

# step = 100
# box = BoundBox(sigma=2, linestyle='-')
#
# KF_a = tracker_2d()
# KF_b = tracker_2d()
# KF_c = tracker_2d()
# KF_d = tracker_2d()
#
# preds = []
# measure = []
# truths = []
# z = np.array([[0, 0], [0, 0], [0, 0], [0, 0]])
#
# for _ in range(step):
#     KF_a.predict()
#     KF_a.update(z[0])
#     KF_b.predict()
#     KF_b.update(z[1])
#     KF_c.predict()
#     KF_c.update(z[2])
#     KF_d.predict()
#     KF_d.update(z[3])
#     preds.append([KF_a.x[:2], KF_b.x[:2], KF_c.x[:2], KF_d.x[:2]])
#     truths.append([box.a, box.b, box.c, box.d])
#     measure.append(z)
#     cen_n, a_n, b_n, c_n, d_n = box.read()
#     z = np.array([a_n, b_n, c_n, d_n])
#
# plt.subplot(2,1,1)
# plt.xlim(-10, 10)
# plt.ylim(-10, 10)
# plt.title('With Kalman Filter')
# for _ in range(step):
#     box.draw_box()
#     draw_some_box(preds[_][0], preds[_][1], preds[_][2], preds[_][3], col = 'green')
#
# plt.subplot(2,1,2)
# plt.xlim(-10, 10)
# plt.ylim(-10, 10)
# plt.title('Without Kalman Filter')
# for _ in range(step):
#     box.draw_box()
#     draw_some_box(measure[_][0], measure[_][1], measure[_][2], measure[_][3], col = 'blue')
#
# plt.show()

'''
Tracking the corners of a box moving in 2D wt/ noise in size
'''

# step = 60
# box = BoundBox(v=(2,3), sigma=5, linestyle='-')
#
# KF_a = tracker_2d()
# KF_b = tracker_2d()
# KF_c = tracker_2d()
# KF_d = tracker_2d()
#
# preds = []
# measure = []
# truths = []
# cen_preds = []
# cen_measure = []
# cen_truths = []
# z = np.array([[0, 0], [0, 0], [0, 0], [0, 0]])
#
# for _ in range(step):
#     KF_a.predict()
#     KF_a.update(z[0])
#     KF_b.predict()
#     KF_b.update(z[1])
#     KF_c.predict()
#     KF_c.update(z[2])
#     KF_d.predict()
#     KF_d.update(z[3])
#     cen_n, a_n, b_n, c_n, d_n = box.read()
#     z = np.array([a_n, b_n, c_n, d_n])
#     preds.append([KF_a.x[:2], KF_b.x[:2], KF_c.x[:2], KF_d.x[:2]])
#     truths.append([box.a, box.b, box.c, box.d])
#     measure.append(z)
#     cen_preds.append([(KF_a.x[0] + KF_b.x[0]) / 2, (KF_a.x[1] + KF_c.x[1]) / 2])
#     cen_truths.append([box.cen[0], box.cen[1]])
#     cen_measure.append([cen_n[0], cen_n[1]])
#
# plt.subplot(2,1,1)
# plt.title('With Kalman Filter')
# for _ in range(step):
#     draw_some_box(preds[_][0], preds[_][1], preds[_][2], preds[_][3], col = 'palegreen')
# plt.plot([cen_preds[_][0] for _ in range(len(cen_preds))], [cen_preds[_][1] for _ in range(len(cen_preds))], color='green', marker='.')
#
# plt.subplot(2,1,2)
# plt.title('Without Kalman Filter')
# for _ in range(step):
#     draw_some_box(measure[_][0], measure[_][1], measure[_][2], measure[_][3], col = 'skyblue')
# plt.plot([cen_measure[_][0] for _ in range(len(cen_preds))], [cen_measure[_][1] for _ in range(len(cen_preds))], color='blue', marker='+')
# plt.show()

'''
Stabilizing a box changing in size, not moving
'''
# dt = 1.0
# step = 100
# box = BoundBox(v=(0,0), sigma=3, linestyle='-')
#
# KF_a = tracker_2d()
# KF_b = tracker_2d()
# KF_c = tracker_2d()
# KF_d = tracker_2d()
#
# preds = []
# measure = []
# truths = []
# z = np.array([[0, 0], [0, 0], [0, 0], [0, 0]])
#
# for _ in range(step):
#     KF_a.predict()
#     KF_a.update(z[0])
#     KF_b.predict()
#     KF_b.update(z[1])
#     KF_c.predict()
#     KF_c.update(z[2])
#     KF_d.predict()
#     KF_d.update(z[3])
#     a_n, b_n, c_n, d_n = box.change_size()
#     z = np.array([a_n, b_n, c_n, d_n])
#     preds.append([[KF_a.x[0], KF_a.x[1]], [KF_b.x[0], KF_a.x[1]], [KF_a.x[0], KF_c.x[1]], [KF_b.x[0], KF_d.x[1]]])
#     truths.append([box.a, box.b, box.c, box.d])
#     measure.append(z)
#
# plt.subplot(2,1,1)
# plt.title('With Kalman Filter')
# plt.xlim(-10,10)
# plt.ylim(-10,10)
# for _ in range(step):
#     draw_some_box(preds[_][0], preds[_][1], preds[_][2], preds[_][3], col = 'palegreen')
#
# plt.subplot(2,1,2)
# plt.title('Without Kalman Filter')
# plt.xlim(-10,10)
# plt.ylim(-10,10)
# for _ in range(step):
#     draw_some_box(measure[_][0], measure[_][1], measure[_][2], measure[_][3], col = 'skyblue')
# plt.show()

'''
Tracking the corners of a box moving in 2D w/ noise in size
'''

# step = 50
# to_update = 30
# box = BoundBox(v=(3,4), sigma=5, linestyle='-')
#
# KF_a = tracker_2d()
# KF_b = tracker_2d()
# KF_c = tracker_2d()
# KF_d = tracker_2d()
#
# preds = []
# measure = []
# truths = []
# cen_preds = []
# cen_measure = []
# cen_truths = []
# z = np.array([[0, 0], [0, 0], [0, 0], [0, 0]])
#
# for _ in range(to_update):
#     KF_a.predict()
#     KF_a.update(z[0])
#     KF_b.predict()
#     KF_b.update(z[1])
#     KF_c.predict()
#     KF_c.update(z[2])
#     KF_d.predict()
#     KF_d.update(z[3])
#     cen_n, a_n, b_n, c_n, d_n = box.change_size_n_move()
#     z = np.array([a_n, b_n, c_n, d_n])
#     preds.append([[KF_a.x[0], KF_a.x[1]], [KF_b.x[0], KF_a.x[1]], [KF_a.x[0], KF_c.x[1]], [KF_b.x[0], KF_d.x[1]]])
#     truths.append([box.a, box.b, box.c, box.d])
#     measure.append(z)
#     cen_preds.append([(KF_a.x[0] + KF_b.x[0]) / 2, (KF_a.x[1] + KF_c.x[1]) / 2])
#     cen_truths.append([box.cen[0], box.cen[1]])
#     cen_measure.append([cen_n[0], cen_n[1]])
#
# for _ in range(step-to_update):
#     KF_a.predict()
#     KF_b.predict()
#     KF_c.predict()
#     KF_d.predict()
#     cen_n, a_n, b_n, c_n, d_n = box.change_size_n_move()
#     z = np.array([a_n, b_n, c_n, d_n])
#     preds.append([[KF_a.x[0], KF_a.x[1]], [KF_b.x[0], KF_a.x[1]], [KF_a.x[0], KF_c.x[1]], [KF_b.x[0], KF_d.x[1]]])
#     truths.append([box.a, box.b, box.c, box.d])
#     measure.append(z)
#     cen_preds.append([(KF_a.x[0] + KF_b.x[0]) / 2, (KF_a.x[1] + KF_c.x[1]) / 2])
#     cen_truths.append([box.cen[0], box.cen[1]])
#     cen_measure.append([cen_n[0], cen_n[1]])
#
# plt.subplot(2,1,1)
# plt.title('With Kalman Filter')
# for _ in range(step):
#     draw_some_box(preds[_][0], preds[_][1], preds[_][2], preds[_][3], col = 'palegreen')
# plt.plot([cen_preds[_][0] for _ in range(len(cen_preds))], [cen_preds[_][1] for _ in range(len(cen_preds))], color='green', marker='*')
# plt.plot([cen_truths[_][0] for _ in range(len(cen_preds))], [cen_truths[_][1] for _ in range(len(cen_preds))], color='red', marker='.',  mfc='none')
#
# plt.subplot(2,1,2)
# plt.title('Without Kalman Filter')
# for _ in range(step):
#     draw_some_box(measure[_][0], measure[_][1], measure[_][2], measure[_][3], col = 'skyblue')
# plt.plot([cen_measure[_][0] for _ in range(len(cen_preds))], [cen_measure[_][1] for _ in range(len(cen_preds))], color='blue', marker='+')
# plt.plot([cen_truths[_][0] for _ in range(len(cen_preds))], [cen_truths[_][1] for _ in range(len(cen_preds))], color='red', marker='.',  mfc='none')
# plt.show()

'''
Adaptive Tracker
'''
data = pd.read_csv('/Users/ash/Downloads/gaze_positions.csv')

x = np.array(data['norm_pos_x'])
y = np.array(data['norm_pos_y'])


skip = 10000
step = 1000
KF_2D = tracker_4dof()
preds = []
measure = []
z = np.array([x[skip], y[skip]])
count = [_ for _ in range(step)]
eps_max = 0.6
scale = 10
dumb = 0
old_Q = KF_2D.Q
for _ in range(skip, skip+step):

    KF_2D.predict()
    KF_2D.update(z)
    preds.append([KF_2D.x[0], KF_2D.x[1]])
    measure.append([z[0], z[1]])
    z = np.array([x[_], y[_]])
    # res_x.append(KF_2D.y[0])
    # res_y.append(KF_2D.y[1])
    res = KF_2D.y
    S =  KF_2D.S
    eps = np.dot(res.T, inv(S)).dot(res)
    if eps > eps_max:
        KF_2D.Q = old_Q*scale
        dumb += 1
    elif dumb > 0:
        KF_2D.Q = old_Q
        dumb = dumb - 1

plt.subplot(2,1,1)
# plt.xlim(0,1280)
# plt.ylim(0,720)
plt.title('With Kalman Filter')
plt.scatter([preds[_][0] for _ in range(len(preds))], [preds[_][1] for _ in range(len(preds))], color='green', marker='.')


plt.subplot(2,1,2)
# plt.xlim(0,1280)
# plt.ylim(0,720)
plt.title('Without Kalman Filter')
plt.scatter([measure[_][0] for _ in range(len(preds))], [measure[_][1] for _ in range(len(preds))], color='blue', marker='.')

plt.show()
