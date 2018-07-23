from filterpy.kalman import KalmanFilter
import random
import numpy as np
import matplotlib.pyplot as plt

class Center_2D(object):
    def __init__(self, sigma, p=(0,0), v=(0,0)):
        self.vel = np.array(v)
        self.pos = np.array(p)
        self.sigma = sigma


    def read(self):
        self.pos[0] += self.vel[0]
        self.pos[1] += self.vel[1]
        return [self.pos[0] + random.gauss(0, self.sigma),
                self.pos[1] + random.gauss(0, self.sigma)]


class BoundBox(object):
    def __init__(self, h=5, w=5, cen=(0,0), v=(0,0), sigma=1.0, col = 'red', linestyle='--'):
        self.cen = np.array(cen)
        self.a = np.array((self.cen[0] - w/2, self.cen[1] + h/2))
        self.b = np.array((self.cen[0] + w/2, self.cen[1] + h/2))
        self.c = np.array((self.cen[0] - w/2, self.cen[1] - h/2))
        self.d = np.array((self.cen[0] + w/2, self.cen[1] - h/2))
        self.v = np.array(v)
        self.sigma = sigma
        self.col = col
        self.linestyle = linestyle

    def draw_box(self):
        plt.plot([self.a[0], self.b[0] ], [self.a[1], self.b[1]], 'k-', color=self.col, linestyle=self.linestyle)
        plt.plot([self.a[0], self.c[0] ], [self.a[1], self.c[1]], 'k-', color=self.col, linestyle=self.linestyle)
        plt.plot([self.d[0], self.b[0] ], [self.d[1], self.b[1]], 'k-', color=self.col, linestyle=self.linestyle)
        plt.plot([self.c[0], self.d[0] ], [self.c[1], self.d[1]], 'k-', color=self.col, linestyle=self.linestyle)

    def read(self):
        noise = random.gauss(0, self.sigma)
        self.cen = self.cen + self.v
        self.a = self.a + self.v
        self.b = self.b + self.v
        self.c = self.c + self.v
        self.d = self.d + self.v
        cen_n, a_n, b_n, c_n, d_n = self.cen + noise, self.a + noise, self.b + noise, self.c + noise, self.d + noise
        return cen_n, a_n, b_n, c_n, d_n

    def change_size(self):
        noise_1 = random.gauss(0, self.sigma)
        noise_2 = random.gauss(0, self.sigma)
        noise_3 = random.gauss(0, self.sigma)
        noise_4 = random.gauss(0, self.sigma)
        size_a = np.array([self.a[0] + noise_1, self.a[1] + noise_2])
        size_b = np.array([self.b[0] + noise_4, self.b[1] + noise_2])
        size_c = np.array([self.c[0] + noise_1, self.c[1] + noise_3])
        size_d = np.array([self.d[0] + noise_4, self.d[1] + noise_3])
        return size_a, size_b, size_c, size_d

    def change_size_n_move(self):
        self.cen = self.cen + self.v
        self.a = self.a + self.v
        self.b = self.b + self.v
        self.c = self.c + self.v
        self.d = self.d + self.v
        noise = random.gauss(0, self.sigma)
        noise_1 = random.gauss(0, self.sigma)
        noise_2 = random.gauss(0, self.sigma)
        noise_3 = random.gauss(0, self.sigma)
        noise_4 = random.gauss(0, self.sigma)
        size_a = np.array([self.a[0] + noise_1, self.a[1] + noise_2])
        size_b = np.array([self.b[0] + noise_4, self.b[1] + noise_2])
        size_c = np.array([self.c[0] + noise_1, self.c[1] + noise_3])
        size_d = np.array([self.d[0] + noise_4, self.d[1] + noise_3])
        cen_n, a_n, b_n, c_n, d_n = self.cen + noise, size_a + noise, size_b + noise, size_c + noise, size_d + noise
        return cen_n, a_n, b_n, c_n, d_n


def box_visualize(box, step=25):

    cen_x, cen_y = [], []
    for _ in range(step):
        box.draw_box()
        cen_x.append(box.cen[0])
        cen_y.append(box.cen[1])
        box.read()
    plt.plot(cen_x, cen_y, marker='.', color='green', mfc='none')


def draw_some_box(a, b, c, d, col = 'red', linestyle='--'):
    plt.plot([a[0], b[0]], [a[1], b[1]], 'k-', color=col, linestyle=linestyle)
    plt.plot([a[0], c[0]], [a[1], c[1]], 'k-', color=col, linestyle=linestyle)
    plt.plot([d[0], b[0]], [d[1], b[1]], 'k-', color=col, linestyle=linestyle)
    plt.plot([c[0], d[0]], [c[1], d[1]], 'k-', color=col, linestyle=linestyle)

# frame = Frame()
# frame.draw()
#
# box = NonLinearBoundBox(h=100, w= 200, a=(0,-2), v=(15,8), sigma=20)
#
# box_visualize(box, step=100)
# plt.show()