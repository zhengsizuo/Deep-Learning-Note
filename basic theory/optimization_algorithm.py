"""
The second homework in Modeling and Identification course.
Gradient Descent algorithm family, including momentum, nesterov
Author: zhs
Date: Nov 18, 2018
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def f(x):
    """原函数Z = x^2 + 50y^2"""
    return x[0]*x[0] + 50*x[1]*x[1]

def g(x):
    """梯度函数"""
    return np.array([2*x[0], 100*x[1]])

def momentum(x_start, learning_rate, g, discount=0.7):
    """动量梯度法"""
    x = np.array(x_start, dtype='float64')
    pre_grad = np.zeros_like(x)
    x_list = [x.copy()]
    for i in range(50):
        grad = g(x)
        pre_grad = discount*pre_grad + learning_rate*grad
        x -= pre_grad
        x_list.append(x.copy())
        print('[ Epoch {0} ] grad = {1}, x = {2}'.format(i, grad, x))
        # plt.plot(x[0], x[1])
        if abs(sum(grad)) < 1e-6:
            break

    return x_list


def nesterov(x_start, learning_rate, g, discount=0.7):
    """Nesterov梯度法"""
    x = np.array(x_start, dtype='float64')
    pre_grad = np.zeros_like(x)
    x_list = [x.copy()]
    for i in range(50):
        x_next = x - discount*pre_grad
        grad = g(x_next)
        pre_grad = discount*pre_grad + learning_rate*grad
        x -= pre_grad
        x_list.append(x.copy())

        print('[ Epoch {0} ] grad = {1}, x = {2}'.format(i, grad, x))
        if abs(sum(grad)) < 1e-6:
            break

    return x_list

def plot_3dfig(X, Y, Z):
    """绘制函数曲面图"""
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='rainbow')


def plot_contour(X, Y, Z, x_array):
    """绘制等高线图和下降动图"""
    fig = plt.figure(figsize=(15, 7))
    plt.contour(X, Y, Z, 10)
    plt.scatter(0, 0, marker='*', s=50, c='r', label='Optima')

    x_array = np.array(x_array)
    for j in range(len(x_array)-1):
        plt.plot(x_array[j:j+2, 0], x_array[j:j+2, 1])
        plt.scatter(x_array[j, 0], x_array[j, 1], c='k')
        plt.scatter(x_array[j+1, 0], x_array[j+1, 1], c='k')
        plt.pause(0.8)

        plt.legend(loc='best')


xi = np.linspace(-200, 200, 1000)
yi = np.linspace(-100, 100, 1000)
X, Y = np.meshgrid(xi, yi)
Z = X**2 + 50*Y**2

# plot_3dfig(X, Y, Z)

innitil_value = [150, 75]
# x_arr = momentum(innitil_value, 0.016, g, discount=0.9)
x_arr = nesterov(innitil_value, 0.014, g, discount=0.7)  # 0.014是学习率极限值，0.015时发散
plot_contour(X, Y, Z, x_arr)
plt.show()
