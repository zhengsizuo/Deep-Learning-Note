"""
Modified from MorvanZhou's code:https://github.com/MorvanZhou/PyTorch-Tutorial/blob/master/tutorial-contents/203_activation.py
Author: zhs
Date: 2020.3.11
Dependencies:
torch: 0.4
matplotlib
"""
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt

# fake data
x = torch.linspace(-5, 5, 200)  # x data (tensor), shape=(100, 1)
x = Variable(x, requires_grad=True)
x_np = x.data.numpy()   # numpy array for plotting

# following are popular activation functions
y_relu = torch.relu(x).data
y_sigmoid = torch.sigmoid(x)
y_tanh = torch.tanh(x)
y_softplus = F.softplus(x) # there's no softplus in torch
# y_softmax = torch.softmax(x, dim=0).data.numpy() softmax is a special kind of activation function, it is about probability

# plt to visualize these activation function
plt.figure(1, figsize=(8, 6))
plt.subplot(221)
plt.plot(x_np, y_relu.numpy(), c='red', label='relu')
plt.ylim((-1, 5))
plt.legend(loc='best')

plt.subplot(222)
v = torch.ones_like(x)
y_sigmoid.backward(v)  # 标量对标量求导，需要用vector-Jacobian乘积

max_value = np.zeros_like(x_np)
for i in range(len(max_value)):
    max_value[i] = 0.25

plt.plot(x_np, y_sigmoid.data.numpy(), c='red', label='sigmoid')
plt.plot(x_np, x.grad.detach().numpy(), label='sigmoid_diff')
plt.plot(x_np, max_value, c='y', linestyle='--', label='diff_max(0.25)')
plt.ylim((-0.2, 1.2))
plt.legend(loc='best')


plt.subplot(223)
y_tanh.backward(v)
plt.plot(x_np, y_tanh.data.numpy(), c='red', label='tanh')
plt.plot(x_np, x.grad.detach().numpy(), label='tanh_diff')
plt.ylim((-1.2, 1.3))
plt.legend(loc='best')

plt.subplot(224)
y_softplus.backward(v)
plt.plot(x_np, y_softplus.data.numpy(), c='red', label='softplus')
plt.plot(x_np, x.grad.detach().numpy(), label='softplus_diff')
plt.ylim((-0.2, 6))
plt.legend(loc='best')


plt.show()