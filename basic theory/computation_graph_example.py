# 李宏毅老师计算图PPT中的例子
import torch
import torch.nn as nn
import numpy.linalg as lg

torch.manual_seed(1)
# a = torch.tensor(3., requires_grad=True)
# b = torch.tensor(2., requires_grad=True)
#
# c = a + b
# d = b + 1
# e = c * d
#
# c.retain_grad()
# d.retain_grad()
# e.backward()
#
# print(c.grad, d.grad)
# print(a.grad, b.grad.data)

def pinv(tensor):
    # 求tensor的伪逆
    tensor_inv = torch.tensor(lg.pinv(tensor.numpy()))
    return tensor_inv

# 手动搭建双层神经网络
x = torch.randn(1, 3)
y = torch.tensor([[1., 0.]])
# y = torch.LongTensor([1])

# 随机初始化权重
w1 = torch.randn(3, 5, requires_grad=True)
w2 = torch.randn(5, 2, requires_grad=True)

# criterion = nn.MSELoss()
criterion = nn.CrossEntropyLoss()

# 隐含层数目5
z1 = torch.matmul(x, w1)
a1 = torch.sigmoid(z1)
z1.retain_grad(), a1.retain_grad()

z2 = torch.matmul(a1, w2)
y_pre = torch.sigmoid(z2)
y_pre.retain_grad(), z2.retain_grad()

# loss = criterion(y_pre, y)
log_y_pre = torch.log(y_pre)
loss = -torch.dot(y[0], log_y_pre[0])
loss.backward()

# print(z2.grad, y_hat.grad)
y_grad = y_pre.grad
z2_grad = z2.grad

dy_dz2 = torch.matmul(pinv(y_grad), z2_grad).numpy()
print("dy/dz2:\n", dy_dz2)

dz2_da1 = torch.matmul(pinv(z2.grad), a1.grad).numpy()
w2_grad = w2.grad.view(1, 10)
dz2_dw2 = torch.matmul(pinv(z2.grad), w2_grad).numpy()
print("dz2/da1:\n", dz2_da1)
print("dz2/dw2:\n", dz2_dw2)

da1_dz1 = torch.matmul(pinv(a1.grad), z1.grad).numpy()
print("da1/dz1:\n", da1_dz1)
dz1_dw1 = torch.matmul(z1.grad, w1.grad.T).numpy()
print("dz1/dw1:", dz1_dw1)
print("x:", x.numpy())
