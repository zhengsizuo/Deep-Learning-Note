"""
Stock prediction using LSTM, modified from https://github.com/NGYB/Stocks/blob/master/StockPricePrediction/StockPricePrediction_v4a_lstm.ipynb
Author: zhs
Date: 2020.3.12
"""
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from RNNModel import RNN, device

# parameters set
data_path = 'data/VTI.csv'
N = 3  # 过去N天的数据作为特征
# LSTM相关参数
hidden_size = 50
num_layers = 2
output_size = 1
batch_size = 4
num_epochs = 2
learning_rate = 0.01
#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def get_x_y(data, N, offset):
    """
    Split data into x (features) and y (target)
    offset: beginning inddex
    """
    x, y = [], []
    for i in range(offset, len(data)):
        x.append(data[i - N:i])
        y.append(data[i])
    x = np.array(x)
    y = np.array(y)

    return x, y


def get_x_scaled_y(data, N, offset):
    """
    Split data into x (features) and y (target)
    We scale x to have mean 0 and std dev 1, and return this.
    We do not scale y here.
    Inputs
        data     : pandas series to extract x and y
        N
        offset
    Outputs
        x_scaled : features used to predict y. Scaled such that each element has mean 0 and std dev 1
        y        : target values. Not scaled
        mu_list  : list of the means. Same length as x_scaled and y
        std_list : list of the std devs. Same length as x_scaled and y
    """
    x_scaled, y, mu_list, std_list = [], [], [], []
    for i in range(offset, len(data)):
        # 对每个x做一次均值和标准差统计，即对每N条数据，便于后续恢复预测数据
        mu_list.append(np.mean(data[i - N:i]))
        std_list.append(np.std(data[i - N:i]))
        x_scaled.append((data[i - N:i] - mu_list[i - offset]) / std_list[i - offset])
        y.append(data[i])
    x_scaled = np.array(x_scaled)
    y = np.array(y)

    return x_scaled, y, mu_list, std_list


def trans_to_tensors(train_x, train_y, batch_size):
    """
    transfer original data into torch tensor according to batch_size
    :param train_x: ndarray
    :param train_y: ndarray
    :param batch_size: int
    :return: torch.tensor
    """
    num_batch = train_x.shape[0] / batch_size
    train_x = np.transpose(train_x, (0, 2, 1))  # 交换第2和第3维度
    tensor_x, tensor_y = [], []
    for i in range(batch_size, len(train_x), batch_size):
        tensor_x.append(train_x[i-batch_size:i])
        tensor_y.append(train_y[i-batch_size:i])

    tensor_x, tensor_y = np.array(tensor_x), np.array(tensor_y)
    return torch.from_numpy(tensor_x).to(dtype=torch.float32), torch.from_numpy(tensor_y).to(dtype=torch.float32)


def cal_MAPE(y_true, y_pred):
    """
    Compute mean absolute percentage error (MAPE)
    """
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


### parse data csv file to get some basic information ###
df = pd.read_csv(data_path, sep=",")

# Convert Date column to datetime
df.loc[:, 'Date'] = pd.to_datetime(df['Date'], format='%Y-%m-%d')

# Change all column headings to be lower case, and remove spacing
df.columns = [str(x).lower().replace(' ', '_') for x in df.columns]

# Get month of each sample
df['month'] = df['date'].dt.month

# Sort by datetime
df.sort_values(by='date', inplace=True, ascending=True)

print(df.head())
# print("The total number of the stock data:\n{}".format(df.count()))  # 无缺失值

# Get sizes of each of the datasets
num_cv = int(0.2*len(df))
num_test = int(0.2*len(df))
num_train = len(df) - num_cv - num_test
print("num_train = " + str(num_train))
print("num_cv = " + str(num_cv))
print("num_test = " + str(num_test))

# Split into train, validation, and test set, 取date和adj_close列
train_set = df[:num_train][['date', 'adj_close']]
cv_set = df[num_train:num_train+num_cv][['date', 'adj_close']]
train_cv_set = df[:num_train+num_cv][['date', 'adj_close']]  # train set + validation set
test_set = df[num_train+num_cv:][['date', 'adj_close']]

scaler = StandardScaler()
# use train_set before tuning, use train_cv_set after tuning
train_scaled = scaler.fit_transform(train_cv_set['adj_close'].values.reshape(-1, 1))
print("scaler.mean_ = " + str(scaler.mean_))
print("scaler.var_ = " + str(scaler.var_))

# Split into scaled x and scaled y
x_train_scaled, y_train_scaled = get_x_y(train_scaled, N, N)
print("x_train_scaled.shape = " + str(x_train_scaled.shape))  # (444, 9, 1)
print("y_train_scaled.shape = " + str(y_train_scaled.shape))  # (444, 1)

x_cv_scaled, y_cv, mu_cv_list, std_cv_list = get_x_scaled_y(train_cv_set['adj_close'].values.reshape(-1, 1), N, num_train)
x_cv_scaled = np.transpose(x_cv_scaled, (0, 2, 1))
print("x_cv_scaled.shape = " + str(x_cv_scaled.shape))
print("y_cv.shape = " + str(y_cv.shape))

x_test_scaled, y_test, mu_test_list, std_test_list = get_x_scaled_y(test_set['adj_close'].values.reshape(-1, 1), N, N)
x_test_scaled = np.transpose(x_test_scaled, (0, 2, 1))


### build the LSTM model and train ###
model = RNN(N, hidden_size, num_layers, output_size).to(device)
tensor_x, tensor_y = trans_to_tensors(x_train_scaled, y_train_scaled, batch_size)
# print(tensor_x.shape, tensor_y.shape)
# Loss and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

total_step = len(tensor_x)
for ep in range(num_epochs):
    for i in range(total_step):
        # Forward pass
        y_pre = model(tensor_x[i].to(device))
        loss = criterion(y_pre, tensor_y[i].to(device))

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i + 1) % 10 == 0:
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                  .format(ep + 1, num_epochs, i + 1, total_step, loss.item()))

# Test the model on the validation set
model.eval()
with torch.no_grad():
    # after tuning, change x_cv to x_test
    x_test_scaled = torch.from_numpy(x_test_scaled).to(dtype=torch.float32)
    y_test_pre = model(x_test_scaled)

est = (y_test_pre.numpy() * np.array(std_test_list).reshape(-1, 1)) + np.array(mu_test_list).reshape(-1,1)
print("est.shape = " + str(est.shape))

# Calculate RMSE
rmse_bef_tuning = np.sqrt(mean_squared_error(y_test, est))
print("RMSE = %0.3f" % rmse_bef_tuning)

# Calculate MAPE
mape_pct_bef_tuning = cal_MAPE(y_test, est)
print("MAPE = %0.3f%%" % mape_pct_bef_tuning)

est_df = pd.DataFrame({'est': est.reshape(-1),
                       'date': test_set['date'][:-3]})

ax = train_set.plot(x='date', y='adj_close', style='b-', grid=True)
ax = cv_set.plot(x='date', y='adj_close', style='y-', grid=True, ax=ax)
ax = test_set.plot(x='date', y='adj_close', style='g-', grid=True, ax=ax)
ax = est_df.plot(x='date', y='est', style='r-', grid=True, ax=ax)
ax.legend(['train', 'dev', 'test', 'est'])
ax.set_xlabel("date")
ax.set_ylabel("USD")
