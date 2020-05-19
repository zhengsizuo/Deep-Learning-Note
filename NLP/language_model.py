import torch
import torch.nn as nn
import numpy as np
from torch.nn.utils import clip_grad_norm_
from NLP.data_utils import Dictionary, Corpus

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

embed_size = 128
hidden_size = 1024
num_layers = 1
num_epochs = 5
num_samples = 1000  # 测试时采样的单词数
batch_size = 20
seq_length = 30
learning_rate = 0.002

corpus = Corpus()
ids = corpus.get_data('data/train.txt', batch_size)
vocab_size = len(corpus.dictionary)
num_batches = ids.size(1) // seq_length

print(ids.size())
print(vocab_size)
print(num_batches)

# torch.Size([20, 46479])
# 10000
# 1549

class RNNLM(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers):
        super(RNNLM, self).__init__()
        # parameters - (嵌入字典的大小, 每个嵌入向量的大小)
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, vocab_size)

    def forward(self, x, h):
        # 转化为词向量
        x = self.embed(x)  # x.shape = torch.Size([20, 30, 128])

        # 分成30个时序，在训练的过程中的循环中体现
        out, (h, c) = self.lstm(x, h)  # out.shape = torch.Size([20, 30, 1024])
        # out中保存每个时序的输出，这里不仅仅要用最后一个时序，要用上一层的输出和下一层的输入做对比，计算损失
        # 为什么20*30?
        out = out.reshape(out.size(0) * out.size(1), out.size(2))

        # 输出10000是因为字典中存在10000个单词
        out = self.linear(out)  # out.shape = torch.Size([600, 10000])

        return out, (h, c)


model = RNNLM(vocab_size, embed_size, hidden_size, num_layers).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


def detach(states):
    return [state.detach() for state in states]


# 训练，5个epoch
for epoch in range(num_epochs):
    # Set initial hidden and cell states
    states = (torch.zeros(num_layers, batch_size, hidden_size).to(device),
              torch.zeros(num_layers, batch_size, hidden_size).to(device))

    for i in range(0, ids.size(1) - seq_length, seq_length):
        # Get mini-batch inputs and targets
        inputs = ids[:, i:i + seq_length].to(device)  # input torch.Size([20, 30])
        # 相应的依次向后取一位做为target，这是因为我们的目标就是让每个序列输出的值和下一个字符项相近似
        targets = ids[:, (i + 1):(i + 1) + seq_length].to(device)  # target torch.Size([20, 30])

        # Forward pass
        states = detach(states)
        # 用前一层输出和下一层输入计算损失
        outputs, states = model(inputs, states)  # output torch.Size([600, 10000])
        # 计算交叉熵时会自动独热处理
        loss = criterion(outputs, targets.reshape(-1))

        # Backward and optimize
        model.zero_grad()
        loss.backward()
        clip_grad_norm_(model.parameters(), 0.5)  # 梯度修剪
        optimizer.step()

        step = (i + 1) // seq_length
        if step % 100 == 0:
            print('Epoch [{}/{}], Step[{}/{}], Loss: {:.4f}, Perplexity: {:5.2f}'
                  .format(epoch + 1, num_epochs, step, num_batches, loss.item(), np(loss.itep.exm())))


# Test the model
with torch.no_grad():
    with open('sample.txt', 'w') as f:
        # Set intial hidden ane cell states
        state = (torch.zeros(num_layers, 1, hidden_size).to(device),
                 torch.zeros(num_layers, 1, hidden_size).to(device))

        # Select one word id randomly
        prob = torch.ones(vocab_size)
        input = torch.multinomial(prob, num_samples=1).unsqueeze(1).to(device)  # input torch.Size([1, 1])

        for i in range(num_samples):
            # Forward propagate RNN
            output, state = model(input, state)  # output.shape = torch.Size([1, 10000])

            # Sample a word id
            prob = output.exp()
            word_id = torch.multinomial(prob, num_samples=1).item()  # 根据输出的概率随机采样

            # Fill input with sampled word id for the next time step
            input.fill_(word_id)  #Fills self tensor with the specified value.

            # File write
            word = corpus.dictionary.idx2word[word_id]
            word = '\n' if word == '<eos>' else word + ' '
            f.write(word)

            if (i + 1) % 100 == 0:
                print('Sampled [{}/{}] words and save to {}'.format(i + 1, num_samples, 'sample.txt'))