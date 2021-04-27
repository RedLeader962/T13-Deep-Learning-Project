import torch
import torch.nn as nn

x = nn.LSTMCell(1,2)
print(x.weight_ih.shape)

p = nn.LSTM(1,2,1)
torch.save(p.state_dict(),'hello.pth')


#p.all_weights = x.weight_ih
print(p.hidden_size)
print(p.input_size)

def assign_LSTM_param_to_LSTMCell(lstm, lstmcell):
    param_lstm = lstm.state_dict()
    param_lstmcell = lstmcell.state_dict()

    state_dict = {}
    for w1, w2 in zip(param_lstm, param_lstmcell):
        shape_w1 = param_lstm[w1].shape
        shape_w2 = param_lstmcell[w2].shape

        assert shape_w1 == shape_w2, f'Lstm a une dimension de {shape_w1} alors que LSTMCell a une dimension de {shape_w2}'

        state_dict[w2] = param_lstm[w1]

    lstmcell.load_state_dict(state_dict)


assign_LSTM_param_to_LSTMCell(p,x)


for i, k in zip(p.named_parameters(), x.named_parameters()):
    print(i, k)

x = torch.load('hello.pth')


#x.load_state_dict(torch.load('hello.pth'))
