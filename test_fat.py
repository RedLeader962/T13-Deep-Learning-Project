import torch
import torch.nn as nn

x = nn.LSTMCell(1,2)
print(x.weight_ih.shape)


torch.save(x,'hello.pth')

print()
p = nn.LSTM(1,2,1)


#p.all_weights = x.weight_ih
print(p.hidden_size)
print(p.input_size)

def assign_LSTM_param_to_LSTMCell(lstm, lstmcell):
    param_lstm = lstm.named_parameters()
    param_lstmcell = lstmcell.named_parameters()

    for (name1, weight1), (name2,weight2) in zip(param_lstm, param_lstmcell):
        assert weight1.shape == weight2.shape, f'Lstm a une dimension de {weight1.shape} alors que LSTMCell a une dimension de {weight2.shape}'
        weight1 = weight2

assign_LSTM_param_to_LSTMCell(p,x)
