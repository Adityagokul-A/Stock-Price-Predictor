import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 

import yfinance as yf

device = 'cuda' if torch.cuda.is_available() else 'cpu'

stock_data = yf.download('GOOG', start='2010-01-01', end='2024-07-15') #apple stock data -> pandas dataframe

data = pd.DataFrame(stock_data['Close'])
#check for na's
#data.dropna()
#print(data.head())
N = 5
BATCH_SIZE = 32

"""
let us take previous N days closing data and try to predict the next day's closing data
"""

SS = StandardScaler()
MM =MinMaxScaler()

N_day_data = SS.fit_transform([data.iloc[i:i+N]['Close'] for i in range(0, len(data) - (len(data) % N)-N)])
N_day_target = MM.fit_transform(pd.DataFrame([data.iloc[i]['Close'] for i in range(N, len(data) - (len(data) % N))]))
X_train,X_test,y_train,y_test = train_test_split(N_day_data, N_day_target, test_size=0.08, shuffle=False) #don't shuffle time series data !!!!
y_test_plot = MM.inverse_transform(y_test).T[0]


X_train = torch.tensor(X_train,dtype=torch.float32) 
X_test = torch.tensor(X_test,dtype=torch.float32) 


y_train = torch.flatten(torch.tensor(y_train,dtype=torch.float32))
y_test = torch.flatten(torch.tensor(y_test,dtype=torch.float32))


#print(X_train.shape, y_train.shape)
train_dataset= TensorDataset(X_train, y_train)
train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE)

test_dataset= TensorDataset(X_test, y_test)
test_dataloader = DataLoader(test_dataset, batch_size=1)
print(len(test_dataloader))

class LSTM(nn.Module):
    def __init__(self):
        super(LSTM,self).__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=1,batch_first=True)

    def forward(self, input):
        input = torch.reshape(input, (input.shape[0],input.shape[1],1))
        out,_ = self.lstm(input)

        prediction = out[-1]
        
        return prediction

model = LSTM()
model.to(device)
print(model)


optimizer = Adam(model.parameters(), lr = 0.01)
loss_fn = nn.MSELoss()

EPOCHS = 1000

for epoch in range(EPOCHS): 
    running_loss = 0.0
    for i, data in enumerate(train_dataloader, 0):
        inputs, labels = data
        inputs = inputs.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()
        torch.cuda.empty_cache()
        running_loss += loss.item()
        if i % 50 == 49:
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 50))
            running_loss = 0.0

MODEL_PATH = ',model.pth'
torch.save(model.state_dict(), MODEL_PATH)

print("Training done")

y_pred = np.array([])
with torch.no_grad():
    model.eval()
    for data in test_dataloader:
        inputs, labels = data
        inputs = inputs.to(device)
        labels = labels.to(device)
        outputs = model(inputs)
        #print(MM.inverse_transform(outputs.cpu().numpy()).T[0])
        y_pred = np.append(y_pred,MM.inverse_transform(outputs.cpu().numpy()).T[0][-1])

print(len(y_pred), len(y_test_plot))
plt.plot(y_test_plot,label = "orig")
plt.plot(y_pred, label = "pred")

plt.show()
