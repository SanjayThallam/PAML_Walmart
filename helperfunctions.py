import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn
import matplotlib.pyplot as plt
import warnings
from datetime import datetime
import random 
import os
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
import time 
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import mean_squared_error
import multiprocessing
import torch.nn.functional as F
import numpy as np 
import pandas as pd 
import streamlit as st


def initialize():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    #st.markdown(device)
    SEED = 1
    os.environ["PL_GLOBAL_SEED"] = str(SEED)
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    #read_data()

# Set the directory path to where your CSV files are located
INPUT_DIR_PATH = '' # or the appropriate path

def read_data():
    calendar_df = pd.read_csv(INPUT_DIR_PATH + 'calendar.csv')
    #st.markdown('Calendar has {} rows and {} columns'.format(calendar_df.shape[0], calendar_df.shape[1]))

    sales_train_evaluation_df = pd.read_csv(INPUT_DIR_PATH + 'sales_train_evaluation.csv')
   #st.markdown('Sales train evaluation has {} rows and {} columns'.format(sales_train_evaluation_df.shape[0], sales_train_evaluation_df.shape[1]))
    st.markdown('Data Read In')
    return calendar_df, sales_train_evaluation_df

# Read the data
calendar_df, sales_train_evaluation_df = read_data()

def create_string(store_choice, category_choice, number):
    #store_chocei = string
    #categorty chocie = string
    #number = int 
    #string builder

    number_str = str(number).zfill(3)
    myString = category_choice + "_" + number_str + "_" + store_choice + "_evaluation"
    return myString

num_epochs = 100
hidden_size = 64
num_layers = 1

def update_params(num_epochs1, num_hidden1, num_layers1):
    global num_epochs, hidden_size, num_layers
    num_epochs = num_epochs1
    hidden_size = num_hidden1
    num_layers = num_layers1

class M5_Multivariate_Dataset(Dataset):
    def __init__(self, df, seq_length):
        self.df = df
        self.seq_length = seq_length
        self.total_length = len(df[:,0]) - seq_length 
    
    def __len__(self):
        return self.total_length
    
    def __getitem__(self, index):
        if index < self.total_length:
            sequence = self.df[index: index + self.seq_length,:]
            target = self.df[index + self.seq_length,0]
            return torch.tensor(sequence, dtype=torch.float32), \
                    torch.tensor(target, dtype=torch.float32)


class RNN(nn.Module):

    def __init__(self, input_size, hidden_size, num_layers):
        super().__init__()
        
        self.rnn = nn.RNN(input_size=input_size, hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=True)
        
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        # out: (batch_size, seq_length, hidden_size)
        out, _ = self.rnn(x)
        # take the out at the last time step
        # out: (batch_size, hidden_size)
        out = out[:, -1]
        # pred: (batch_size, 1)
        pred = self.fc(out)
        # output: batch_size
        return pred.squeeze()

class GRU(nn.Module):

    def __init__(self, input_size, hidden_size, num_layers):
        super().__init__()
        
        self.gru = nn.GRU(input_size=input_size, hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=True)
        
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        # out: (batch_size, seq_length, hidden_size)
        out, _ = self.gru(x)
        # take the out at the last time step
        # out: (batch_size, hidden_size)
        out = out[:, -1]
        # pred: (batch_size, 1)
        pred = self.fc(out)
        # output: batch_size
        return pred.squeeze()

class CNN(nn.Module):

    def __init__(self, input_size, hidden_size, num_layers):
        super().__init__()
        
        self.cnn = nn.Conv1d(in_channels=28, out_channels = 64, kernel_size = 3)
        
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        # out: (batch_size, seq_length, hidden_size)
        out, _ = self.cnn(x)
        # take the out at the last time step
        # out: (batch_size, hidden_size)
        out = out[:, -1]
        # pred: (batch_size, 1)
        pred = self.fc(out)
        # output: batch_size
        return pred.squeeze()



class LSTM(nn.Module):

    def __init__(self, input_size, hidden_size, num_layers):
        super().__init__()
        
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=True)
        
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        # out: (batch_size, seq_length, hidden_size)
        out, _ = self.lstm(x)
        # take the out at the last time step
        # out: (batch_size, hidden_size)
        out = out[:, -1]
        # pred: (batch_size, 1)
        pred = self.fc(out)
        # output: batch_size
        return pred.squeeze()

#create the train functions for lstm and rnn

def runtwo(item_to_model,input_value, num_epochs, hidden_size, num_layers, model_choice):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    def train(model, num_epochs, tr_dl, va_dl):
        loss_hist_train = [0] * num_epochs
        loss_hist_valid = [0] * num_epochs
        
        model.to(device)
        start_time = time.time()
        for epoch in range(num_epochs): 
            model.train()
            avg_loss = 0
            for i, (history, target) in enumerate(tr_dl):
                # forward
                outputs = model(history.to(device))
                loss = loss_fn(outputs, target.to(device))
                # backward
                optimizer.zero_grad()
                loss.backward()
                # update
                optimizer.step()
                # log
                avg_loss += loss.item() / len(tr_dl)

            model.eval()
            avg_val_loss = 0
            with torch.no_grad():
                for history, target in va_dl:
                    outputs = model(history.to(device))
                    loss = loss_fn(outputs, target.to(device))
                    avg_val_loss += loss.item() / len(va_dl)

            elapsed_time = time.time() - start_time
            
            loss_hist_train[epoch] = avg_loss
            loss_hist_valid[epoch] = avg_val_loss
            
            if epoch % 10 == 0:
                st.write(
                        f"Epoch {epoch:02d}/{num_epochs:02d} \t t={elapsed_time:.0f}s \t"
                        f"loss={avg_loss:.3f} \t",
                        f"val_loss={avg_val_loss:.3f}"
                    )
        return loss_hist_train, loss_hist_valid
    calendar_df, sales_train_evaluation_df = read_data()
    # Filter the sales data for the item to model
    sales_train_evaluation_df = sales_train_evaluation_df[sales_train_evaluation_df.id == item_to_model]

    # Find all columns that have a day notation 'd_'
    d_cols = [col for col in sales_train_evaluation_df.columns if col.startswith('d_')]

    # add the day of week of yesterday as a new series
    ste_t = sales_train_evaluation_df[['id']+ d_cols].set_index('id').T
    ste_td = ste_t.merge(calendar_df[['date','d']].set_index('d'),
            left_index = True, right_index=True, validate='1:1').set_index('date')
    ste_td = ste_td.reset_index()
    ste_td = ste_td.assign(date = pd.to_datetime(ste_td.date))
    ste_td = ste_td.assign(tmr = ste_td.date.shift(-1))
    ste_td.loc[ste_td.shape[0] - 1,'tmr'] = ste_td.loc[ste_td.shape[0] - 1,'date']  + pd.DateOffset(1)
    ste_td = ste_td.assign(tmr_dow = ste_td.tmr.dt.day_name().astype('category'))
    ste_td.tail(10)

    unique_levels = len(set(ste_td['tmr_dow']))
    print('tmr_dow', 'unique levels =', unique_levels)
    print(ste_td['tmr_dow'].value_counts(dropna = False))
    dow_encoded = F.one_hot((torch.from_numpy(ste_td['tmr_dow'].cat.codes.values) % unique_levels).long(), num_classes = unique_levels)
    y = torch.from_numpy(ste_td[item_to_model].values).float()
    data = torch.cat([y.reshape(-1,1), dow_encoded], axis = 1)

    # normalize to range from -1 to 1
    scaler = MinMaxScaler(feature_range=(-1, 1))
    ste_normalized = scaler.fit_transform(data)

    
    tr_arr = ste_normalized[:1717,:]
    va_arr = ste_normalized[1717:1913,:]
    te_arr = ste_normalized[1913:,:]

    print(tr_arr.shape, va_arr.shape, te_arr.shape)


    BATCH_SIZE = 32
    SEQ_LENGTH = 28
    tr_ds = M5_Multivariate_Dataset(tr_arr, SEQ_LENGTH)
    va_ds = M5_Multivariate_Dataset(va_arr, SEQ_LENGTH)
    tr_dl = DataLoader(tr_ds, batch_size=32, shuffle=True, num_workers=0)
    va_dl = DataLoader(va_ds, batch_size=32, shuffle=False, num_workers=0)
    num_workers = multiprocessing.cpu_count()
    print(len(tr_dl), len(va_dl))

    print('Training Set:\n')
    for history, target in tr_dl:  
        print('batch dimensions:', history.size())
        print('label dimensions:', target.size())
        print(target[:3])
        break
        
    # Checking the dataset
    print('\nValidation Set:')
    for history, target in va_dl:  
        print('batch dimensions:', history.size())
        print('target dimensions:', target.size())
        print(target[:3])
        break
    learning_rate = 3e-4
    input_size = tr_arr.shape[1]
    ##### Init the Model #######################
    model = None
    if model_choice == "RNN":
        model = RNN(input_size, hidden_size, num_layers)
    elif model_choice == "LSTM":
        model = LSTM(input_size, hidden_size, num_layers)
    elif model_choice == "GRU":
        model = GRU(input_size, hidden_size, num_layers)
    elif model_choice == "CNN":
        model = CNN(input_size, hidden_size, num_layers)
    model.to(device)

    ##### Set Criterion Optimzer and scheduler ####################
    loss_fn = torch.nn.MSELoss().to(device)    # mean-squared error for regression
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-6)

    history = train(model, num_epochs, tr_dl, va_dl)

    st.write(model)
    l1 = history[0]
    l2 = history[1]
    df = pd.DataFrame(zip(l1,l2), columns =['Train_mse', 'Validation_mse'])
    df.to_csv('history_multi.csv')

    my_dpi = 100
    fig1, ax = plt.subplots(nrows = 1, ncols = 1, figsize=(540/my_dpi, 360/my_dpi), dpi=my_dpi)
    ax.plot(history[0], lw=2)
    ax.plot(history[1], lw=2)
    ax.legend(['Train loss', 'Validation loss'], fontsize=12)
    ax.set_xlabel('Epochs', size=12)
    plt.tight_layout()
    st.pyplot(fig = fig1)

    PREDICTION_LENGTH = 28

    va_last_seq = va_arr[-SEQ_LENGTH:]
    x = torch.from_numpy(va_last_seq).float()
    x = x.to(device).unsqueeze(0)
    y_preds = []
    for step_ahead in range(PREDICTION_LENGTH):
        y_pred_one = model(x)
        y_preds.append(y_pred_one.item())
        y_pred_multi = torch.from_numpy(te_arr[step_ahead,1:]).float()
        
        y_pred_multi = torch.cat([y_pred_multi, y_pred_one.reshape(1).cpu().detach()], axis = 0)
        x = torch.cat([x[:,1:,:], y_pred_multi.unsqueeze(0).unsqueeze(0).to(device)], axis = 1)


    x = torch.from_numpy(va_last_seq).float()
    x = x.to(device).unsqueeze(0)
    x.shape

    my_dpi = 100
    fig2, ax = plt.subplots(nrows = 1, ncols = 1, figsize=(540/my_dpi, 360/my_dpi), dpi=my_dpi)
    ax.plot(y_preds, lw=2)
    ax.plot(te_arr[:,0], lw=2)
    ax.legend(['Forecast', 'Actual'], fontsize=12)
    ax.set_xlabel('Epochs', size=12)
    plt.tight_layout()
    st.pyplot(fig = fig2)


    st.write('MSE on the test period =', mean_squared_error(te_arr[:,0], y_preds))
    st.write('RMSE on the test period =', np.sqrt(mean_squared_error(te_arr[:,0], y_preds)))
