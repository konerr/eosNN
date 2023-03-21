from torch.autograd import Variable
import torch.optim as optim
import numpy as np
import torch as T
import torch.nn as nn
import torch.nn.functional as F

if T.cuda.is_available():
    device = T.device("cuda")
    print('Running on ' + T.cuda.get_device_name(0))
else:
    device = T.device("cpu")

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(3, 100)
        self.fc2 = nn.Linear(100, 100)
        self.fc3 = nn.Linear(100, 2)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        
        return x 

class Dataset(T.utils.data.Dataset):
# Inputs: rho_m, e_m, y_exp
# Outputs: p, T

    def __init__(self, src_file):
        data = np.loadtxt(src_file)
        dataMin = data.min(0,keepdims=True)
        dataMax = data.max(0,keepdims=True)
        data_norm = norm_01(data, dataMin, dataMax)

        x_tmp = data_norm[:,0:3] #inputs
        y_tmp = data_norm[:,3:5] #outputs

        self.x_data = T.tensor(x_tmp).to(device)
        self.y_data = T.tensor(y_tmp).to(device)
        self.dataMin = dataMin
        self.dataMax = dataMax

    def __len__(self):
        return len(self.x_data)

    def __getitem__(self, idx):
        if T.is_tensor(idx):
          idx = idx.tolist()
        inp = self.x_data[idx, :]
        out = self.y_data[idx, :]
        dataMin = self.dataMin
        dataMax = self.dataMax
        sample = \
          { 'inp' : inp, 'out' : out, 'dmin' : dataMin, 'dmax': dataMax }
        return sample

def norm_01(data, dataMin, dataMax):
    return (data-dataMin)/(dataMax-dataMin)

def norm_gaussian(data):
    return (data-dataAvg)/dataStd

def renorm_01(data):
    return dataMin+(dataMax-dataMin)*data

def renorm_gaussian(data):
    return dataMean+dataStd*data

def main():
    train_file = "train.txt"
    train_ds = VolFracDataset(train_file)
    train_ldr = T.utils.data.DataLoader(train_ds, batch_size=256, shuffle=False)
    
    validate_file = "validate.txt"
    validate_ds = VolFracDataset(validate_file)
    validate_ldr = T.utils.data.DataLoader(validate_ds, batch_size=256, shuffle=False)
    
    test_file = "test.txt"
    test_ds = VolFracDataset(test_file)
    test_ldr = T.utils.data.DataLoader(test_ds, batch_size=256, shuffle=False)
    test_loss = 0

    optimizer = T.optim.Adam(net.parameters(), lr=0.001, \
                             betas=(0.9, 0.999), eps=1e-08, \
                             weight_decay=0, amsgrad=False)
    criterion = nn.MSELoss() #Loss function
    net = Net().float().to(device)

    #y=data[data[:,0].argsort()] # Sort data
    nEpoch = 1
    loss_arr = np.zeros((nEpoch,3))
    for epoch in range(nEpoch):
        loss = 0; loss_per_epoch = 0
        jj=0
        for (batch_idx, batch) in enumerate(train_ldr):
            X = batch['inp'].to(device)
            Y = batch['out'].to(device)
            optimizer.zero_grad()
            net_out = net(X.float()).reshape(-1)
            loss = criterion(net_out,Y.float())
            loss.backward() #back propagation 
            optimizer.step()
            loss_per_epoch +=loss.item()
            jj+=1
        train_loss = loss_per_epoch/jj

        #saving model should be here
        if ( (epoch % 20 == 0) or (epoch == nEpoch-1) ): 
            PATH = "./model/save_" + str(epoch) + ".pt"
            T.save({
                'epoch': epoch,
                'model_state_dict': net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': train_loss,
                }, PATH)
        
        # Validation process starts here:
        loss = 0; loss_per_epoch = 0
        jj=0
        for (batch_idx, batch) in enumerate(validate_ldr):
            X = batch['inp'] 
            Y = batch['out'] 
            net_out = net(X.float()).reshape(-1)
            loss = criterion(net_out,Y.float())
            loss_per_epoch += loss.item()
            jj += 1
        val_loss = loss_per_epoch/jj
        loss_arr[epoch,0] = epoch
        loss_arr[epoch,1] = train_loss
        loss_arr[epoch,2] = val_loss

        print("Epoch=%d: Train loss=%7.6E: Val loss=%7.6E" % (epoch, train_loss, val_loss))

    np.savetxt('loss.txt', loss_arr, header='Epoch Train_loss Validate_loss')
        
    #testing loop
    jj = 0
    for (batch_idx, batch) in enumerate(test_ldr):
        X = batch['inp'] #inputs
        Y = batch['out'] #output
        net_out = net(X.float()).reshape(-1) 
        test_loss += criterion(net_out, Y.float())
        jj += 1
        
#    test_loss /= len(test_ldr.dataset)
    test_loss /= jj
    print("Test loss: %7.6E" % test_loss)
 
main()
            
            
