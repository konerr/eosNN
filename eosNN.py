from torch.autograd import Variable
import torch.optim as optim
import numpy as np
import torch as T
import torch.nn as nn
import torch.nn.functional as F

# np.random.seed(13993)

if T.cuda.is_available():
    device = T.device("cuda")
    print('Running on ' + T.cuda.get_device_name(0))
else:
    device = T.device("cpu")

class FFNet(nn.Module):
    def __init__(self, n_layers=3, lyr_wdt=100, act_func=F.relu):
        super(FFNet, self).__init__()
        self.layers = nn.ModuleList([nn.Linear(3, lyr_wdt)])
        for i in range(1, n_layers):
            self.layers.append(nn.Linear(lyr_wdt,lyr_wdt))
        self.final_layer = nn.Linear(lyr_wdt,2)
        self.act_func = act_func

    def forward(self, x):
        for lyr in self.layers:
            x = self.act_func(lyr(x))
        #x = self.act_func(self.fc1(x)) # F.relu(self.fc1(x))
        #x = self.act_func(self.fc2(x)) # F.relu(self.fc2(x))
        x = self.final_layer(x)
        
        return x 

class Dataset(T.utils.data.Dataset):
# Inputs: rho_m, e_m, y_exp
# Outputs: p, T

    def __init__(self, src_file):
        data = np.loadtxt(src_file)
        # dataMin = data.min(0,keepdims=True)
        # dataMax = data.max(0,keepdims=True)
        # data_norm = norm_01(data, dataMin, dataMax)
        dataAvg = data.mean(0,keepdims=True)
        dataStd = data.std(0,keepdims=True)
        data_norm = norm_gaussian(data, dataAvg, dataStd)

        x_tmp = data_norm[:,0:3] #inputs
        y_tmp = data_norm[:,3:5] #outputs

        self.x_data = T.tensor(x_tmp, dtype=T.float).to(device)
        self.y_data = T.tensor(y_tmp, dtype=T.float).to(device)
        # self.dataMin = dataMin
        # self.dataMax = dataMax
        self.dataAvg = dataAvg
        self.dataStd = dataStd

    def __len__(self):
        return len(self.x_data)

    def __getitem__(self, idx):
        if T.is_tensor(idx):
          idx = idx.tolist()
        inp = self.x_data[idx, :]
        out = self.y_data[idx, :]
        dataAvg = self.dataAvg
        dataStd = self.dataStd
        sample = \
          { 'inp' : inp, 'out' : out, 'dmin' : dataAvg, 'dmax': dataStd }
        return sample

def norm_01(data, dataMin, dataMax):
    return (data-dataMin)/(dataMax-dataMin)

def norm_gaussian(data, dataAvg, dataStd):
    return (data-dataAvg)/(3.0*dataStd)

def renorm_01(data, dataMin, dataMax):
    return dataMin+(dataMax-dataMin)*data

def renorm_gaussian(data, dataAvg, dataStd):
    return dataMean+3.0*dataStd*data

def main():
    data_file = "PTEdata.dat"
    total_ds = Dataset(data_file)
    # Find total number of samples
    total_samples = total_ds.__len__()
    sample_ind = list(range(total_samples))
    train_lst = sample_ind[:int(0.6*total_samples)]
    val_lst = sample_ind[int(0.6*total_samples):int(0.8*total_samples)]
    test_lst = sample_ind[int(0.8*total_samples):]
   
    train_ds = T.utils.data.Subset(total_ds, train_lst)
    train_ldr = T.utils.data.DataLoader(train_ds, batch_size=128, shuffle=False)
    val_ds = T.utils.data.Subset(total_ds, val_lst)
    val_ldr = T.utils.data.DataLoader(val_ds, batch_size=128, shuffle=False)    
    test_ds = T.utils.data.Subset(total_ds, test_lst)
    test_ldr = T.utils.data.DataLoader(test_ds, batch_size=128, shuffle=False)
    
    test_loss = 0
    n_layers = 3; lyr_wdt = 100; act_func = F.relu

    net = FFNet(n_layers, lyr_wdt, act_func).to(device)
    optimizer = T.optim.Adam(net.parameters(), lr=0.001, \
                             betas=(0.9, 0.999), eps=1e-08, \
                             weight_decay=0, amsgrad=False)
    criterion = nn.MSELoss() #Loss function

    #y=data[data[:,0].argsort()] # Sort data
    nEpoch = 100
    loss_arr = np.zeros((nEpoch,3))
    for epoch in range(nEpoch):
        loss = 0; loss_per_epoch = 0
        jj=0
        for (batch_idx, batch) in enumerate(train_ldr):
            X = batch['inp'].to(device)
            Y = batch['out'].to(device)
            # print(X)
            optimizer.zero_grad()
            net_out = net(X) #.float())#.reshape(-1)
            loss = criterion(net_out,Y) #.float())
            loss.backward() #back propagation 
            optimizer.step()
            loss_per_epoch +=loss.item()
            jj+=1
        train_loss = loss_per_epoch/jj

        #saving model should be here
        #if ( (epoch % 20 == 0) or (epoch == nEpoch-1) ): 
        #    PATH = "./model/save_" + str(epoch) + ".pt"
        #    T.save({
        #        'epoch': epoch,
        #        'model_state_dict': net.state_dict(),
        #        'optimizer_state_dict': optimizer.state_dict(),
        #        'loss': train_loss,
        #        }, PATH)
        
        # Validation process starts here:
        loss = 0; loss_per_epoch = 0
        jj=0
        with T.no_grad():
            for (batch_idx, batch) in enumerate(val_ldr):
                X = batch['inp'] 
                Y = batch['out'] 
                net_out = net(X)
                loss = criterion(net_out,Y)
                loss_per_epoch += loss.item()
                jj += 1
        val_loss = loss_per_epoch/jj
        loss_arr[epoch,0] = epoch
        loss_arr[epoch,1] = train_loss
        loss_arr[epoch,2] = val_loss

        print("Epoch=%d: Train loss=%7.6E: Val loss=%7.6E" % (epoch, train_loss, val_loss))
        # print("Epoch=%d: Train loss=%7.6E" % (epoch, train_loss))

    # np.savetxt('loss.txt', loss_arr, header='Epoch Train_loss Validate_loss')
    
    #testing loop
    jj = 0
    inputs_lst = []
    outputs_lst = []
    predictions_lst = []
    with T.no_grad():
        for (batch_idx, batch) in enumerate(test_ldr):
            X = batch['inp'] #inputs
            Y = batch['out'] #output
            net_out = net(X)

            # Bring everything back to cpu
            X = X.cpu().numpy()
            Y = Y.cpu().numpy()
            net_out = net_out.detach().cpu().numpy()

            # Append these to the lists
            inputs_lst.append(X)
            outputs_lst.append(Y)
            predictions_lst.append(net_out)

    inputs_np = np.concatenate(inputs_lst,axis=0)
    outputs_np = np.concatenate(outputs_lst,axis=0)
    predictions_np = np.concatenate(predictions_lst,axis=0)

    num_val_p = np.sum((outputs_np[:,0]-predictions_np[:,0])**2)
    den_val_p = np.sum((outputs_np[:,0]-np.mean(outputs_np[:,0]))**2)

    print(1 - (num_val_p/den_val_p)) 

    #test_loss += criterion(net_out, Y.float())
    #jj += 1
        
# #    test_loss /= len(test_ldr.dataset)
#     test_loss /= jj
#     print("Test loss: %7.6E" % test_loss)
 
main()
            
            
