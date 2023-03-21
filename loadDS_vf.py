import os.path
import numpy as np
import torch as T

device = T.device("cpu")

class VolFracDataset(T.utils.data.Dataset):
    def __init__(self, src_file):
        data = np.loadtxt(src_file)
        dataMin = data.min(0,keepdims=True)
        dataMax = data.max(0,keepdims=True)
        data_norm = norm_01(data, dataMin, dataMax)
        print(data_norm)

        x_tmp = data_norm[:,0:3] #np.loadtxt(src_file, usecols=range(0,3))
        y_tmp = data_norm[:,3:5] #np.loadtxt(src_file, usecols=range(3,5))

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
    train_file = dataFile
    # Instantiate the class
    train_ds = VolFracDataset(train_file)
    train_ldr = T.utils.data.DataLoader(train_ds, batch_size=3, shuffle=False)

    for epoch in range(1):
        print("\n==============================\n")
        print("Epoch = " + str(epoch))
        for (batch_idx, batch) in enumerate(train_ldr):
            print("\nBatch = " + str(batch_idx))
            X = batch['inp']
            Y = batch['out']
            print(X)

# Generate random data and write to a file
def genData():
    rho_lim = [1, 1770]
    e_lim = [0.125e6, 15e6]
    Y_lim = [0,1]
    p_lim = [1e4, 1e10]
    T_lim = [1e3, 1e4]
    d0 = np.random.rand(6,5)
    d0[:,0] = rho_lim[0] + d0[:,0]*(rho_lim[1]-rho_lim[0])
    d0[:,1] = e_lim[0] + d0[:,1]*(e_lim[1]-e_lim[0])
    d0[:,2] = Y_lim[0] + d0[:,2]*(Y_lim[1]-Y_lim[0])
    d0[:,3] = p_lim[0] + d0[:,3]*(p_lim[1]-p_lim[0])
    d0[:,4] = T_lim[0] + d0[:,4]*(T_lim[1]-T_lim[0])
    np.savetxt(dataFile, d0)

np.random.seed(15814)
dataFile = 'rand.txt'
if (~os.path.isfile(dataFile)): genData()
main()
#y=data_01[data_01[:,2].argsort()] # Sort data
