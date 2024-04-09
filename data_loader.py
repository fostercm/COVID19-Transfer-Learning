import torch
from torch.utils.data import DataLoader as torch_dataloader
from torch.utils.data import Dataset as torch_dataset
import numpy as np
#import matplotlib.pyplot as plt
import skimage
import skimage.io as io
#import glob
import pandas as pd
#%%
class MyDataset(torch_dataset):
    def __init__(self, path, filenamelist, labellist):
        self.path=path
        self.filenamelist=filenamelist
        self.labellist=labellist
    def __len__(self):
        #return the number of data points
        return len(self.filenamelist)
    def __getitem__(self, idx):
        I=io.imread(self.path+self.filenamelist[idx])
        I=skimage.util.img_as_float32(I)
        I = I.reshape(1,I.shape[0],I.shape[1])
        I = torch.tensor(I, dtype=torch.float32)
        I = I.expand(3, I.shape[1],I.shape[2])
        label=torch.tensor(self.labellist[idx], dtype=torch.int64)
        return I, label
#%%
def get_dataloader(path='C:/S224/'):    
    df_train=pd.read_csv(path+'train.csv')
    dataset_train = MyDataset(path, df_train['filename'].values, df_train['label'].values)
    loader_train = torch_dataloader(dataset_train, batch_size=32, num_workers=0, shuffle=True, pin_memory=True)
    df_val=pd.read_csv(path+'val.csv')
    dataset_val = MyDataset(path, df_val['filename'].values, df_val['label'].values)
    loader_val = torch_dataloader(dataset_val, batch_size=32, num_workers=0, shuffle=True, pin_memory=True)
    df_test=pd.read_csv(path+'test.csv')
    dataset_test = MyDataset(path, df_test['filename'].values, df_test['label'].values)
    loader_test = torch_dataloader(dataset_test, batch_size=32, num_workers=0, shuffle=True, pin_memory=True)
    
	#modify this function to return loader_train, loader_val, and loader_test
    return loader_train, loader_val, loader_test