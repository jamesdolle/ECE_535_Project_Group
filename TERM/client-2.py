
import copy
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

import pdb

import sys
#import tensorflow as tf
sys.path.append('../../')
from FL.utils.dataclass import ClientsParams


# from FL.utils.utils import define_classification_model, softmax
# from FL.utils.dataclass import ClientsParams



class CreateDataset(Dataset):
    """
    An abstract Dataset class wrapped around Pytorch Dataset class.
    """
    

    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = [int(i) for i in idxs]

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        label = self.dataset[item][1]
        image = self.dataset[item][0]
        #for i in range(0,len(self.dataset[item][0])):
        #    new_im = self.dataset[item][0][0][i]
        #    m = tf.reduce_max(new_im, axis=0)
        #    b = tf.math.cumprod(m, exclusive=True)
        #    h = tf.reduce_sum(new_im * b, axis=1)
        #    _, idx, counts = tf.unique_with_counts(h)
        #    groups = tf.argsort(idx)
        #print(groups.numpy())
        #print("Image: ",image,"\n")
        #print("Label: ",torch.tensor(label))
        new_1=[]
        if (label==0):
            new_1+=[image]
	
        return image, torch.tensor(label)
    	#image, label = self.dataset[self.idxs[item]]
    	#return image, torch.tensor(label)


class LocalBase():
    def  __init__(self,args,train_dataset,test_dataset,client_id):
        print("\n INIT DATA \n")
        self.args = args
        self.client_id = client_id
        self.std = 1
        self.mean = 0        
        # use a for loop to iterate through train_data set and add a random
        # https://discuss.pytorch.org/t/how-to-add-noise-to-mnist-dataset-when-using-pytorch/59745
        # torch.randn(tensor.size()) * self.std + self.mean
        # add to ^ number to every image 
        
        cLength = args.train_distributed_data[client_id]
        self.dLength_train = len([int(i) for i in cLength])
        
        cLength = args.test_distributed_data[client_id]
        self.dLength_test = len([int(i) for i in cLength])
        #dataSet = list(train_dataset)
        #print(type(dataSet))
        
        dataSet_train = []
        data_test_noise = []
                
        for x in range(self.dLength_train):
            dataSet_train.append( [train_dataset[x][0], train_dataset[x][1]] )
            #print(dataSet_train[x][0])
            
        for x in range(self.dLength_train):
            dataSet_train[x][0] += torch.randn(dataSet_train[x][0].size()) * self.std + self.mean
            
            
        for x in range(self.dLength_test):
            data_test_noise.append( [train_dataset[x][0], train_dataset[x][1]] )
  
        for x in range(self.dLength_test):
            data_test_noise[x][0] += torch.randn(data_test_noise[x][0].size()) * self.std + self.mean
    
        
        #self.trainDataset=CreateDataset(train_dataset, args.train_distributed_data[client_id])
        #self.testDataset=CreateDataset(test_dataset, args.test_distributed_data[client_id])
        #print(a,type(a),"111111\n")
        #print(b,type(b),"222222\n")
        #c, d = self.trainDataset.__getitem__(75)
        #print(c,type(c),"333333\n")
        #print(d,type(d),"444444\n")
        #print(tensor(1))
        #print("HERE")
        #print(type(b))
        #print(type(data_test_noise[100][0]))
        #print(type(data_test_noise[100][1]))
        
        #print(dataSet_train[107][0])
        #print(useless_tuple_train[107][0])
        #print(type(useless_tuple_train[107]))
        
        #print(train_dataset[100],"!!!\n")
        #print(train_dataset[100][1])
        #print(dataSet_train[100])
        
        self.trainDataset=CreateDataset(dataSet_train, args.train_distributed_data[client_id])
        self.testDataset=CreateDataset(data_test_noise, args.test_distributed_data[client_id])
        #print(dataSet_train[5][1],"!!!!\n")
        
        #print(self.trainDataset.__getitem__(100))
        #for x in range(2,3): ###change to self.dLength_train after debugging
        #    a, b = self.trainDataset.__getitem__(x)
        #for x in range(2,3): ###change to self.dLength-test after debugging
        #    c, d = self.testDataset.__getitem__(x)
        self.trainDataloader=DataLoader(self.trainDataset, args.batch_size, shuffle=True)
        self.testDataloader=DataLoader(self.testDataset, args.batch_size, shuffle=True)

        #self.device = 'cuda' if args.on_cuda else 'cpu'
        self.device = 'cpu' #new code
        self.criterion = nn.CrossEntropyLoss(reduction="mean")
        #print("Is it here?\n")
        self.show_class_distribution(self.trainDataset,self.testDataset)
    
    def show_class_distribution(self,train,test):
        print("Class distribution of id:{}".format(self.client_id))
        class_distribution_train=[ 0 for _ in range(10)]
        class_distribution_test=[ 0 for _ in range(10)]
        for _, c in train:
            class_distribution_train[c]+=1
        for _, c in test:
            class_distribution_test[c]+=1
        print("train",class_distribution_train)
        print("test",class_distribution_test)

    def local_validate(self,model):
        #print(type(model),model,"!!!")
        model.eval()
        #print(type(model.eval()),model.eval(),"@@@@")
        model.to(self.device)
        correct = 0
        batch_loss = []
        #print(type(model.to(self.device)),model.to(self.device),"!!?!!\n")
        #print(type(torch.no_grad()),torch.no_grad(),"@@@@\n")
        with torch.no_grad():
            for images, labels in self.testDataloader:
                images, labels = images.to(self.device), labels.to(self.device)
                if (label==0):
                     if images.shape[1]==1:
                         images=torch.cat((images, images, images), 1)
                     output = model(images)
                     pred = output.argmax(dim=1, keepdim=True)
                     correct += pred.eq(labels.view_as(pred)).sum().item()
                     loss = self.criterion(output, labels)
                     batch_loss.append(loss.item())
        test_acc=100. * correct / len(self.testDataloader.dataset)
        test_loss=sum(batch_loss)/len(batch_loss)
        print('| Client id:{} | Test_Loss: {:.3f} | Test_Acc: {:.3f}'.format(self.client_id,test_loss, test_acc))
        return test_acc, test_loss

    def update_weights(self,model,global_epoch):
        model.train()
        model.to(self.device)
        
        if self.args.optimizer == 'sgd':
            optimizer = torch.optim.SGD(model.parameters(), lr=self.args.lr,momentum=0.9, weight_decay=5e-4)
        elif self.args.optimizer == 'adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=self.args.lr, weight_decay=1e-4)
        
        for epoch in range(1,self.args.local_epochs+1):
            
            batch_loss = []
            correct = 0

            for batch_idx, (images, labels) in enumerate(self.trainDataloader):
                images, labels = images.to(self.device), labels.to(self.device)
                if images.shape[1]==1:
                    images=torch.cat((images, images, images), 1)

                optimizer.zero_grad()
                output = model(images)
                pred = output.argmax(dim=1, keepdim=True)  
                correct += pred.eq(labels.view_as(pred)).sum().item()
                
                loss = self.criterion(output, labels)
                loss.backward()
                #pdb.set_trace()

                optimizer.step()
                    
                #self.logger.add_scalar('loss', loss.item())、あとでどっかに学習のログ
                batch_loss.append(loss.item())

            train_acc,train_loss=100. * correct / len(self.trainDataloader.dataset),sum(batch_loss)/len(batch_loss)
            print('| Global Round : {}/{} | Client id:{} | Local Epoch : {}/{} |  Train_Loss: {:.3f} | Train_Acc: {:.3f}'.format(
                        global_epoch,self.args.global_epochs, self.client_id, epoch,self.args.local_epochs,train_loss, train_acc))
        
        return model.state_dict()      
        
class Fedavg_Local(LocalBase):
    def __init__(self,args,train_dataset,val_dataset,client_id):
        super().__init__(args,train_dataset,val_dataset,client_id)
    
    def localround(self,model,global_epoch,validation_only=False):
        
        self.local_validate(model)
        if validation_only:
            return 
        #update weights
        self.updated_weight=self.update_weights(model,global_epoch)
        
        clients_params=ClientsParams(weight=self.updated_weight)
        
        return clients_params
class TERM_Local(LocalBase):
    def __init__(self,args,train_dataset,val_dataset,client_id):
        super().__init__(args,train_dataset,val_dataset,client_id)
    
    def localround(self,model,global_epoch,validation_only=False):
        
        self.local_validate(model)
        if validation_only:
            return 
        #update weights
        self.updated_weight=self.update_weights(model,global_epoch)
        
        clients_params=ClientsParams(weight=self.updated_weight)
        
        return clients_params

class Afl_Local(LocalBase):
    def __init__(self,args,train_dataset,val_dataset,client_id):
        super().__init__(args,train_dataset,val_dataset,client_id)
        
    def localround(self,model,global_epoch,validation_only=False):

        _, test_loss=self.local_validate(model)
        if validation_only:
            return 
        #update weights
        self.updated_weight=self.update_weights(model,global_epoch)
        
        clients_params=ClientsParams(weight=self.updated_weight,afl_loss=test_loss)
        return clients_params


     
def define_localnode(args,train_dataset,val_dataset,client_id):
    if args.federated_type=='fedavg':#normal
        return Fedavg_Local(args,train_dataset,val_dataset,client_id)
        
    elif args.federated_type=='afl':#afl
        return Afl_Local(args,train_dataset,val_dataset,client_id)
    elif args.federated_type=='TERM':#TERM
        return TERM_Local(args,train_dataset,val_dataset,client_id)
    else:       
        raise NotImplementedError     
    
