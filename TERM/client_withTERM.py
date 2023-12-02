
import copy
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
import tensorflow_probability as tfp
import pdb

import sys
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
        
        dataSet_train = []
        data_test_noise = []
                
        print("-----------------client-------------------")
        for x in range(self.dLength_train):
            dataSet_train.append( [train_dataset[x][0], train_dataset[x][1]] )
            
        for x in range(self.dLength_train):
            dataSet_train[x][0] += torch.randn(dataSet_train[x][0].size()) * self.std + self.mean
            
            
        for x in range(self.dLength_test):
            data_test_noise.append( [train_dataset[x][0], train_dataset[x][1]] )
  
        for x in range(self.dLength_test):
            data_test_noise[x][0] += torch.randn(data_test_noise[x][0].size()) * self.std + self.mean

        data_test_noise=self.TERM(data_test_noise)
        self.trainDataset=CreateDataset(dataSet_train, args.train_distributed_data[client_id])
        self.testDataset=CreateDataset(data_test_noise, args.test_distributed_data[client_id])

        self.trainDataloader=DataLoader(self.trainDataset, args.batch_size, shuffle=True)
        self.testDataloader=DataLoader(self.testDataset, args.batch_size, shuffle=True)

        self.device = 'cpu' #new code
        self.criterion = nn.CrossEntropyLoss(reduction="mean")

        self.show_class_distribution(self.trainDataset,self.testDataset)
    def TERM(self,noise_data):
        #print(noise_data[0][0][0][0],"!!!\n")
        #print(noise_data[0][0][0][0][0])
        #print(len(noise_data)) 3330
        #print(len(noise_data[0])) 2
        #print(len(noise_data[0][0])) 3
        #print(len(noise_data[0][0][0])) 32
        #print(len(noise_data[0][0][0][0])) 32
        #print(torch.max(noise_data))
        #print(noise_data[0][1][0][0][0])
        for c in range(len(noise_data)):
          if self.is1DList(noise_data[c]) == 1:
             median = self.get_medi(noise_data[c])
             noise_data[c]=self.scan(noise_data[c],median)       
          elif self.is1DList(noise_data[c]) == -1:
             pass
          elif self.is1DList(noise_data[c]) == 0:
             for d in range(len(noise_data[c])):
               if self.is1DList(noise_data[c][d]) == 1:
                  median = self.get_medi(noise_data[c][d])       
                  noise_data[c][d]=self.scan(noise_data[c][d],median)       
               elif self.is1DList(noise_data[c][d]) == -1:
                  pass
               elif self.is1DList(noise_data[c][d]) == 0:
                  for e in range(len(noise_data[c][d])):
                    if self.is1DList(noise_data[c][d][e]) == 1:
                      median = self.get_medi(noise_data[c][d][e])       
                      noise_data[c][d][e]=self.scan(noise_data[c][d][e],median)       
                    elif self.is1DList(noise_data[c][d][e]) == -1:
                      pass
                    elif self.is1DList(noise_data[c][d][e]) == 0:
                      for f in range(len(noise_data[c][d][e])): 
                        if self.is1DList(noise_data[c][d][e][f]) == 1:
                          median = self.get_medi(noise_data[c][d][e][f])       
                          noise_data[c][d][e][f]=self.scan(noise_data[c][d][e][f],median)       
                        elif self.is1DList(noise_data[c][d][e][f]) == -1:
                          pass
        return noise_data
    def is1DList(self,alist):
        if type(alist) is int:
           return -1
        for item in alist:
           if type(item) is int:
              continue
           else:
              return 0
        return 1
    def get_medi(self,DList):
       temp_list=DList.sort()
       if len(temp_list)%2==0:
           lw = temp_list[len(temp_list)//2-1].item()
           hw = temp_list[len(temp_list)//2].item()
           median = (lw+hw)/2
       if len(temp_list)%2==1:
           median = temp_list[len(temp_list)//2].item()
       return median
    def scan(self,FList,midd):
       for i in range(len(FList)):
         if FList[i].item()>midd*2:
            FList[i] = midd*2
         elif FList[i].item()<midd/2:
            FList[i] = midd/2
       return FList
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
        model.eval()
        model.to(self.device)
    
        correct = 0
        batch_loss = []
    
        class_correct = [0] * 10  # Assuming CIFAR-10 has 10 classes
        class_total = [0] * 10

        class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    	
        with torch.no_grad():
       	    for images, labels in self.testDataloader:
                images, labels = images.to(self.device), labels.to(self.device)
            
                if images.shape[1] == 1:
                    images = torch.cat((images, images, images), 1)
    
                output = model(images)
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(labels.view_as(pred)).sum().item()
            
                loss = self.criterion(output, labels)
                batch_loss.append(loss.item())
            
                # Calculate class-wise accuracy
                for c in range(10):  # Assuming CIFAR-10 has 10 classes
                    class_indices = labels == c
                    class_correct[c] += pred[class_indices].eq(labels[class_indices].view_as(pred[class_indices])).sum().item()
                    class_total[c] += class_indices.sum().item()
                
            # Calculate overall test accuracy and average test loss
            test_acc = 100. * correct / len(self.testDataloader.dataset)
            test_loss = sum(batch_loss) / len(batch_loss)
        
            # Print the results for the entire dataset
            print('| Client id:{} | Test_Loss: {:.3f} | Test_Acc: {:.3f}'.format(self.client_id, test_loss, test_acc))
        
            # Perform class-wise evaluation
            for c, class_name in enumerate(class_names):  # Assuming CIFAR-10 has 10 classes
                class_acc = 100. * class_correct[c] / class_total[c]
                print(f'| Class: {c} - {class_name.ljust(10, " ")} | Class_Acc: {class_acc:.3f}')

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
            print('Global Round : {}/{} | Client id:{} | Local Epoch : {}/{} | Train_Loss: {:.3f} | Train_Acc: {:.3f}'.format(
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

    else:       
        raise NotImplementedError     
    
