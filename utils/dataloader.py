import scipy.sparse as sp
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
import torch
from torch.utils.data import Dataset
import random

import warnings
warnings.filterwarnings("ignore")

import sys
# import dgl

def load_data(dataset):
    print('Dataset: ' + dataset)
    if dataset == 'ciao':
        return DataLoader("data/ciao/")
    if dataset == 'lastFM':
        return DataLoader("data/lastFM/")
        # return DataLoader2("data/ciao_dc/")
    if dataset =='Dianping':
        return DataLoader("data/Dianping/")
    elif dataset == 'Epinions':
        return DataLoader("data/Epinions/")
    elif dataset == 'gowalla':
        return DataLoader("data/gowalla/")
    elif dataset == 'kuairec':
        return DataLoader("data/kuairec/")
    

class BasicDataset(Dataset):
    def __init__(self):
        print("init dataset")

    @property
    def n_users(self):
        raise NotImplementedError
    
    @property
    def n_items(self):
        raise NotImplementedError
    
    @property
    def trainDataSize(self):
        raise NotImplementedError
    
    @property
    def allPos(self):
        raise NotImplementedError
    
    def getSparseGraph(self):
        """
        build a graph in torch.sparse.IntTensor.
        Details in NGCF's matrix form
        A = 
            |I,   R|
            |R^T, I|
        """
        raise NotImplementedError



class DataLoader(BasicDataset):
    def __init__(self, path="data/ciao/"):
        
        train_file = path + '/train_3.txt'
        test_file = path + '/test_3.txt'
        social_file = path + '/trust_3.txt'

        trainUniqueUsers, trainUser, trainItem = [], [], []
        testUniqueUsers, testUser, testItem = [], [], []
        
        with open(train_file) as f:
            for l in f.readlines():
                if len(l) > 0:
                    l = l.strip('\n').split(' ')
                    items = [int(i) for i in l[1:]]
                    uid = int(l[0])
                    
                    trainUniqueUsers.append(uid)
                    trainUser.extend([uid] * len(items))
                    trainItem.extend(items)
        
        self.num_users = np.max(trainUser)  # 22109
        self.num_items = np.max(trainItem)  # 45463
        
        self.num_users += 1
        self.num_items += 1
        self.traindataSize = len(trainUser)
        
        self.trainUniqueUsers = np.array(trainUniqueUsers)
        self.trainUser = np.array(trainUser)  # train user 
        self.trainItem = np.array(trainItem)  # train item 
        
        with open(test_file) as f:
            for l in f.readlines():
                if len(l) > 0:
                    l = l.strip('\n').split(' ')
                    items = [int(i) for i in l[1:]]
                    uid = int(l[0])
                    if len(items) > 0:
                        testUniqueUsers.append(uid)
                        testUser.extend([uid] * len(items))
                        testItem.extend(items)
                    
        self.testUniqueUsers = np.array(testUniqueUsers)
        self.testUser = np.array(testUser)  
        self.testItem = np.array(testItem)  
        
        self.testDataSize = len(testUser)
        
        self.interactionGraph = None
        print(f"Number of items: {self.n_items}")
        print(f"Number of users: {self.n_users}")
        print(f"{self.trainDataSize} interactions for training")
        print(f"{self.testDataSize} interactions for testing")
        print(f"UI-Graph Sparsity : {(self.trainDataSize + self.testDataSize) / self.n_users / self.n_items}")
        
        # (users,items), bipartite graph
        self.UserItemNet = csr_matrix((np.ones(len(self.trainUser)), (self.trainUser, self.trainItem)),
                                      shape=(self.num_users, self.num_items))
        
        # allItems     = set(range(self.num_items))
        self._allPos = self.getUserPosItems(list(range(self.n_users)))
        
        '''
        self.allNeg = []
        for i in range(self.n_users):
            pos = set(self._allPos[i])
            neg = allItems - pos
            self.allNeg.append(np.array(list(neg)))
        '''    
        self.__testDict = self.__build_test()
        self.__neg_sample_test = None
        
        self.trustNet = pd.read_table(social_file, header=None).to_numpy()
        print(f"{int(self.trustNet.shape[0]/2)} relations")
        print(f"Social-Graph Sparsity : {(self.trustNet.shape[0]/2) / self.n_users / self.num_users }")
        
        self.LGNc = True
    @property
    def n_users(self):
        return self.num_users
    
    @property
    def n_items(self):
        return self.num_items
    
    @property
    def trainDataSize(self):
        return self.traindataSize
    
    @property
    def allPos(self):
        return self._allPos
    
    @property
    def testDict(self):
        return self.__testDict
    
    def getInterGraph(self):
        if self.interactionGraph is None:
            
            # build up matrix with n.users+n.items, n.users+n.items
            adj_mat = sp.dok_matrix((self.n_users + self.n_items, self.n_users + self.n_items), dtype=np.float32)
            adj_mat = adj_mat.tolil()
            R = self.UserItemNet.tolil()
            adj_mat[:self.n_users, self.n_users:] = R  
            adj_mat[self.n_users:, :self.n_users] = R.T  
            if self.LGNc:
                socialNet = csr_matrix((np.ones(len(self.trustNet)), (self.trustNet[:,0], self.trustNet[:,1]) ), shape=(self.n_users,self.n_users), dtype=np.float32)
                S = socialNet.tolil()
                adj_mat[:self.n_users, :self.n_users:] = S
            adj_mat = adj_mat.todok()
            
            # calculate D
            rowsum = np.array(adj_mat.sum(axis=1))
            d_inv = np.power(rowsum, -0.5).flatten()
            d_inv[np.isinf(d_inv)] = 0.
            d_mat = sp.diags(d_inv)
            
            norm_adj = d_mat.dot(adj_mat)
            norm_adj = norm_adj.dot(d_mat)
            norm_adj = norm_adj.tocsr()
            
            self.interactionGraph = self._convert_sp_mat_to_sp_tensor(norm_adj)
            self.interactionGraph = self.interactionGraph.coalesce()
            
            
        return self.interactionGraph
    
    def getSocialGraph(self, trustNet):
        # trustNet = np.concatenate([trustNet, trustNet[:, [1, 0]]], axis=0) 
        socialNet = csr_matrix((np.ones(len(trustNet)), (trustNet[:,0], trustNet[:,1]) ), shape=(self.n_users,self.n_users), dtype=np.float32)
        R = socialNet.tolil()
        adj_mat = (R != 0) * 1.0
        adj_mat = (adj_mat + sp.eye(adj_mat.shape[0])) * 1.0

        # self.socialGraph = self._convert_sp_mat_to_sp_tensor(norm_adj)
        self.socialGraph = self._convert_sp_mat_to_sp_tensor(adj_mat)
        self.socialGraph = self.socialGraph.coalesce()
        
        
        return self.socialGraph
        
    
    def _convert_sp_mat_to_sp_tensor(self, X):
        coo = X.tocoo().astype(np.float32)
        row = torch.Tensor(coo.row).long()
        col = torch.Tensor(coo.col).long()
        index = torch.stack([row, col])
        data = torch.FloatTensor(coo.data)
        return torch.sparse.FloatTensor(index, data, torch.Size(coo.shape))
    
    
    def getUserPosItems(self, users):
        posItems = []
        for user in users:
            posItems.append(self.UserItemNet[user].nonzero()[1])
        return posItems
    
    def padPos(self):
        self.pad_allPos = []
        print(len(self.allPos))

        fix_length = 10
        
        for i in range(len(self.allPos)):
            pos = self.allPos[i]
            
            if len(pos) >= fix_length:
                self.pad_allPos.append(pos)
            else:
                if len(pos) == 0:
                    print(i)
                    sys.exit()
                
                remaining_length = fix_length - len(pos)
                filled_list = list(pos) + random.choices(pos, k=remaining_length)
                self.pad_allPos.append(np.array(filled_list))
    
    def __build_test(self):
        """
        return:
            dict: {user: [items]}
        """
        test_data = {}
        for i, item in enumerate(self.testItem):
            user = self.testUser[i]
            if test_data.get(user):
                test_data[user].append(item)
            else:
                test_data[user] = [item]
        return test_data
    
    def get_neg_sample_test(self, n):
        if self.__neg_sample_test is None:
            self.__neg_sample_test = {}
            for k, v in self.__testDict.items():
                all_num = set(range(0, self.n_items)) - set(v) - set(self.allPos[k])
                negative = []
                while len(negative) < n:
                    rand_num = random.choice(list(all_num))
                    negative.append(rand_num)
                    all_num.remove(rand_num)
                self.__neg_sample_test[k] = negative
                
        return self.__neg_sample_test