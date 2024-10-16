from scipy.io import loadmat
import random
import numpy as np
import math
import os

random_seed = 0
np.random.seed(random_seed) # numpy
random.seed(random_seed) # random and transforms

import sys

with open('rating.txt', 'r') as f:
    lines = f.readlines()

data = []
user_set = set()
for line in lines:
    rating = line.strip().split('\t')
    
    uid = int(rating[0])
    iid = int(rating[1])
    data.append([uid, iid])
    if uid not in user_set:
        user_set.add(uid)
        
print(len(data))

user_count = {}
item_count = {}
for i in range(len(data)):
    uid = data[i][0]
    iid = data[i][1]
    if uid not in user_count:
        user_count[uid] = 0
    if iid not in item_count:
        item_count[iid] = 0
    user_count[uid] += 1
    item_count[iid] += 1
    
print('Number of user in UI grph: ', len(user_count))
print('Number of item in UI grph: ', len(item_count))

print('trust data')

with open('trustnetwork.txt', 'r') as f:
    lines = f.readlines()   

trustnetwork = []
social_count = {}
for line in lines:
    friends = line.strip().split(' ')
    trustnetwork.append([int(friends[0]), int(friends[1])])
    uid1 = int(friends[0])
    uid2 = int(friends[1])
    if uid1 not in social_count:
        social_count[uid1] = 0
    if uid2 not in social_count:
        social_count[uid2] = 0
    social_count[uid1] += 1
    social_count[uid2] += 1
print(len(trustnetwork))


print('Number of user in social grph: ', len(social_count))

user_set = set()
item_set = set()
minimum_record = 3

for k, v in user_count.items():
    # if (k in social_count) and (v > minimum_record):
    if (k in social_count) and (social_count[k] > minimum_record) and (v > minimum_record):
        user_set.add(k)
        
for k, v in item_count.items():
    if v > minimum_record:
        item_set.add(k)

print('Number of user',len(user_set))
print('Max user ID',max(user_set))
print('Number of item',len(item_set))
print('Max item ID',max(item_set))

user_reid_dict = dict(zip(list(user_set), list(range(len(user_set)))))
item_reid_dict = dict(zip(list(item_set), list(range(len(item_set)))))
user_set = set(user_reid_dict.values())
item_set = set(item_reid_dict.values())
print(len(user_set))
print(max(user_set))
print(len(item_set))
print(max(item_set))

data_all = []
for i in range(len(data)):
    uid = data[i][0]
    iid = data[i][1]
    if uid in user_reid_dict and iid in item_reid_dict:
        data_all.append([user_reid_dict[uid], item_reid_dict[iid]])

print('all data:')
print(len(data_all))  # 110,940
user_set = set(user_reid_dict.values())
item_set = set(item_reid_dict.values())

user_items_dict = {}
for i in range(len(data_all)):
    uid = data_all[i][0]
    iid = data_all[i][1]
    if uid in user_items_dict:
        user_items_dict[uid].append(iid)
    else:
        user_items_dict[uid] = [iid]
        
print('Number of user', len(user_items_dict))

ratio = 0.9
train =[]
test = []
for user, items in user_items_dict.items():
    random.shuffle(items)
    if len(items[int(ratio * len(items)):]) >= 1:
        train_items = items[:int(ratio * len(items))]
        test_items = items[int(ratio * len(items)):]
    else:
        train_items = items
        test_items = []  # ------------
    
    train.append([user] + train_items)
    test.append([user] + test_items)

trust_data = []
for i in range(len(trustnetwork)):
    uid1 = trustnetwork[i][0]
    uid2 = trustnetwork[i][1]
    if uid1 in user_reid_dict and uid2 in user_reid_dict:
        trust_data.append([user_reid_dict[uid1], user_reid_dict[uid2]])
trust_data = np.array(trust_data)
print(trust_data.shape)

with open('_train_'+str(minimum_record)+'.txt', 'w') as file:
    for sublist in train:
        line = ' '.join(map(str, sublist))  
        file.write(line + '\n')  

print("Train data saved successfully:")

with open('_test_'+str(minimum_record)+'.txt', 'w') as file:
    for sublist in test:
        line = ' '.join(map(str, sublist))  
        file.write(line + '\n') 

print("Test data saved successfully:")

# with open('_val_'+str(minimum_record)+'.txt', 'w') as file:
#     for sublist in val:
#         line = ' '.join(map(str, sublist))  
#         file.write(line + '\n') 

# print("Val data saved successfully:")

np.savetxt('_trust_'+str(minimum_record)+'.txt', trust_data, delimiter='\t', fmt='%d')
print('saved')