# http://www.cse.msu.edu/~tangjili/trust.html
# http://www.cse.msu.edu/~tangjili/datasetcode/README.txt

from scipy.io import loadmat
import random
import numpy as np
import math
import os

random_seed = 0
np.random.seed(random_seed) # numpy
random.seed(random_seed) # random and transforms

import sys

print('raw data')
data = loadmat('rating.mat')
print(data.keys())
rating = data['rating']
print('Number fo UI record', rating.shape)

print('trust data')
trust_data = loadmat('trustnetwork.mat')
print(trust_data.keys())
trustnetwork = trust_data['trustnetwork']
print('Number of social record', trustnetwork.shape)

data = []
for i in range (rating.shape[0]):
    if rating[i,3] > 3:
        uid = rating[i,0]
        iid = rating[i,1]
        data.append([uid, iid])

print(len(data))


data = []
user_set = set()
for i in range (rating.shape[0]):
    if rating[i,3] > 3:
        uid = rating[i,0]
        iid = rating[i,1]
        data.append([uid, iid])
        if uid not in user_set:
            user_set.add(uid)

print(len(data))

user_items = {}
for d in data:
    user = d[0]
    item = d[1]
    if user not in user_items:
        user_items[user] = set()
        user_items[user].add(item)
    else:
        if item not in user_items[user]:
            user_items[user].add(item)

def jaccard_similarity(user1, user2):
    items_u1 = user_items[user1]
    items_u2 = user_items[user2]
    intersection = len(items_u1.intersection(items_u2))
    union = len(items_u1.union(items_u2))
    if union == 0:
        return 0
    return intersection / union

similarities = []
for i in range(len(trustnetwork)):
    uid1 = trustnetwork[i,0]
    uid2 = trustnetwork[i,1]
    if uid1 in user_set and uid2 in user_set:
        sim = jaccard_similarity(uid1, uid2)
        similarities.append(sim)

mean_similarity = sum(similarities) / len(similarities)
print(mean_similarity)
sys.exit()







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

social_count = {}
for i in range(len(trustnetwork)):
    uid1 = trustnetwork[i,0]
    uid2 = trustnetwork[i,1]
    if uid1 not in social_count:
        social_count[uid1] = 0
    if uid2 not in social_count:
        social_count[uid2] = 0
    social_count[uid1] += 1
    social_count[uid2] += 1

print('Number of user in social grph: ', len(social_count))

user_set = set()
item_set = set()
minimum_record = 3

print()
print('minimum record: ', minimum_record)
for k, v in user_count.items():
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

'''  
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
'''

ratio = 0.8
train =[]
val = []
test = []
for user, items in user_items_dict.items():
    random.shuffle(items)
    if len(items[int(ratio * len(items)):]) >= 1:
        train_items = items[:int(ratio * len(items))]
        val_items = items[int(ratio * len(items)): int(0.9 * len(items))]
        test_items = items[int(0.9 * len(items)):]
    else:
        train_items = items
        test_items = []  # ------------
    
    train.append([user] + train_items)
    test.append([user] + test_items)
    val.append([user] + val_items)



trust_data = []
for i in range(len(trustnetwork)):
    uid1 = trustnetwork[i,0]
    uid2 = trustnetwork[i,1]
    if uid1 in user_reid_dict and uid2 in user_reid_dict:
        trust_data.append([user_reid_dict[uid1], user_reid_dict[uid2]])
trust_data = np.array(trust_data)
print(trust_data.shape)

trust_data = np.concatenate([trust_data, trust_data[:, [1, 0]]], axis=0)

print("current work file:", os.getcwd())

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

with open('_val_'+str(minimum_record)+'.txt', 'w') as file:
    for sublist in val:
        line = ' '.join(map(str, sublist))  
        file.write(line + '\n') 

print("Val data saved successfully:")

np.savetxt('_trust_'+str(minimum_record)+'.txt', trust_data, delimiter='\t', fmt='%d')
print('saved')



sys.exit()




# filter
print('filter rating > 3')
data = []
for i in range (rating.shape[0]):
    if rating[i,3] > 3:
        uid = rating[i,0]
        iid = rating[i,1]
        helpfulness = rating[i,4]
        data.append([uid, iid, helpfulness])
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
    
user_set = set()
item_set = set()
for k, v in user_count.items():
    if v > 3:
        user_set.add(k)
for k, v in item_count.items():
    if v > 3:
        item_set.add(k)

print('summary')
# 1: 167678
# 2: 140628
print(len(user_set))
print(max(user_set))
print(len(item_set))
print(max(item_set))

user_reid_dict = dict(zip(list(user_set), list(range(len(user_set)))))
item_reid_dict = dict(zip(list(item_set), list(range(len(item_set)))))
user_set = set(user_reid_dict.values())
item_set = set(item_reid_dict.values())
print(len(user_set))
print(max(user_set))
print(len(item_set))
print(max(item_set))
print()

data_all = []
for i in range(len(data)):
    uid = data[i][0]
    iid = data[i][1]
    if uid in user_reid_dict and iid in item_reid_dict:
        data_all.append([user_reid_dict[uid], item_reid_dict[iid], data[i][2]])

print('all data:')
print(len(data_all))  # 140,628

user_items_dict = {}
for i in range(len(data_all)):
    uid = data_all[i][0]
    iid = data_all[i][1]
    if uid in user_items_dict:
        user_items_dict[uid].append(iid)
    else:
        user_items_dict[uid] = [iid]
print(len(user_items_dict))

print('Split')
ratio = 0.8
train =[]
test = []
for user, items in user_items_dict.items():
    random.shuffle(items)
    if len(items) > 1:
        train_items = items[:int(0.8 * len(items))]
        test_items = items[int(0.8 * len(items)):]
    else:
        train_items = items
        test_items = []  # ------------
    
    train.append([user] + train_items)
    test.append([user] + test_items)
    

# print(len(train))
# print(len(test))

print()

print('trust data')
trust_data = loadmat('trustnetwork.mat')
print(trust_data.keys())
trustnetwork = trust_data['trustnetwork']
print(trustnetwork.shape)

trust_data = []
for i in range(len(trustnetwork)):
    uid1 = trustnetwork[i,0]
    uid2 = trustnetwork[i,1]
    if uid1 in user_reid_dict and uid2 in user_reid_dict:
        trust_data.append([user_reid_dict[uid1], user_reid_dict[uid2]])
trust_data = np.array(trust_data)
print(trust_data.shape)


# user_id: interacted item_id
user_interacted_dict = {}
for i in range(len(data_all)):
    uid = data_all[i][0]
    iid = data_all[i][1]
    if uid not in user_interacted_dict:
        user_interacted_dict[uid] = set()
    user_interacted_dict[uid].add(iid)
print(len(user_interacted_dict))

print("current work file:", os.getcwd())

with open('train_3.txt', 'w') as file:
    for sublist in train:
        line = ' '.join(map(str, sublist))  
        file.write(line + '\n')  

print("Train data saved successfully:")

with open('test_3.txt', 'w') as file:
    for sublist in test:
        line = ' '.join(map(str, sublist))  
        file.write(line + '\n') 

print("Test data saved successfully:")

np.savetxt('trust_3.txt', trust_data, delimiter='\t', fmt='%d')
print('saved')



'''
data_all = np.array(data_all)
np.random.shuffle(data_all)
print(data_all.shape)

print()

print('trust data')
trust_data = loadmat('trustnetwork.mat')
print(trust_data.keys())
trustnetwork = trust_data['trustnetwork']
print(trustnetwork.shape)

trust_data = []
for i in range(len(trustnetwork)):
    uid1 = trustnetwork[i,0]
    uid2 = trustnetwork[i,1]
    if uid1 in user_reid_dict and uid2 in user_reid_dict:
        trust_data.append([user_reid_dict[uid1], user_reid_dict[uid2]])
trust_data = np.array(trust_data)
print(trust_data.shape)



# user_id: interacted item_id
user_interacted_dict = {}
for i in range(data_all.shape[0]):
    uid = data_all[i,0]
    iid = data_all[i,1]
    if uid not in user_interacted_dict:
        user_interacted_dict[uid] = set()
    user_interacted_dict[uid].add(iid)

# user_id: trusted user_id
trust_dict = {}
for i in range(trust_data.shape[0]):
    uid1 = trust_data[i, 0]
    uid2 = trust_data[i, 1]
    if uid1 not in trust_dict:
        trust_dict[uid1] = []
    if uid2 not in trust_dict:
        trust_dict[uid2] = []
    trust_dict[uid1].append(uid2)
    trust_dict[uid2].append(uid1)

train = data_all[:int(0.8*data_all.shape[0]), :]
test = data_all[int(0.8*data_all.shape[0]):, :]

print()
print(train.shape)
print(test.shape)
print(trust_data.shape)
print()

print("current work file:", os.getcwd())
np.savetxt('data.txt', train, delimiter='\t', fmt='%d')
np.savetxt('test.txt', test, delimiter='\t', fmt='%d')
np.savetxt('trust.txt', trust_data, delimiter='\t', fmt='%d')
print('saved')

'''