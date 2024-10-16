from utils import utils
from utils import losses
from utils import sampling
from utils.step_sampler import create_diffusion_step_sampler
import torch
from torch import optim
from scipy.sparse import csr_matrix
import time
import numpy as np
import json
import random
import datetime

import sys


class TrainLoop:
    def __init__(self, 
                 world, 
                 dataset,
                 model,
                 denoise_model):
        self.world = world
        self.dataset = dataset
        self.batch_size = self.world.config['batch_size']
        
        self.sde, self.sampling_eps = utils.load_sde(self.world.config['sde'], self.world.config_sde)
        self.diffuse_loss = losses.get_loss_fn(self.sde, self.world.args.reduce_mean, self.world.args.continuous, self.world.args.likelihood_weighting)
        
        self.weight_decay = self.world.args.weight_decay
        self.weight_decay_social = self.world.args.weight_decay_social
        self.ssl_wd = self.world.args.ssl_wd
        
        self.model = model
        self.lr = self.world.config['lr']
        self.opt = optim.Adam(self.model.parameters(), 
                              lr=self.lr)  
        
        self.epoches = world.args.epoches
        self.stepscheduler = 'uniform'  # lossaware  uniform
        self.scheduler = create_diffusion_step_sampler(self.stepscheduler, self.sde.N)
        
        self.sampling_fn = sampling.get_sampling_fn(self.world,self.sde)
        
        self.ema = utils.load_ema(self.model, self.world.args.ema)
        
        self.record = []
        self.best = {}
        
        self.best['recall'] = [0.0] * len(self.world.topks)
        self.best['precision'] = [0.0] * len(self.world.topks)
        self.best['ndcg'] = [0.0] * len(self.world.topks)
        
        self.pretrain_epoch = 10
        self.cfg_prob = 1.0
        
    def run_loop(self):
        trustNet = self.dataset.trustNet
        
        print("Start training")
        self.start_time = time.time()
        
        for epoch in range(1, self.epoches+1):
            self.model.train()
            S = utils.UniformSample_original_python(self.dataset)
            
            users = torch.Tensor(S[:, 0]).long()
            posItems = torch.Tensor(S[:, 1]).long()
            negItems = torch.Tensor(S[:, 2]).long()
            users, posItems, negItems = utils.shuffle(users, posItems, negItems)  
            total_batch = len(users) // self.batch_size + 1
            aver_ssl_loss = 0.
            aver_bpr_loss = 0.
            aver_diff_loss = 0.
            n = self.world.args.N
            m = self.world.args.M
            
            original_social_graph = self.dataset.getSocialGraph(trustNet)
            
            if epoch == 1:
                social_g = original_social_graph.to(self.world.device)
                self.social_drop_out = False

            elif epoch % (m+n) == 1 or epoch == 2:
                print(epoch, ': select')
                sparse_social_graph = self.model.delete_social_edge(trustNet)
                social_g = self.dataset.getSocialGraph(sparse_social_graph)
                social_g = social_g.to(self.world.device)
                self.social_drop_out = False
                
            elif epoch % (m+n) == n+1:
                print(epoch, ': random')
                
                num_social = trustNet.shape[0]
                select_num = int(self.world.keep_prob * trustNet.shape[0])
                selected_rows = np.random.choice(num_social, select_num, replace=False)
                sparse_social_graph = trustNet[selected_rows]
                social_g = self.dataset.getSocialGraph(sparse_social_graph)
                social_g = social_g.to(self.world.device)
                '''
                social_g = self.dataset.getSocialGraph(trustNet)
                social_g = social_g.to(self.world.device)
                '''
                self.social_drop_out = False
            
            if epoch < self.pretrain_epoch:
                for (batch_i,
                    (batch_users)) \
                    in enumerate(utils.minibatch(users, batch_size=self.batch_size)):
                    self.opt.zero_grad()
                    
                    batch_users = batch_users.to(self.world.device)
                    social_emb = self.model.getSocialEmbedding(batch_users, social_g, self.social_drop_out)
                    his = self.model.getCondition(batch_users)
                    
                    labels, weights = self.scheduler.sample(social_emb.shape[0], self.world.device)
                    # sgm loss
                    diff_loss = self.diffuse_loss(self.model, social_emb, his, labels)

                    if self.stepscheduler == 'lossaware':
                        self.scheduler.update_with_all_losses(
                        labels, diff_loss.detach()
                    )
                    diff_loss = torch.mean(diff_loss)
                    diff_loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.world.config['grad_norm'])
                    self.opt.step()
                    aver_diff_loss += diff_loss.mean().detach().cpu().item() / total_batch
                    
            with torch.no_grad():
                user_social_list = torch.arange(self.model.num_users)
                socialEmbs = []
                for (batch_i,
                 (batch_users)) \
                in enumerate(utils.minibatch(user_social_list, batch_size=self.batch_size)):
                    batch_users = batch_users.to(self.world.device)
                    social_emb = self.model.getSocialEmbedding(batch_users, social_g, self.social_drop_out)
                    his = self.model.getCondition(batch_users)
                    sample, n = self.sampling_fn(self.model, social_emb, his)
                    socialEmbs.append(sample)
                    
                socialEmbs = torch.cat(socialEmbs, dim=0)
                socialEmbs = socialEmbs.cpu()
                
                socialEmbs1 = []
                for (batch_i,
                 (batch_users)) \
                in enumerate(utils.minibatch(user_social_list, batch_size=self.batch_size)):
                    batch_users = batch_users.to(self.world.device)
                    social_emb = self.model.getSocialEmbedding(batch_users, social_g, self.social_drop_out)
                    his = self.model.getCondition(batch_users)
                    sample, n = self.sampling_fn(self.model, social_emb, his)
                    socialEmbs1.append(sample)
                    
                socialEmbs1 = torch.cat(socialEmbs1, dim=0)
                socialEmbs1 = socialEmbs1.cpu()
                
            for (batch_i,
                 (batch_users, batch_pos, batch_neg)) \
                in enumerate(utils.minibatch(users, posItems, negItems, batch_size=self.batch_size)):
                
                users_social_diff = socialEmbs[[batch_users.long()]].to(self.world.device)
                users_social_diff1 = socialEmbs1[[batch_users.long()]].to(self.world.device)
                
                self.opt.zero_grad()
                batch_users = batch_users.to(self.world.device)
                batch_pos = batch_pos.to(self.world.device)
                batch_neg = batch_neg.to(self.world.device)
                (users_emb, pos_emb, neg_emb,
                userEmb0,  posEmb0, negEmb0, 
                userEmbSocial0) = self.model.getEmbedding(batch_users, batch_pos, batch_neg)
                social_emb = self.model.getSocialEmbedding(batch_users, social_g, self.social_drop_out)
                
                his = self.model.getCondition(batch_users)
                labels, weights = self.scheduler.sample(social_emb.shape[0], self.world.device)
                #  # sgm loss
                diff_loss = self.diffuse_loss(self.model, social_emb, his, labels)
                if self.stepscheduler == 'lossaware':
                        self.scheduler.update_with_all_losses(
                        labels, diff_loss.detach()
                    )
                diff_loss = torch.mean(diff_loss)
                
                ssl_loss, loss_social_ssl = self.model.ssl_loss(users_emb, users_social_diff, users_social_diff1)
                ssl_loss = ssl_loss * self.ssl_wd
                loss_social_ssl = loss_social_ssl * 0.005  
                pos_scores = torch.mul(users_emb, pos_emb)
                pos_scores = torch.sum(pos_scores, dim=1)
                neg_scores = torch.mul(users_emb, neg_emb)
                neg_scores = torch.sum(neg_scores, dim=1)
                
                bpr_loss = torch.mean(torch.nn.functional.softplus(neg_scores - pos_scores))
                reg_loss = (1/2)*(userEmb0.norm(2).pow(2) + 
                                    posEmb0.norm(2).pow(2)  +
                                    negEmb0.norm(2).pow(2)
                                    )/float(len(users))

                reg_loss_social = (1/2)* (userEmbSocial0.norm(2).pow(2))/float(len(users))
                reg_loss_social = reg_loss_social* self.weight_decay_social
                
                reg_loss = reg_loss* self.weight_decay
                loss = bpr_loss + ssl_loss + reg_loss + reg_loss_social + loss_social_ssl + 0.6* diff_loss
                loss.backward()

                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.world.config['grad_norm'])

                self.opt.step()
                
                aver_diff_loss += diff_loss.mean().detach().cpu().item() / total_batch
                aver_ssl_loss += ssl_loss.detach().cpu().item() / total_batch
                aver_bpr_loss += bpr_loss.detach().cpu().item() / total_batch
                
            
            
            
            duration = time.time()-self.start_time
            minutes = int(duration / 60)
            seconds = int(duration % 60)
            print(f"Epoch: {epoch}, BPR Loss: {aver_bpr_loss}, DiffLoss: {aver_diff_loss}, sslLoss: {aver_ssl_loss} ")
            print(f"{minutes:02d}:{seconds:02d}")    
            print('Max: ', torch.cuda.max_memory_allocated())
            print('Memory: ', torch.cuda.max_memory_allocated())
            
            
            if epoch % 5 == 0 or epoch==self.epoches or (epoch > 700 and epoch % 5 == 0):  # --------------
                print("TEST")
                if self.world.args.test_mode == 1:
                    self.test_mode1()
                '''    
                elif self.world.args.test_mode == 2:
                    self.test_mode2()'''
                duration = time.time()-self.start_time
                minutes = int(duration / 60)
                seconds = int(duration % 60)
                print(f"{minutes:02d}:{seconds:02d}")
                print('inference')
                print('Max: ', torch.cuda.max_memory_allocated())
                print('Memory: ', torch.cuda.max_memory_allocated())
            
        self.save()    
    
    def save(self):
        
        combined_dict = {**vars(self.world.args), **self.best}
        
        path = 'log/jssl_' + self.world.dataset+ str(datetime.datetime.now().strftime("%d-%H%M")) + '_' + str(self.world.args.ui_n_layers)+ str(self.world.args.social_n_layers) + \
            str(self.world.args.condition_n_layers)+ '_' +str(self.world.args.num_scale) + str(self.best['recall'][0])+str(self.best['ndcg'][0]) + \
                '.txt'
        with open(path, 'w') as f:
            for i in combined_dict:
                f.write(i + ":" + str(combined_dict[i]) + '\n')
        print('saved')
    
    def test_mode1(self):
        print('Over all items')
        testDict = self.dataset.testDict
        u_batch_size = self.world.config['test_batch_size']
        self.model.eval()

        max_K = max(self.world.topks)
        results = {'precision': np.zeros(len(self.world.topks)),
                    'recall': np.zeros(len(self.world.topks)),
                    'ndcg': np.zeros(len(self.world.topks))}
        
        #self.ema.copy_to(self.model.parameters())
        sampling_fn = sampling.get_sampling_fn(self.world,self.sde)
        
        social_g = self.dataset.getSocialGraph(self.dataset.trustNet)
        social_g = social_g.to(self.world.device)
        
        with torch.no_grad():
            users = list(testDict.keys())
            try:
                assert u_batch_size <= len(users) / 10
            except AssertionError:
                print(f"test_u_batch_size is too big for this dataset, try a small one {len(users) // 10}")
            users_list = []
            rating_list = []
            groundTrue_list = []
            total_batch = len(users) // u_batch_size + 1
            
            for batch_users in utils.minibatch(users, batch_size=u_batch_size):
                
                allPos = self.dataset.getUserPosItems(batch_users)
                groundTrue = [testDict[u] for u in batch_users]
                
                batch_users_gpu = torch.Tensor(batch_users).long()
                batch_users_gpu = batch_users_gpu.to(self.world.device)
                users_emb_i, _, all_items = self.model.getUsersEmbedding(batch_users_gpu, social_g)
                '''
                # implement sgm
                his = self.model.getCondition(batch_users)
                sample, n = sampling_fn(self.model, users_emb_s, his)
                '''
                
                rating = self.model.f(torch.matmul(users_emb_i, all_items.t()))
                

                exclude_index = []
                exclude_items = []
                for range_i, items in enumerate(allPos):
                    exclude_index.extend([range_i] * len(items))
                    exclude_items.extend(items)
                    
                rating[exclude_index, exclude_items] = -(1<<10)
                
                _, rating_K = torch.topk(rating, k=max_K)
                # rating = rating.cpu().numpy()
                del rating
                users_list.append(batch_users)
                rating_list.append(rating_K.cpu())
                groundTrue_list.append(groundTrue)

            assert total_batch == len(users_list)
            X = zip(rating_list, groundTrue_list)

            pre_results = []
            for x in X:
                pre_results.append(utils.test_one_batch(x, self.world.topks))
            # scale = float(u_batch_size/len(users))

            for result in pre_results:
                results['recall'] += result['recall']
                results['precision'] += result['precision']
                results['ndcg'] += result['ndcg']
            results['recall'] /= float(len(users))
            results['precision'] /= float(len(users))
            results['ndcg'] /= float(len(users))   

        print(f"recall:   \t{results['recall']},\nprecision:\t{results['precision']},\nndcg:     \t{results['ndcg']}")
        
            
        self.update_best(results)
        
        print(f"Best:\n recall: {self.best['recall']},\n precision: {self.best['precision']},\n ndcg{self.best['ndcg']}")
            
        return results  
        
    def update_best(self, results):
        for key in results:
            for i in range(len(results[key])):
                if results[key][i] > self.best[key][i]:
                    self.best[key][i] = float('%.6f' % results[key][i])
    
    def test_mode2(self):
        testDict = self.dataset.testDict
        negDict = self.dataset.get_neg_sample_test(100)
        u_batch_size = self.world.config['test_batch_size']
        self.model.eval()
        max_K = max(self.world.topks)
        results = {'precision': np.zeros(len(self.world.topks)),
                    'recall': np.zeros(len(self.world.topks)),
                    'ndcg': np.zeros(len(self.world.topks))}
        
        sampling_fn = sampling.get_sampling_fn(self.world,self.sde)
        
        social_g = self.dataset.getSocialGraph(self.dataset.trustNet)
        social_g = social_g.to(self.world.device)
        
        with torch.no_grad():
            users = list(testDict.keys())
            try:
                assert u_batch_size <= len(users) / 10
            except AssertionError:
                print(f"test_u_batch_size is too big for this dataset, try a small one {len(users) // 10}")
            users_list = []
            rating_list = []
            groundTrue_list = []
            neg_list = []
            total_batch = len(users) // u_batch_size + 1
            
            for batch_users in utils.minibatch(users, batch_size=u_batch_size):
                
                allPos = self.dataset.getUserPosItems(batch_users)
                groundTrue = [testDict[u] for u in batch_users]
                neg_sample = [negDict[u] for u in batch_users]
                
                batch_users_gpu = torch.Tensor(batch_users).long()
                batch_users_gpu = batch_users_gpu.to(self.world.device)
                
                users_emb_i, users_emb_s, all_items = self.model.getUsersEmbedding(batch_users_gpu, social_g)
                # print(users_emb_i.shape)
                # print(users_emb_s.shape)  
                # implement sgm
                his = self.model.getCondition(batch_users)
                sample, n = sampling_fn(self.model,users_emb_s, his)
                users_emb_f = self.model.fuse(users_emb_i, sample)
                rating = self.model.f(torch.matmul(users_emb_f, all_items.t()))
                del batch_users_gpu
                del users_emb_i
                del users_emb_s
                del all_items
                del users_emb_f
                rating = rating.cpu().numpy()
                mask = np.zeros_like(rating)
                
                for range_i, items in enumerate(groundTrue):
                    mask[range_i, items] = 1
                for range_i, items in enumerate(neg_sample):
                    mask[range_i, items] = 1
                '''
                print(groundTrue[0][0])
                print(neg_sample[0][1])
                print(mask[0, groundTrue[0][0]])
                print(mask[0, groundTrue[0][0]+1])
                print(mask[0, neg_sample[0][1]])
                print(mask[0, neg_sample[0][1]+1])
                test = groundTrue[0][0]
                if test in groundTrue[0] or test in neg_sample[0]:
                    print('correct')
                else:
                    print('wrong')
                '''
                rating = np.where(mask == 1, rating, -np.inf)
                
                _, rating_K = torch.topk(torch.from_numpy(rating), k=max_K)
                
                del rating
                users_list.append(batch_users)
                rating_list.append(rating_K.cpu())
                groundTrue_list.append(groundTrue)

            assert total_batch == len(users_list)
            X = zip(rating_list, groundTrue_list)

            pre_results = []
            for x in X:
                pre_results.append(utils.test_one_batch(x, self.world.topks))
            scale = float(u_batch_size/len(users))

            for result in pre_results:
                results['recall'] += result['recall']
                results['precision'] += result['precision']
                results['ndcg'] += result['ndcg']
            results['recall'] /= float(len(users))
            results['precision'] /= float(len(users))
            results['ndcg'] /= float(len(users))   

        print(f"recall:   \t{results['recall']},\nprecision:\t{results['precision']},\nndcg:     \t{results['ndcg']}")
        
        if results['recall'][0] > self.best['recall'][0]:
            self.best = results
           
        if results['ndcg'][0] > self.ndcg_best['ndcg'][0]:
            self.ndcg_best = results
            
        if results['recall'][0] < self.best['recall'][0]:
            print(f"Best recall 1: recall: {self.best['recall']}, precision: {self.best['precision']}, ndcg{self.best['ndcg']}")
            print()
        
        if self.best['ndcg'][0] < self.ndcg_best['ndcg'][0]:
            print(f"Best ndcg1: recall: {self.ndcg_best['recall']}, precision: {self.ndcg_best['precision']}, ndcg{self.ndcg_best['ndcg']}")
            print()
        return results  
                