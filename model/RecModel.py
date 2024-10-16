import torch
from torch import nn
import math
import numpy as np

from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
import torch.nn.functional as F

class socialRecModel(nn.Module):
    def __init__(self, args, 
                 num_users, num_items,
                 interGraph):
    
        super(socialRecModel, self).__init__()
        print("Initialize Rec model")
        
        self.args = args
        self.num_users = num_users
        self.num_items = num_items
        self.latent_dim = self.args.emb_dim
        
        self.embedding_user = nn.Embedding(
            num_embeddings=self.num_users, embedding_dim=self.latent_dim)
        self.embedding_item = nn.Embedding(
            num_embeddings=self.num_items, embedding_dim=self.latent_dim)
        
        self.embedding_user_social = torch.nn.Embedding(
            num_embeddings=self.num_users, embedding_dim=self.latent_dim) 
        
        self.f = F.softmax
        
        self.__init_weight()
        
        self.interGraph = interGraph
        
        self.inter_n_layers = self.args.ui_n_layers
        self.social_n_layers = self.args.social_n_layers
        self.condition_n_layers = self.args.condition_n_layers
        
        self.fuse_module = nn.Sequential(
            nn.Linear(self.latent_dim*2, self.latent_dim*2),
            nn.LeakyReLU(), 
            nn.Linear(self.latent_dim*2, self.latent_dim)
            )
        
        self.latent_dim = self.args.emb_dim
        self.diffuser = nn.Sequential(
            nn.Linear(self.latent_dim*3, self.latent_dim*3),
            nn.LeakyReLU(),
            nn.Linear(self.latent_dim*3, self.latent_dim))  # sturcture, initialize parameter
        self.step_mlp = nn.Sequential(
            nn.Linear(self.latent_dim, self.latent_dim),
        )
        
        self.__init__weight()
        
        self.ui_dropout = args.ui_dropout
        self.keep_prob_ui = args.keep_prob_ui
        
        
        self.social_encoder = self.args.social_encoder
        if self.social_encoder == 'GCN':
            self.gcnLayers = nn.Sequential(*[GCNLayer(self.latent_dim, self.latent_dim, False) for i in range(self.social_n_layers)])
        elif self.social_encoder == 'GAT':
            self.gatLayers = nn.Sequential(*[GATLayer(self.latent_dim, self.latent_dim, False) for i in range(self.social_n_layers)])
            
        self.ui_encoder = self.args.ui_encoder
        if self.ui_encoder == 'GCN':
            self.gcnLayersUI = nn.Sequential(*[GCNLayer(self.latent_dim, self.latent_dim, False) for i in range(self.inter_n_layers)])
        
        self.tau = self.args.tau

        self.mlp_social2social_layer1 = torch.nn.Linear(self.latent_dim, self.latent_dim)
        self.mlp_social2social_layer2 = torch.nn.Linear(self.latent_dim, self.latent_dim)
        self.mlp_social2social_act = nn.functional.elu
        
        self.mlp_prefer2prefer_layer1 = torch.nn.Linear(self.latent_dim, self.latent_dim)
        self.mlp_prefer2prefer_layer2 = torch.nn.Linear(self.latent_dim, self.latent_dim)
        self.mlp_prefer2prefer_act = nn.functional.elu
        
    def __init_weight(self):
        nn.init.normal_(self.embedding_user.weight, std=0.1)
        nn.init.normal_(self.embedding_item.weight, std=0.1)
        nn.init.normal_(self.embedding_user_social.weight, std=0.1)  # ---------------
        
        print('Use NORMAL distribution initilizer for embeddings of users and items')
    
    def __init__weight(self):
        for layer in self.diffuser:
            if isinstance(layer, nn.Linear):
                size = layer.weight.size()
                std = np.sqrt(2.0 / (size[0] + size[1]))
                layer.weight.data.normal_(0.0, std)
                layer.bias.data.normal_(0.0, 0.001)
                
    def forward(self, x, t, c = None):
        t = timestep_embedding(t, self.latent_dim, x.device)
        t = self.step_mlp(t)  # time embedding 
        # c = self.c_projection(c)
        model_input = torch.cat([x, c, t], dim=1)
        model_output = self.diffuser(model_input)
        del model_input
        del t
        return model_output
    
    def __dropout(self, keep_prob):
        size = self.interGraph.size()
        index = self.interGraph.indices().t()
        values = self.interGraph.values()
        random_index = torch.rand(len(values)) + keep_prob
        random_index = random_index.int().bool()
        index = index[random_index]
        values = values[random_index]/keep_prob
        g = torch.sparse.FloatTensor(index.t(), values, size)
        return g
    
    def __dropout_social(self, g, keep_prob):
        size = g.size()
        index = g.indices().t()
        values = g.values()
        random_index = torch.rand(len(values)) + keep_prob
        random_index = random_index.int().bool()
        index = index[random_index]
        values = values[random_index]/keep_prob
        g = torch.sparse.FloatTensor(index.t(), values, size)
        return g
    
    
    def InterEncode(self):
        if self.ui_encoder == 'LCN':
            
            users_emb = self.embedding_user.weight
            items_emb = self.embedding_item.weight
            all_emb = torch.cat([users_emb, items_emb])
            embs = [all_emb]
            
            if self.training and self.ui_dropout:
                g_droped = self.__dropout(self.keep_prob_ui)
            else:
                g_droped = self.interGraph  
                
            for layer in range(self.inter_n_layers):
                all_emb = torch.sparse.mm(g_droped, all_emb)
                embs.append(all_emb)
            
            embs = torch.stack(embs, dim=1)
            light_out = torch.mean(embs, dim=1)
            users, items = torch.split(light_out, [self.num_users, self.num_items])
            
        elif self.ui_encoder == 'GCN':
            users_emb = self.embedding_user.weight
            items_emb = self.embedding_item.weight
            all_emb = torch.cat([users_emb, items_emb])
            embs = [all_emb]
            emb = embs[-1]
            
            
            if self.training and self.ui_dropout:
                g_droped = self.__dropout(self.keep_prob_ui)
            else:
                g_droped = self.interGraph  
                
            for gcn in self.gcnLayersUI:
                emb = gcn(emb, g_droped)  
                embs.append(emb)
                emb = F.leaky_relu(emb)  
            
            light_out = embs[-1]
            users, items = torch.split(light_out, [self.num_users, self.num_items])
                
        return users, items
    
    def SocialEncode(self, g, social_drop_out=False):
        if self.social_encoder == 'LCN':
            # seperate embedding layer  
            users_embs = [self.embedding_user_social.weight]
            
            if self.training and social_drop_out:
                g_droped = self.__dropout_social(g, self.args.keep_prob)
            else:
                g_droped = g
            
            for layer in range(self.social_n_layers):
                emb = torch.sparse.mm(g, users_embs[-1])  # sparse g 
                users_embs.append(emb)
            users_embs = torch.stack(users_embs, dim = 1)
            users = torch.mean(users_embs, dim=1)  # pooling 
            # users = torch.sum(users_embs, dim=1)  # pooling 
            # users = users_embs.sum(0)
        elif self.social_encoder == 'GCN':
            users_embs = [self.embedding_user_social.weight]
            emb = self.embedding_user_social.weight
            
            if self.training and social_drop_out:
                g_droped = self.__dropout_social(g, self.args.keep_prob)
            # elif self.training:
            #     g_droped = self.__dropout_social(g, 0.95)
            else:
                g_droped = g
            
            for gcn in self.gcnLayers:

                emb = gcn(emb, g_droped)  
                users_embs.append(emb)
                emb = F.leaky_relu(emb)  
            users = users_embs[-1]
            
        elif self.social_encoder == 'GAT':
            users_embs = [self.embedding_user_social.weight]
            for gat in self.gatLayers:
                emb = gat(users_embs[-1],g)
                
                users_embs.append(emb)
            
            users_embs = torch.stack(users_embs, dim = 1)
            users = torch.mean(users_embs, dim=1)
            
        return users
    
    def getUsersEmbedding(self, users, g):
        all_users, all_items = self.InterEncode()
        social_users = self.SocialEncode(g)

        users_emb_i = all_users[users.long()]
        users_emb_s = social_users[users.long()]
        del all_users
        return users_emb_i, users_emb_s, all_items
            
    def getSocialEmbedding(self, users, g, social_drop_out= False):
        all_emb = self.SocialEncode(g, social_drop_out)
        social_emb = all_emb[users]
        
        del all_emb
        return social_emb
    
    def getEmbedding2View(self, users):
        all_users, all_items = self.InterEncode()
        users_emb = all_users[users]
        del all_users, all_items
        return users_emb
        
    def getEmbedding(self, users, pos_items, neg_items):
        all_users, all_items = self.InterEncode()
        
        users_emb = all_users[users]
        pos_emb = all_items[pos_items]
        neg_emb = all_items[neg_items]
        
        users_emb_ego = self.embedding_user(users)
        pos_emb_ego = self.embedding_item(pos_items)
        neg_emb_ego = self.embedding_item(neg_items)
        
        users_emb_social_ego = self.embedding_user_social(users)
        
        del all_users, all_items
        return users_emb, pos_emb, neg_emb, \
                users_emb_ego, pos_emb_ego, neg_emb_ego, \
                    users_emb_social_ego
        
    
    """
    users_emb: U-I embedding
    users_social_diff: Social embedding
    """
    def fuse(self, users_emb, users_social_diff):
        model_input = torch.cat([users_emb, users_social_diff], dim=1)
        model_output = self.fuse_module(model_input)
        return model_output
    
    def getCondition(self, user_id):
        users_emb = self.embedding_user.weight
        items_emb = self.embedding_item.weight
        
        all_emb = torch.cat([users_emb, items_emb])
       
        embs = [all_emb]
        g_droped = self.interGraph  # dropout 
        for layer in range(self.condition_n_layers):
            all_emb = torch.sparse.mm(g_droped, all_emb)
            embs.append(all_emb)
        embs = torch.stack(embs, dim=1)

        light_out = torch.mean(embs, dim=1)
        users, items = torch.split(light_out, [self.num_users, self.num_items])
        return users[user_id]
    
    def calc_ssl_sim(self, emb1, emb2, tau, normalization = False):
        # (emb1, emb2) = (F.normalize(emb1, p=2, dim=0), F.normalize(emb2, p=2, dim=0))\
        if normalization:
            emb1 = nn.functional.normalize(emb1, p=2, dim=1, eps=1e-12)
            emb2 = nn.functional.normalize(emb2, p=2, dim=1, eps=1e-12)
        (emb1_t, emb2_t) = (emb1.t(), emb2.t())
        pos_scores_users = torch.exp(torch.div(F.cosine_similarity(emb1, emb2, dim=1, eps=1e-8), tau))  # Sum by row
        
        # denominator cosine_similarity: following codes
        if self.args.interOrIntra == 'inter':
            denominator_scores = torch.mm(emb1, emb2_t)
            norm_emb1 = torch.norm(emb1, dim=-1)
            norm_emb2 = torch.norm(emb2, dim=-1)
            
            norm_emb = torch.mm(norm_emb1.unsqueeze(1), norm_emb2.unsqueeze(1).t())
            denominator_scores1 = torch.exp(torch.div(denominator_scores / norm_emb, tau)).sum(1)  # Sum by row
            denominator_scores2 = torch.exp(torch.div(denominator_scores / norm_emb, tau)).sum(0)  # Sum by column
            # denominator cosine_similarity: above codes
            ssl_loss1 = -torch.mean(torch.log(pos_scores_users / denominator_scores1))
            ssl_loss2 = -torch.mean(torch.log(pos_scores_users / denominator_scores2))
        else:  # interAintra
            denominator_scores = torch.mm(emb1, emb2_t)
            norm_emb1 = torch.norm(emb1, dim=-1)
            norm_emb2 = torch.norm(emb2, dim=-1)
            norm_emb = torch.mm(norm_emb1.unsqueeze(1), norm_emb2.unsqueeze(1).t())
            denominator_scores1 = torch.exp(torch.div(denominator_scores / norm_emb, tau)).sum(1)  # Sum by row
            denominator_scores2 = torch.exp(torch.div(denominator_scores / norm_emb, tau)).sum(0)  # Sum by column
            
            denominator_scores_intraview1 = torch.mm(emb1, emb1_t)
            norm_intra1 = torch.mm(norm_emb1.unsqueeze(1), norm_emb1.unsqueeze(1).t())
            denominator_intra_scores1 = torch.exp(torch.div(denominator_scores_intraview1 / norm_intra1, tau))
            diag1 = torch.diag(denominator_intra_scores1)
            d_diag1 = torch.diag_embed(diag1)
            denominator_intra_scores1 = denominator_intra_scores1 - d_diag1  # here we set the elements on diagonal to be 0.
            intra_denominator_scores1 = denominator_intra_scores1.sum(1)  # Sum by row## .sum(1)

            denominator_scores_intraview2 = torch.mm(emb2, emb2_t)
            norm_intra2 = torch.mm(norm_emb2.unsqueeze(1), norm_emb2.unsqueeze(1).t())
            denominator_intra_scores2 = torch.exp(torch.div(denominator_scores_intraview2 / norm_intra2, tau))
            diag2 = torch.diag(denominator_intra_scores2)
            d_diag2 = torch.diag_embed(diag2)
            denominator_intra_scores2 = denominator_intra_scores2 - d_diag2
            intra_denominator_scores2 = denominator_intra_scores2.sum(1)

            # denominator cosine_similarity: above codes
            ssl_loss1 = -torch.mean(torch.log(pos_scores_users / (denominator_scores1 + intra_denominator_scores1)))
            ssl_loss2 = -torch.mean(torch.log(pos_scores_users / (denominator_scores2 + intra_denominator_scores2)))

        return ssl_loss1 + ssl_loss2
    
    def mlp_social2social(self, input_emb, mlpOrNot= True):
        if mlpOrNot:
            x = self.mlp_social2social_layer1(input_emb)
            x = self.mlp_social2social_act(x)
            y = self.mlp_social2social_layer2(x)
        else:
            y = input_emb
        return y
    
    def mlp_prefer2prefer(self, input_emb, mlpOrNot=True):
        if mlpOrNot:
            x = self.mlp_prefer2prefer_layer1(input_emb)
            x = self.mlp_prefer2prefer_act(x)
            y = self.mlp_prefer2prefer_layer2(x)
        else:
            y = input_emb
        return y
    
    def ssl_loss(self, users_preference1, users_social1, users_social2):
        
        users_social1 = self.mlp_social2social(users_social1)
        users_social2 = self.mlp_social2social(users_social2)
        
        users_preference1 = self.mlp_prefer2prefer(users_preference1)
        # users_preference2 = self.mlp_prefer2prefer(users_preference2)
        
        loss_social = self.calc_ssl_sim(users_preference1, users_social1, self.tau) + self.calc_ssl_sim(users_preference1, users_social2, self.tau)
        
        loss_social_ssl = self.calc_ssl_sim(users_social1, users_social2, self.tau, normalization = True)
        
        return loss_social, loss_social_ssl
    
    def delete_social_edge(self, trustNet):
        
        user_emb, _  = self.InterEncode()
        user_emb = user_emb.detach().cpu()
        '''
        # cosine similarity
        batch_size = 5000
        num_batches = len(trustNet) // batch_size + 1
        score_all = []
        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, len(trustNet))
            batch_data = trustNet[start_idx:end_idx]
            user1_emb = user_emb[batch_data[:, 0]]
            user2_emb = user_emb[batch_data[:, 1]]
            # score = np.diag(torch.matmul(user1_emb, user2_emb.t()))
            score = torch.nn.functional.cosine_similarity(user1_emb, user2_emb) 
            score_all.append(score)
        score_all = np.concatenate(score_all)
        # print(score_all.shape)
        '''
        # inner dot
        score_all = []
        for i in range(len(trustNet)):
            user1_emb = user_emb[trustNet[i,0]]
            user2_emb = user_emb[trustNet[i,1]]
            score = torch.dot(user1_emb, user2_emb)
            score_all.append(score)
        
        trust_dict = {}
        score_dict = {}
        for i in range(trustNet.shape[0]):
            uid1 = trustNet[i,0]
            uid2 = trustNet[i,1]
            if uid1 not in trust_dict:
                trust_dict[uid1] = []
            if uid1 not in score_dict:
                score_dict[uid1] = []
            trust_dict[uid1].append(uid2)
            score_dict[uid1].append([uid2, score_all[i]])
        
        for k, v in trust_dict.items():
            if len(v) < self.args.epsilon:
                continue
            else: 
                keep_num = int(len(v)* self.args.R)
                
            score_sort = sorted(score_dict[k], key=lambda x: x[1],reverse=True)
            social_relation = [sublist[0] for sublist in score_sort[:keep_num]]
            trust_dict[k] = social_relation
        
        user1 = []
        user2 = []
        for k, values in trust_dict.items():
            for v in values:
                user1.append(k)
                user2.append(v)
                
        trustNet = np.column_stack((user1, user2))
        # trustNet = np.concatenate([trustNet, trustNet[:, [1, 0]]], axis=0)  #################################################################
        return trustNet
        
            
            
def timestep_embedding(timesteps, dim, device, max_period=10000):
    half = dim // 2
    seq = torch.arange(start=0, end=half, dtype=torch.float32)
    freqs = torch.exp(
        -math.log(max_period) *  seq/ half
    ).to(device)
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding

class GCNLayer(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GCNLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output
        

class GATLayer(nn.Module):
    """
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    """
    def __init__(self, in_features, out_features, alpha=0.2, dropout=0.95, concat=False):
        super(GATLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Parameter(torch.empty(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.empty(size=(2*out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, h, adj):
        Wh = torch.mm(h, self.W) # h.shape: (N, in_features), Wh.shape: (N, out_features)
        
        e = self._prepare_attentional_mechanism_input(Wh)
        zero_vec = -9e15*torch.ones_like(e)
        
        attention = torch.where(adj.to_dense() > 0, e, zero_vec)
        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.matmul(attention, Wh)
        
        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime
        
    def _prepare_attentional_mechanism_input(self, Wh):
        # Wh.shape (N, out_feature)
        # self.a.shape (2 * out_feature, 1)
        # Wh1&2.shape (N, 1)
        # e.shape (N, N)
        Wh1 = torch.matmul(Wh, self.a[:self.out_features, :])
        Wh2 = torch.matmul(Wh, self.a[self.out_features:, :])
        # broadcast add
        e = Wh1 + Wh2.T
        return self.leakyrelu(e)
    

class DenoiseModel(nn.Module):
    def __init__(self, args):
        super(DenoiseModel, self).__init__()
        
        self.args = args
        self.latent_dim = self.args.emb_dim
        self.diffuser = nn.Sequential(
            nn.Linear(self.latent_dim*3, self.latent_dim*3),
            nn.LeakyReLU(),
            nn.Linear(self.latent_dim*3, self.latent_dim))  # sturcture, initialize parameter
        self.step_mlp = nn.Sequential(
            nn.Linear(self.latent_dim, self.latent_dim),
        )
        self.__init_weight()
        
    def __init_weight(self):
        for layer in self.diffuser:
            if isinstance(layer, nn.Linear):
                size = layer.weight.size()
                std = np.sqrt(2.0 / (size[0] + size[1]))
                layer.weight.data.normal_(0.0, std)
                layer.bias.data.normal_(0.0, 0.001)
                
    def forward(self, x, t, c = None):
        t = timestep_embedding(t, self.latent_dim, x.device)
        t = self.step_mlp(t)  # time embedding ---------------------------------------
        
        #c = self.c_projection(c)
        model_input = torch.cat([x, c, t], dim=1)
        model_output = self.diffuser(model_input)
        del model_input
        del t
        return model_output