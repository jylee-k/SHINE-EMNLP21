import numpy as np
import pickle as pkl
import json
import torch
import torch.nn.functional as F
import torch.optim as optim
import time
import math
from sklearn import metrics
from utils import fetch_to_tensor
from model import SHINE

from sklearn.metrics import confusion_matrix, classification_report, f1_score
from sklearn.model_selection import train_test_split
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt

class Trainer(object):
    def __init__(self, params):
        self.dataset_name = params.dataset
        self.max_epoch = params.max_epoch
        self.save_path = params.save_path
        self.device = params.device
        self.hidden_size = params.hidden_size
        self.lr = params.lr
        self.weight_decay = params.weight_decay
        self.concat_word_emb = params.concat_word_emb
        self.concat_phrase_emb = params.concat_phrase_emb
        self.type_names = params.type_num_node # params.type_num_node = ['query', 'tag', 'word', 'phrase_tag', 'phrase', 'entity']
        self.data_path = params.data_path

        self.adj_dict, self.features_dict, self.train_idx, self.valid_idx, self.test_idx, self.labels, self.nums_node = self.load_data()
        self.label_num = len(set(self.labels))
        self.labels = torch.tensor(self.labels).type(torch.LongTensor).to(self.device)
        self.out_features_dim = [self.label_num, self.hidden_size, self.hidden_size, self.hidden_size, self.hidden_size, self.hidden_size, self.hidden_size]
        print(self.out_features_dim)
        in_fea_final = self.out_features_dim[1] + self.out_features_dim[2] + self.out_features_dim[3] + self.out_features_dim[4] #+ self.out_features_dim[5]
        print('in_fea_final: ', in_fea_final)
        self.in_features_dim = [0, self.nums_node[1], self.nums_node[2], self.nums_node[4], self.nums_node[5], self.nums_node[-1], in_fea_final]

        if self.concat_word_emb: self.in_features_dim[-1] += self.features_dict['word_emb'].shape[-1]
        if self.concat_phrase_emb: self.in_features_dim[-1] += self.features_dict['phrase_emb'].shape[-1]
        self.model = SHINE(self.adj_dict, self.features_dict, self.in_features_dim, self.out_features_dim, params)
        self.model = self.model.to(self.device)
        total_trainable_params = sum(p.numel() for p in self.model.parameters())
        print(f'{total_trainable_params:,} training parameters.')
        self.parameters = filter(lambda p: p.requires_grad, self.model.parameters())
        self.optim = optim.Adam([{'params': self.model.parameters()},
                                 {'params': self.model.GCNs[0].parameters()},
                                 {'params': self.model.GCNs[2].parameters()},
                                 {'params': self.model.GCNs_2[2].parameters()},
                                 {'params': self.model.GCNs[1].parameters()},
                                 {'params': self.model.GCNs_2[0].parameters()},
                                 {'params': self.model.GCNs_2[1].parameters()}], lr=self.lr, weight_decay=self.weight_decay)

    def train(self):
        # global_best_acc = 0
        # global_best_f1 = 0
        # global_best_epoch = 0
        best_test_acc = 0
        best_test_f1 = 0
        best_test_precision = 0
        best_test_recall = 0
        best_test_loss =1e9
        best_valid_epoch = 0
        best_valid_macro_f1 = 0
        best_valid_micro_f1 = 0
        best_valid_acc = 0
        best_valid_loss = 1e9
        acc_valid = 0
        loss_valid = 0
        f1_valid = 0
        acc_test=0
        loss_test = 0
        f1_test = 0
        # best_acc = 0
        # best_f1 = 0
    
        for i in range(1, self.max_epoch + 1):
            t = time.time()
            output = self.model(i)
            train_scores = output[self.train_idx]
            train_labels = self.labels[self.train_idx]
            loss = F.cross_entropy(train_scores, train_labels)
            self.optim.zero_grad()
            loss.backward()
            self.optim.step()
            loss = loss.item()
            acc = torch.eq(torch.argmax(train_scores, dim=-1), train_labels).float().mean().item()
            print('Epoch {}  loss: {:.4f} acc: {:.4f} time{:.4f}'.format(i, loss, acc,time.time()-t))
            acc_valid, loss_valid, macro_f1_valid, micro_f1_valid, acc_test, loss_test, macro_f1_test, macro_precision_test, macro_recall_test, y_pred, y_true = self.test(i) 
            # if acc_test > global_best_acc:
            #     global_best_acc = acc_test
            #     global_best_macro_f1 = macro_f1_test
            #     global_best_epoch = i
                #self.cm(y_true.cpu().numpy(), y_pred.cpu().numpy())
            if loss_valid < best_valid_loss:
                best_valid_loss = loss_valid
                best_valid_acc = acc_valid
                best_valid_macro_f1 = macro_f1_valid 
                best_valid_micro_f1 = micro_f1_valid
                best_valid_epoch = i
            if macro_f1_test > best_test_f1:
                # best_test_loss = loss_test
                best_test_acc = acc_test 
                best_test_f1 = macro_f1_test
                best_test_precision = macro_precision_test
                best_test_recall = macro_recall_test
                best_test_epoch = i
                # take global best test epoch
                # best_acc = global_best_acc
                # best_f1 = global_best_f1
                # best_epoch = global_best_epoch
            if i%50==0:
                print('Best validation: VALID ACC', best_valid_acc, ' MACRO F1', best_valid_macro_f1, 'MICRO F1', best_valid_micro_f1, 'EPOCH', best_valid_epoch) 
                print('Best test: TEST ACC', best_test_acc, 'TEST F1', best_test_f1, 'TEST PRECISION', best_test_precision, 'TEST RECALL', best_test_recall, 'EPOCH', best_test_epoch)
                # print('GLOBAL: TEST ACC', global_best_acc, 'TEST F1', global_best_f1, 'EPOCH', global_best_epoch)
        return best_test_f1, best_test_precision, best_test_recall

    def test(self, epoch):
        t = time.time()
        self.model.training = False
        output = self.model(0)
        with torch.no_grad():
            train_scores = output[self.train_idx]
            train_labels = self.labels[self.train_idx]
            loss_train = F.cross_entropy(train_scores, train_labels).item()
            acc_train = torch.eq(torch.argmax(train_scores, dim=-1), train_labels).float().mean().item()
            valid_scores = output[self.valid_idx]
            valid_labels = self.labels[self.valid_idx]
            loss_valid = F.cross_entropy(valid_scores, valid_labels).item()
            acc_valid = torch.eq(torch.argmax(valid_scores, dim=-1), valid_labels).float().mean().item()
            macro_f1_valid = metrics.f1_score(valid_labels.detach().cpu().numpy(),torch.argmax(valid_scores,-1).detach().cpu().numpy(),average='macro')
            micro_f1_valid = metrics.f1_score(valid_labels.detach().cpu().numpy(),torch.argmax(valid_scores,-1).detach().cpu().numpy(),average='micro')
            print('Epoch {}  loss: {:.4f} acc: {:.4f}'.format(epoch, loss_train, acc_train),
                'Valid  loss: {:.4f}  acc: {:.4f}  macro_f1: {:.4f}  micro_f1: {:.4f}'.format(loss_valid, acc_valid, macro_f1_valid, micro_f1_valid))
            test_scores = output[self.test_idx]
            test_labels = self.labels[self.test_idx]
            y_pred = torch.argmax(test_scores, dim=-1)
            loss_test = F.cross_entropy(test_scores, test_labels).item()
            acc_test = torch.eq(torch.argmax(test_scores, dim=-1), test_labels).float().mean().item()
            macro_f1_test = metrics.f1_score(test_labels.detach().cpu().numpy(),torch.argmax(test_scores,-1).detach().cpu().numpy(),average='macro')
            macro_precision_test = metrics.precision_score(test_labels.detach().cpu().numpy(),torch.argmax(test_scores,-1).detach().cpu().numpy(),average='macro', zero_division=0)
            macro_recall_test = metrics.recall_score(test_labels.detach().cpu().numpy(),torch.argmax(test_scores,-1).detach().cpu().numpy(),average='macro')
            # micro_f1_test = metrics.f1_score(test_labels.detach().cpu().numpy(),torch.argmax(test_scores,-1).detach().cpu().numpy(),average='micro')
            print('Test  loss: {:.4f} acc: {:.4f} macro f1: {:.4f} macro precision: {:.4f} macro recall: {:.4f} time: {:.4f}'.format(loss_test, acc_test, macro_f1_test, macro_precision_test, macro_recall_test, time.time() - t))
        self.model.training = True
        return acc_valid, loss_valid, macro_f1_valid, micro_f1_valid, acc_test, loss_test, macro_f1_test, macro_precision_test, macro_recall_test, y_pred, test_labels
    
    def cm(self, y_true, y_pred):
        cf_matrix = confusion_matrix(y_true, y_pred)
        print(cf_matrix)
        print(classification_report(y_true, y_pred))
        print('macro f1: ',f1_score(y_true, y_pred, average='macro'))
        print('micro f1: ',f1_score(y_true, y_pred, average='micro'))
        # Creating a dataframe for a array-formatted Confusion matrix,so it will be easy for plotting.
        cm_df = pd.DataFrame(cf_matrix,
                     index = ['0','1','2', '3'], 
                     columns = ['0','1','2','3'])
        plt.figure(figsize = (12,7))
        sn.heatmap(cm_df, annot=True, fmt='g')
        plt.title('Confusion Matrix')
        plt.ylabel('Actal Values')
        plt.xlabel('Predicted Values')
        plt.savefig('output.png')

    def load_data(self):
        start = time.time()
        adj_dict = {}
        feature_dict = {}
        nums_node = [] # list of the number of nodes in the adjacency matrices
        for i in range(1, len(self.type_names)): # self.type_names = ['query', 'tag', 'word', 'phrase_tag', 'phrase', 'entity']
            # opens '01': adj_query2tag, '02': adj_query2word, '03': adj_query2phrase_tag '04': adj_query2phrase, '05': adj_query2entity,
            adj_dict[str(0) + str(i)] = pkl.load(
                    open(self.data_path + './adj_{}2{}.pkl'.format(self.type_names[0], self.type_names[i]), 'rb'))
            if i == 1 or i == 3: # add number of nodes for tags 
                nums_node.append(adj_dict[str(0) + str(i)].shape[0]) # 
                print(str(0) + str(i), adj_dict[str(0) + str(i)].shape)
            if i != 5: 
                adj_dict[str(i) + str(i)] = pkl.load( # opens '11': adj_tag, '22': adj_word, '44': adj_phrase, '33': adj_phrase_tag
                    open(self.data_path + './adj_{}.pkl'.format(self.type_names[i]), 'rb'))
                nums_node.append(adj_dict[str(i) + str(i)].shape[0])
                print(str(i) + str(i), adj_dict[str(i) + str(i)].shape)
            if i == 5: # 'entity'
                feature_dict[str(i)] = pkl.load( # opens '5': entity_emb
                    open(self.data_path + './{}_emb.pkl'.format(self.type_names[i]), 'rb'))
                nums_node.append(feature_dict[str(i)].shape[0])
                print(i, feature_dict[str(i)].shape[0])
                nums_node.append(feature_dict[str(i)].shape[1])
                print(i, feature_dict[str(i)].shape[1])
            else:
                feature_dict[str(i)] = np.eye(nums_node[i], dtype=np.float64)
        # nums_node = [11233455]
        feature_dict['word_emb'] = torch.tensor(pkl.load(
            open(self.data_path + './word_emb.pkl', 'rb')), dtype=torch.float).to(self.device)
        feature_dict['phrase_emb'] = torch.tensor(pkl.load(
            open(self.data_path + './phrase_emb.pkl', 'rb')), dtype=torch.float).to(self.device)
        ent_emb=feature_dict['5']
        ent_emb_normed = ent_emb / np.sqrt(np.square(ent_emb).sum(-1, keepdims=True))
        adj_dict['55'] = np.matmul(ent_emb_normed, ent_emb_normed.transpose())
        adj_dict['55'] = adj_dict['55'] * np.float32(adj_dict['55'] > 0)
        adj_dict['22'] = np.array(adj_dict['22'].toarray())
        adj_dict['44'] = np.array(adj_dict['44'].toarray())
        adj_dict['02'] = np.array(adj_dict['02'].toarray())
        adj_dict['05'] = np.array(adj_dict['05'].toarray())
        adj_dict['04'] = np.array(adj_dict['04'].toarray())
        
        
        adj = {}
        feature = {}
        for i in adj_dict.keys():
            adj[i] = fetch_to_tensor(adj_dict, i, self.device)
        for i in feature_dict.keys():
            feature[i] = fetch_to_tensor(feature_dict, i, self.device)
        
        train_set = json.load(open(self.data_path + './train_idx.json'))
        test_idx = json.load(open(self.data_path + './test_idx.json'))
        # labels = json.load(open(self.data_path + './labels.json'))
        # labels = json.load(open(self.data_path + './coarse_labels.json'))
        labels = json.load(open(self.data_path + './fine_labels.json'))
        
        train_idx, valid_idx = train_test_split(train_set, test_size=0.2, random_state=42)
        
        # data_index = train_set + test_set
        # Sumofquery = len(data_index)
        # label_dict = {}
        # for i in set(labels):
        #     label_dict[i] = []
        # for j, label in enumerate(labels):
        #     label_dict[label].append(j)
        # len_train_idx = len(label_dict) * 20 
        # train_list = []
        # valid_list = []
        # byclass = True if self.dataset_name=='ohsu_title' else False 
        # if not byclass:
        #     for i in label_dict.items():
        #         train_list.append(20)
        # else:
        #     ratio = len_train_idx / Sumofquery
        #     residue = []
        #     max_supp = len_train_idx
        #     for val in label_dict.values():
        #         num = math.modf(len(val) * ratio)
        #         residue.append(num[0])
        #         train_list.append(int(num[1]))
        #         max_supp -= int(num[1])
        #     sorted_list = sorted(range(len(residue)), key= lambda i : residue[i], reverse=True)[ : max_supp]
        #     for i, val in enumerate(train_list):
        #         if i in sorted_list:
        #             train_list[i] += 1
        # valid_list = train_list
        # train_idx = []
        # valid_idx = []
        # test_idx = []
        # for i, idxs in enumerate(label_dict.values()):
        #     np.random.shuffle(idxs)
        #     for j, idx in enumerate(idxs):
        #         if j < train_list[i]:
        #             train_idx.append(idx)
        #         elif j >= train_list[i] and j < train_list[i] + valid_list[i]:
        #             valid_idx.append(idx)
        #         else:
        #             test_idx.append(idx)
        print("train samples: ", len(train_idx))
        print(train_idx[:10])
        print("valid samples: ", len(valid_idx))
        print(valid_idx[:10])
        print("test samples: ", len(test_idx)) 
        print(test_idx[:10])
        print('data process time: {}'.format(time.time()-start))
        return adj, feature, train_idx, valid_idx, test_idx, labels, nums_node

    def save(self, path=None):
        if not path:
            path = self.save_path
        torch.save(self.model.state_dict(), path + './{}/save_model_new'.format(self.dataset_name))

    def load(self):
        self.model.load_state_dict(torch.load(self.save_path))
