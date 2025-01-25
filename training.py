"""
Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved

Author: Dejiao Zhang (dejiaoz@amazon.com)
Date: 02/26/2021
"""
import numpy as np
from sklearn import cluster

from utils.logger import statistics_log
from utils.metric import Confusion
from dataloader.dataloader import unshuffle_loader

import torch.nn as nn
from learner.contrastive_utils import PairConLoss, Attention_loss

import matplotlib.pyplot as plt
from collections import Counter
from plabel_allocator import *
import pandas as pd
from sklearn.manifold import TSNE


class SCCLvTrainer(nn.Module):
    def __init__(self, model, tokenizer, optimizer, optimizer2, train_loader, args, scheduler=None):
        super(SCCLvTrainer, self).__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.optimizer = optimizer
        self.optimizer2 = optimizer2
        self.scheduler = scheduler
        self.train_loader = train_loader
        self.args = args
        self.eta = self.args.eta
        
        self.cluster_loss = nn.KLDivLoss(size_average=False)
        self.ce_loss = nn.CrossEntropyLoss()
        self.contrast_loss = PairConLoss(temperature=self.args.temperature, m=self.args.m)
        self.attention_loss = Attention_loss(temperature=self.args.temperature)

        N = len(self.train_loader.dataset)
        self.a = torch.full((N, 1), 1/N).squeeze()

        self.b = torch.ones((self.args.num_classes,), dtype=torch.float64).to('cuda') / self.args.num_classes

        self.u = None
        self.v = None
        self.h = torch.FloatTensor([1])
        self.allb = [[self.b[i].item()] for i in range(self.args.classes)]
        self.label_ratios = torch.zeros(8)
        
        print(f"*****Intialize SCCLv, temp:{self.args.temperature}, eta:{self.args.eta}\n")

    def soft_ce_loss(self, pred, target, step):
        tmp = target ** 2 / torch.sum(target, dim=0)
        target = tmp / torch.sum(tmp, dim=1, keepdim=True)
        return torch.mean(-torch.sum(target * (F.log_softmax(pred, dim=1)), dim=1))

    def get_batch_token(self, text):
        token_feat = self.tokenizer.batch_encode_plus(
            text,
            max_length=self.args.max_length,
            return_tensors='pt', 
            padding='max_length', 
            truncation=True
        )
        return token_feat
        

    def prepare_transformer_input(self, batch):
        text1, text2, text3 = batch['text'], batch['augmentation_1'], batch['augmentation_2']
        feat1 = self.get_batch_token(text1)
        if self.args.augtype == 'explicit':
            feat2 = self.get_batch_token(text2)
            feat3 = self.get_batch_token(text3)
            input_ids = torch.cat([feat1['input_ids'].unsqueeze(1), feat2['input_ids'].unsqueeze(1), feat3['input_ids'].unsqueeze(1)], dim=1)
            attention_mask = torch.cat([feat1['attention_mask'].unsqueeze(1), feat2['attention_mask'].unsqueeze(1), feat3['attention_mask'].unsqueeze(1)], dim=1)
        else:
            input_ids = feat1['input_ids'] 
            attention_mask = feat1['attention_mask']
            
        return input_ids.cuda(), attention_mask.cuda()

    
    def loss_function(self, input_ids, attention_mask, selected, i):
        _, embd2, embd3 = self.model.get_embeddings(input_ids, attention_mask, task_type=self.args.augtype)

        # Instance-CL loss
        feat2, feat3 = self.model.contrast_logits(embd2, embd3)
        losses = self.contrast_loss(feat2, feat3)
        loss = self.eta * losses["loss"]
        losses['contrast'] = losses["loss"]
        self.args.tensorboard.add_scalar('loss/contrast_loss', losses['loss'].item(), global_step=i)

        # Attention loss
        if i < self.args.pre_step + self.args.pre_interval:
            target_label = self.L[selected].cuda()
        else:
            target_label = self.L.squeeze(0).cuda()
        h2, s2 = self.model.get_attention(embd2)
        h3, s3 = self.model.get_attention(embd3)
        s = (s2 + s3) / 2
        attention_loss = self.attention_loss(h2, h3, s, target_label)

        # Clustering loss
        if i >= self.args.pre_step + 1:
            P2 = self.model(embd2)
            P3 = self.model(embd3)  # predicted labels before softmax
            target_label = None
            if len(self.L.shape) != 1:
                if self.args.soft == True:
                    target = self.L.cuda()
                    cluster_loss = self.soft_ce_loss(P2, target, i) + self.soft_ce_loss(P3, target, i)
                else:
                    target_label = self.L.squeeze(0)
            else:
                target_label = self.L[selected].cuda()

            if target_label != None:
                cluster_loss = self.ce_loss(P2, target_label) + self.ce_loss(P3, target_label)
            loss += cluster_loss

            self.args.tensorboard.add_scalar('loss/cluster_loss', cluster_loss.item(), global_step=i)
            losses["cluster_loss"] = cluster_loss.item()
            
        losses['loss'] = loss
        self.args.tensorboard.add_scalar('loss/loss', loss, global_step=i)
        return loss, losses, attention_loss

    def train_step_explicit(self, input_ids, attention_mask, selected, i):
        if (self.args.pre_step != -1 and i == self.args.pre_step) or (self.args.pre_step == -1 and i == 0):
            self.label_ratios = self.generate_K_label_ratios()
            print('生成：', self.label_ratios)

        if i >= self.args.pre_step + self.args.pre_interval:
            simple_ps = self.get_labels(i, self.label_ratios, input_ids, attention_mask)
            cate_p = self.generate_OT_label_ratios(simple_ps)
            self.label_ratios = cate_p
        loss, losses, atten_loss = self.loss_function(input_ids, attention_mask, selected, i)
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()


        if atten_loss != 0 :
            # first_layer = self.model.contrast_head[0]  # contrast_head网络的参数
            # print("第一层权重值: ", first_layer.weight)
            # print("第一层权重梯度: ", first_layer.weight.grad)
            #
            # in_proj_weight_value = self.model.TransformerEncoderLayer.self_attn.in_proj_weight.data  # attention网络的参数
            # print("in_proj_weight 数值: ", in_proj_weight_value)
            # in_proj_weight_grad = self.model.TransformerEncoderLayer.self_attn.in_proj_weight.grad
            # print("in_proj_weight 梯度: ", in_proj_weight_grad)

            for param in self.model.parameters():
                param.requires_grad = False
            for param in self.model.TransformerEncoderLayer.parameters():
                param.requires_grad = True
            atten_loss.backward()
            self.optimizer2.step()
            # first_layer = self.model.contrast_head[0]  # 获取第一层，即 nn.Linear
            # print("第一层权重值: ", first_layer.weight)
            # print("第一层权重梯度: ", first_layer.weight.grad)
            #
            # in_proj_weight_value = self.model.TransformerEncoderLayer.self_attn.in_proj_weight.data
            # print("in_proj_weight 数值: ", in_proj_weight_value)
            # in_proj_weight_grad = self.model.TransformerEncoderLayer.self_attn.in_proj_weight.grad
            # print("in_proj_weight 梯度: ", in_proj_weight_grad)
            self.optimizer2.zero_grad()
            for param in self.model.parameters():
                param.requires_grad = True

        return losses

    def generate_OT_label_ratios(self, memberships):
        cate_P = torch.sum(memberships, dim=0)
        cate_P = cate_P / torch.sum(cate_P)
        return cate_P

    def generate_K_label_ratios(self):
        dataloader = unshuffle_loader(self.args)

        self.model.eval()
        for i, batch in enumerate(dataloader):
            with torch.no_grad():
                text, label = batch['text'], batch['label']
                feat = self.get_batch_token(text)
                embeddings = self.model.get_embeddings(feat['input_ids'].cuda(), feat['attention_mask'].cuda(),
                                                       task_type="evaluate")
                if i == 0:
                    all_embeddings = embeddings.detach()
                else:
                    all_embeddings = torch.cat((all_embeddings, embeddings.detach()), dim=0)
        embeddings = all_embeddings.cpu().numpy()
        kmeans = cluster.KMeans(n_clusters=self.args.classes, random_state=self.args.seed, max_iter=3000, tol=0.01)
        kmeans.fit(embeddings)
        kpred_labels = torch.tensor(kmeans.labels_.astype(int))
        arr_pred = Counter(np.array(kpred_labels))
        print("label_counts", arr_pred)
        unique_label = sorted(arr_pred.keys())
        label_ratios = [arr_pred.get(label, 0) / kpred_labels.shape[0] for label in unique_label]
        label_ratios = torch.tensor(label_ratios)
        return label_ratios

    def optimize_labels(self, step, label_ratios, input_ids, attention_mask):

        emb1, emb2, emb3 = self.model.get_embeddings(input_ids, attention_mask, task_type=self.args.augtype)  # embedding
        p = F.softmax(self.model(emb1), dim=1)
        PS = p.detach().cpu()
        _, s2 = self.model.get_attention(emb2)
        _, s3 = self.model.get_attention(emb3)
        s = (s2 + s3) / 2


        a = torch.ones((PS.shape[0],), dtype=torch.float64).to('cuda') / PS.shape[0]
        pseudo_label, c_b = curriculum_structure_aware_PL(a, self.b, PS, s, label_ratios, lambda1=self.args.lambda1, lambda2=self.args.lambda2, lambda3=self.args.lambda3,
                                                                 version='fast',
                                                                 reg_e=0.1,
                                                                 reg_sparsity=None)

        self.b = c_b
        self.L = pseudo_label.unsqueeze(0)
        return PS
  
    def get_labels(self, step, label_ratios, input_ids, attention_mask):
        # 更新self.L
        sim_P = self.optimize_labels(step, label_ratios, input_ids, attention_mask)
        return sim_P

    def train(self):
        self.optimize_times = ((np.linspace(self.args.start, 1, self.args.M)**2)[::-1] * self.args.max_iter).tolist()
        # 训练前评估
        self.evaluate_embedding(-1)
        
        for i in np.arange(self.args.max_iter+1):
            self.model.train()
            try:
                batch, selected = next(train_loader_iter)
            except:
                train_loader_iter = iter(self.train_loader)
                batch, selected = next(train_loader_iter)


            input_ids, attention_mask = self.prepare_transformer_input(batch)
            losses = self.train_step_explicit(input_ids, attention_mask, selected, i)

            if (self.args.print_freq>0) and ((i%self.args.print_freq==0) or (i==self.args.max_iter) or (i==self.args.pre_step + self.args.pre_interval - 1)):
                print('当前epoch：', i)

            if (i >=self.args.pre_step) and ((i % self.args.print_freq == 0) or (i == self.args.max_iter) or (i == self.args.pre_step + self.args.pre_interval - 1)):
                statistics_log(self.args.tensorboard, losses=losses, global_step=i)
                flag = self.evaluate_embedding(i)
                if flag < 0:
                    break
        return None

    def evaluate_embedding(self, step):
        dataloader = unshuffle_loader(self.args)
        print('---- {} evaluation batches ----'.format(len(dataloader)))

        self.model.eval()
        for i, batch in enumerate(dataloader):
            with torch.no_grad():
                text, label = batch['text'], batch['label'] 
                feat = self.get_batch_token(text)
                embeddings = self.model.get_embeddings(feat['input_ids'].cuda(), feat['attention_mask'].cuda(), task_type="evaluate")
                pred = torch.argmax(self.model(embeddings), dim=1)

                if i == 0:
                    all_labels = label
                    all_embeddings = embeddings.detach()
                    all_pred = pred.detach()
                else:
                    all_labels = torch.cat((all_labels, label), dim=0)
                    all_embeddings = torch.cat((all_embeddings, embeddings.detach()), dim=0)
                    all_pred = torch.cat((all_pred, pred.detach()), dim=0)

        # # Assume embeddings and tsne_labels are already defined
        # embeddings = all_embeddings.cpu().numpy()
        # tsne_labels = np.array(all_labels)
        #
        # df_embedding = pd.DataFrame(embeddings)
        # df_labels = pd.DataFrame(tsne_labels)
        # df_labels.to_csv('./0_结果/1_tsne/stack/' + str(step) + '_label.csv', index=False, header=False)
        # df_embedding.to_csv('./0_结果/1_tsne/stack/' + str(step) + '_embedding.csv', index=False, header=False)
        #
        # X_embedded = TSNE(n_components=2).fit_transform(embeddings)
        #
        # plt.figure()
        #
        # plt.figure()
        #
        # # 手动设置颜色列表，确保有 20 种不同的颜色
        # colors = ['#1abc9c', '#16a085', '#2ecc71', '#27ae60', '#c0392b', '#2980b9', '#9b59b6', '#e74c3c', '#34495e',
        #           '#2c3e50',
        #           '#003f5c', '#f39c12', '#e67e22', '#d35400', '#d9d0b4', '#3498db', '#8e44ad', '#bdc3c7', '#95a5a6',
        #           '#7f8c8d']
        #
        # for label in np.unique(tsne_labels):
        #     plt.scatter(X_embedded[tsne_labels == label, 0], X_embedded[tsne_labels == label, 1],
        #                 label=label, s=0.3, color=custom_colors[label])  # 使用自定义颜色列表
        #
        # plt.title("t-SNE Visualization")
        # plt.xlabel("Dimension 1")
        # plt.ylabel("Dimension 2")
        # plt.legend()
        # plt.savefig('./0_结果/1_tsne/stack_初始图/tSNE' + str(step) + '.png')
        # plt.show()



        # Initialize confusion matrices
        confusion = Confusion(max(self.args.num_classes, self.args.classes))
        embeddings = all_embeddings.cpu().numpy()
        pred_labels = all_pred.cpu()
        if step <= self.args.pre_step:
            kmeans = cluster.KMeans(n_clusters=self.args.classes, random_state=self.args.seed, max_iter=3000, tol=0.01)
            kmeans.fit(embeddings)
            kpred_labels = torch.tensor(kmeans.labels_.astype(int))
            self.L = kpred_labels
            pred_labels = kpred_labels
        # clustering accuracy
        clusters_num = len(set(pred_labels.numpy()))

        self.args.tensorboard.add_scalar('Test/preded_clusters', clusters_num, step)
        confusion.add(pred_labels, all_labels)
        _, _ = confusion.optimal_assignment(self.args.num_classes)

        acc = confusion.acc()
        clusterscores = confusion.clusterscores(all_labels, pred_labels)

        ressave = {"acc":acc}
        ressave.update(clusterscores)
        for key, val in ressave.items():
            self.args.tensorboard.add_scalar('Test/{}'.format(key), val, step)

        arr_pred = Counter(np.array(pred_labels))

        stop_flag = 0
        y_pred = pred_labels.numpy()
        if step == -1:
            self.y_pred_last = np.copy(y_pred)
        else:
            change_rate = np.sum(y_pred != self.y_pred_last).astype(np.float32) / y_pred.shape[0]
            self.args.tensorboard.add_scalar('Test/change_rate', change_rate, step)
            self.y_pred_last = np.copy(y_pred)
            print('[Step] {} Label change rate: {:.3f} tol: {:.3f}'.format(step, change_rate, self.args.tol))
            if (step > self.args.pre_step and change_rate < self.args.tol) or step >= 4000:
                print('Reached tolerance threshold, stop training.')
                stop_flag = -1
            elif(step > self.args.pre_step + self.args.pre_interval and change_rate > 0.8):
                print('Great fluctuation, stop training.')
                stop_flag = -1
        if stop_flag + 1 >= 0:
            if step <= self.args.pre_step:
                print('preded classes number:', clusters_num)
                print('[Step]', step)
                print('[Kmeans] Clustering scores:', clusterscores)
                print('[Kmeans] ACC: {:.4f}'.format(acc))
                print('Kmeans：', len(arr_pred), arr_pred)
            else:
                print('preded classes number:', clusters_num)
                print('[Step]', step)
                print('[Model] Clustering scores:', clusterscores)
                print('[Model] ACC: {:.4f}'.format(acc))
                print('Model：', len(arr_pred), arr_pred)
        return stop_flag




