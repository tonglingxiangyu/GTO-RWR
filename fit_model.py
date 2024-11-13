import argparse
import numpy as np
import time
import random
import torch
import torch.nn as nn
from torch.autograd import Variable
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.metrics import roc_curve, auc, roc_auc_score, classification_report
from sklearn.preprocessing import label_binarize
from scipy import interp
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
import itertools
from sklearn.model_selection import train_test_split
from loader import (load_sage, load_gat)
from models import (EGraphSage, EResGAT)
import os
#os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb=512'
import warnings

global global_edge_features
from models.egraphsage import global_edge_features

warnings.filterwarnings("ignore")
#os.environ['CUDA_VISIBLE_DEVICE']='0'

np.random.seed(1)
random.seed(1)
data_class = {
              "Darknet":8,
              "CES-CIC":7,
              "iot":10,
              "tor": 8,
              "unsw": 9
              }
data_lr = {
           "Darknet":0.003,
           "CES-CIC":0.003,
           "iot":0.01,
           "tor": 0.01,
           "unsw": 0.01
           }
"""
            bi_rwr: 0.005
            mul_rwr: 0.01
"""
test_size = {
             "Darknet":45000,
             "CES-CIC":75000,
             "iot":140000,
             "tor": 800,
             "unsw": 3500}

def fit(args):
    alg = args.alg
    data = args.dataset
    binary = args.binary
    residual = args.residual
    rwr = args.RWR
    GTO = args.GTO
    patience = 5 # 早停法的阈值
    epochs = 100000
    """
        bi_rwr: 20
        multi_rwr: 100

    """
    if data == 'tor':
        if binary:
            d_class = 'bi'
        else:
            d_class = 'mul'
    else:
        # d_class = 'data'
        d_class = 'alldata'

    path = "datasets/"+ data +'/' + d_class + '/' 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cuda = torch.cuda.is_available()
    if alg == "sage":
        enc2, edge_feat, label, node_map, adj = load_sage(path, binary, GTO, rwr, cuda, device)
        model = EGraphSage(data_class[data], enc2, edge_feat, node_map, adj, residual)
    else:
        edge_feat, label, adj, adj_lists, config = load_gat(path, device, binary)
        model = EResGAT(
            num_of_layers=config['num_of_layers'],
            num_heads_per_layer=config['num_heads_per_layer'],
            num_features_per_layer=config['num_features_per_layer'],
            num_identity_feats=config['num_identity_feats'],
            edge_feat=edge_feat,
            adj=adj,
            adj_lists=adj_lists,
            device=device,
            add_skip_connection=config['add_skip_connection'],
            residual=residual,
            bias=config['bias'],
            dropout=config['dropout']
        ).to(device)
    label_list = label.tolist()
    # unique_labels, counts = np.unique(label, return_counts=True)
    # print('unique_labels', unique_labels)
    # print('counts', counts)

    # loss and optimizer
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad,
                                            model.parameters()),
                                 lr=data_lr[data],
                                 amsgrad=False)
    #optimizer = torch.optim.RMSprop(model.parameters(), lr=data_lr[data])
    # optimizer = torch.optim.SGD(model.parameters(), lr=data_lr[data], momentum=0.9)

    # train test split
    num_edges = len(edge_feat)
    print('num_edges ', num_edges)
    if alg == "sage":
        # train_val, test = train_test_split(np.arange(num_edges), test_size=0.1, stratify=label, random_state=42) # seed = 42
        # train, val = train_test_split(train_val, test_size=0.111, stratify=label[train_val],  random_state=42) # 这里的test_size是验证集的大小, 8:1:1
        train_val, test = train_test_split(np.arange(num_edges), test_size=0.2, stratify=label, random_state=42) # seed = 42
        train, val = train_test_split(train_val, test_size=0.125, stratify=label[train_val],  random_state=42) # 这里的test_size是验证集的大小, 7:1:2
    else:
        train_val, test = train_test_split(np.arange(num_edges), test_size=0.2, stratify=label.cpu(), random_state=123)
        train, val = train_test_split(train_val, test_size=0.125, stratify=label[train_val].cpu(),  random_state=123)

    print('train_num:', len(train))
    print('val_num:', len(val))
    print('test_num:', len(test))

    batch_size = 500

    # adjusted_val = len(val) - (len(val) % batch_size)
    # val = val[:adjusted_val]
    # adjusted_test = len(test) - (len(test) % batch_size)
    # test = test[:adjusted_test]

    times = []
    train_loss = []
    train_acc = []
    train_f1 = []
    val_loss = []
    val_acc =[]
    val_f1 = []
    path_model = './runs/' + alg + "/" + 'best.pt'
    early_stopping = EarlyStopping(patience=patience, verbose=True, path=path_model)

    for epoch in range(epochs):
        print("Epoch: ", epoch + 1)
        random.shuffle(train)
        epoch_start = time.time()
        losses = 0
        accs = 0
        precisions = 0
        recalls = 0
        f1s = 0
        macro_precisions = 0
        macro_recalls = 0
        macro_f1s = 0
        num_batch = int(len(train) / batch_size)
        
        for batch in range(num_batch):  # batches in train data
            batch_edges = train[batch_size * batch: batch_size * (batch + 1)]  # 500 records per batch
            start_time = time.time()
            # training
            #model = model.cuda()
            model.train()
            # if binary == False: # 不分批训练
            #     batch_edges = train
            output, _ = model(batch_edges)
            if alg == "sage":
                batch_output = output.data.numpy()
                loss = model.loss(batch_edges,
                                        Variable(torch.LongTensor(label[np.array(batch_edges)])))
                losses += loss.item()

                acc = accuracy_score(label[batch_edges],
                                     batch_output.argmax(axis=1))
                accs += acc.item()

                recall = recall_score(label[batch_edges],
                                      batch_output.argmax(axis=1),
                                     average="weighted")
                recalls += recall.item()
                macro_recall = recall_score(label[batch_edges],
                                      batch_output.argmax(axis=1),
                                     average="macro")
                macro_recalls += macro_recall.item()

                precision = precision_score(label[batch_edges],
                                            batch_output.argmax(axis=1),
                                     average="weighted")
                precisions += precision.item()
                macro_precision = precision_score(label[batch_edges],
                                            batch_output.argmax(axis=1),
                                     average="macro")
                macro_precisions += macro_precision.item()

                f1 = f1_score(label[batch_edges],
                              batch_output.argmax(axis=1),
                                     average="weighted")
                f1s += f1.item()
                macro_f1 = f1_score(label[batch_edges],
                              batch_output.argmax(axis=1),
                                     average="macro")
                macro_f1s += macro_f1.item()
            else:
                _, out, _, idx = output
                batch_output = out.index_select(0, idx)
                loss = loss_fn(batch_output, label[batch_edges])
                losses += loss.item()

                acc = accuracy_score(label[batch_edges].cpu(),
                                     torch.argmax(batch_output.cpu(), dim=-1))
                accs += acc.item()

                recall = recall_score(label[batch_edges].cpu(),
                                      torch.argmax(batch_output.cpu(), dim=-1),
                                      average="weighted")
                recalls += recall.item()
                macro_recall = recall_score(label[batch_edges].cpu(),
                                      torch.argmax(batch_output.cpu(), dim=-1),
                                      average="macro")
                macro_recalls += macro_recall.item()

                precision = precision_score(label[batch_edges].cpu(),
                                            torch.argmax(batch_output.cpu(), dim=-1),
                                            average="weighted")
                precisions += precision.item()
                macro_precision = precision_score(label[batch_edges].cpu(),
                                            torch.argmax(batch_output.cpu(), dim=-1),
                                            average="macro")
                macro_precisions += macro_precision.item()

                f1 = f1_score(label[batch_edges].cpu(),
                              torch.argmax(batch_output.cpu(), dim=-1),
                              average="weighted")
                f1s += f1.item()
                macro_f1 = f1_score(label[batch_edges].cpu(),
                              torch.argmax(batch_output.cpu(), dim=-1),
                              average="macro")
                macro_f1s += macro_f1.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            end_time = time.time()
            times.append(end_time - start_time)

            print('batch: {:03d}'.format(batch + 1),
                  'loss_train: {:.4f}'.format(loss.item()),
                  'acc_train: {:.4f}'.format(acc.item()),
                  'recall_train: {:.4f}'.format(recall.item()),
                  'macro_recall_train: {:.4f}'.format(macro_recall.item()),
                  'precision_train: {:.4f}'.format(precision.item()),
                  'macro_precisionl_train: {:.4f}'.format(macro_precision.item()),
                  'f1_train: {:.4f}'.format(f1.item()),
                  'macro_f1_train: {:.4f}'.format(macro_f1.item()),
                  'time: {:.4f}s'.format(end_time - start_time))
            # if binary == False: # 不分批训练
            #     break
            # if batch >= 10:
            #     break
        if epoch == 0:
            # 将全局变量的边特征保存到Numpy文件
            global global_edge_features
            if global_edge_features:
                global_edge_features = np.array(global_edge_features)
                print('global_edge_features', global_edge_features.shape)
                global_edge_features = np.vstack(global_edge_features)
                print('global_edge_features', global_edge_features.shape)
                # global_edge_features['label'] = label_list[:global_edge_features.shape[0]]
                np.save(path + 'global_edge_features.npy', global_edge_features)
        global_edge_features = []
        epoch_end = time.time()
    
        losses /= num_batch
        accs /= num_batch
        precisions /= num_batch
        recalls /= num_batch
        f1s /= num_batch
        # macro_precisions /= num_batch
        # macro_recalls /= num_batch
        # macro_f1s /= num_batch
        train_loss.append(losses)
        train_acc.append(accs)
        train_f1.append(f1s)

        # Validation
        print("validation")
        #label = label.float()
        model.eval()
        loss_val, acc_val, recall_val, pre_val, f1_val, macro_recall_val, macro_pre_val, macro_f1_val, val_output = predict_(alg, binary, model, label, loss_fn, val, save_data=False)
        val_loss.append(loss_val)
        val_acc.append(acc_val)
        val_f1.append((f1_val))

        print('loss_val: {:.4f}'.format(loss_val),
              'acc_val: {:.4f}'.format(acc_val),
              'recall_val: {:.4f}'.format(recall_val),
              'macro_recall_val: {:.4f}'.format(macro_recall_val),
              'precision_val: {:.4f}'.format(pre_val),
              'macro_precisionl_val: {:.4f}'.format(macro_pre_val),
              'f1_val: {:.4f}'.format(f1_val),
              'macro_f1_val: {:.4f}'.format(macro_f1_val),
              'average batch time: {:.4f}s'.format(np.mean(times)),
              'epoch time: {:.2f}min'.format((epoch_end - epoch_start)/60.0))

        early_stopping(acc_val, model)
        if early_stopping.early_stop:
            print("Early stopping, acc:", early_stopping.val_loss_min)
            break



    # 绘制统计折线图
    plot_picture(train_loss, alg, "train loss")
    plot_picture(train_acc, alg, "train accuracy")
    plot_picture(train_f1, alg, "train f1-score")

    plot_picture(val_loss, alg, "validation loss")
    plot_picture(val_acc, alg, "validation accuracy")
    plot_picture(val_f1, alg, "validation f1-score")

    plt.close()

    # Testing
    print("testing")
    # 从最佳模型权重加载模型
    model.load_state_dict(torch.load(path_model))
    model.eval()
    if binary:
        loss_test, acc_test, recall_test, pre_test, f1_test, auc_test, macro_recall_test, macro_pre_test, macro_f1_test, predict_output = predict_(alg, binary, model, label, loss_fn, test, save_data=True)
    else:
        loss_test, acc_test, recall_test, pre_test, f1_test, weighted_auc, macro_recall_test, macro_pre_test, macro_f1_test, macro_auc, predict_output = predict_(alg, binary, model, label, loss_fn, test, save_data=True)

    if alg == "sage":
        print("Test set results:", "loss= {:.4f}".format(loss_test),
              "acc= {:.4f}".format(acc_test), # acc没有average参数，它是直接给出的
              'recall_test: {:.4f}'.format(recall_test),
              'macro_recall_test: {:.4f}'.format(macro_recall_test),
              'precision_test: {:.4f}'.format(pre_test),
              'macro_precisionl_test: {:.4f}'.format(macro_pre_test),
              'f1_test: {:.4f}'.format(f1_test),
              'macro_f1_test: {:.4f}'.format(macro_f1_test),
              "label acc=", f1_score(label[test], predict_output, average=None)) # 这里的average=None参数意味着它不会对F1分数进行平均处理。因此，这个调用将返回每个类别的F1分数数组，而不是一个单一的平均F1分数值。
    else:
        print("Test set results:", "loss= {:.4f}".format(loss_test),
              "accuracy= {:.4f}".format(acc_test),
              'recall_test: {:.4f}'.format(recall_test),
              'macro_recall_test: {:.4f}'.format(macro_recall_test),
              'precision_test: {:.4f}'.format(pre_test),
              'macro_precisionl_test: {:.4f}'.format(macro_pre_test),
              'f1_test: {:.4f}'.format(f1_test),
              'macro_f1_test: {:.4f}'.format(macro_f1_test),
              "label acc=", f1_score(label[test].cpu(), predict_output, average=None))
    if binary:
        print('auc:', auc_test)
    else:
        print('weighted_auc: {:.4f}'.format(weighted_auc),
        'macro_auc: {:.4f}'.format(macro_auc))

    metrics_path = './runs/' + alg + '/metrics.txt'
    with open(metrics_path, 'a') as f:
        f.write(f"Accuracy:{acc_test}, Recall: {recall_test}, Precision: {pre_test}, F1-Score: {f1_test}\n")
        f.write(f"Macro_Recall: {macro_recall_test}, Macro_Precision: {macro_pre_test}, Macro_F1-Score: {macro_f1_test}")

    # predict_output = predict_output.argmax(1)
    # predict_output = torch.Tensor.cpu(predict_output).detach().numpy()
    #print("predict:", predict_output)
    if alg == "sage":
        actual = label[test]
    else:
        actual = label[test].cpu()
    #print("actual:", actual)
    # 绘制混淆矩阵
    plot_confusion_matrix(cm=confusion_matrix(actual, predict_output),
                          target_names=np.unique(actual),
                          recall = recall_test,
                          macro_recall = macro_recall_test,
                          title="Confusion Matrix",
                          path = 'cm.png',
                          alg = alg,
                          normalize=True)
    
    plot_confusion_matrix(cm=confusion_matrix(actual, predict_output),
                          target_names=np.unique(actual),
                          recall = recall_test,
                          macro_recall = macro_recall_test,
                          title="Confusion Matrix",
                          path = 'cm_num.png',
                          alg = alg,
                          normalize=False)

def predict_(alg, binary, model, label, loss_fn, data_idx, save_data):
    predict_output = []
    size = 500

    losses = 0
    accs = 0
    precisions = 0
    recalls = 0
    f1s = 0
    macro_precisions = 0
    macro_recalls = 0
    macro_f1s = 0
    
    all_labels = []
    all_probabilities = []

    num_batch = len(data_idx) // size
    if len(data_idx) % size != 0:
        num_batch += 1

    for batch in range(num_batch):  # 加1确保处理所有批次
        if batch < num_batch:
            batch_edges = data_idx[size * batch:size * (batch + 1)]
        else:
            # 最后一个批次包含所有剩余的数据
            batch_edges = data_idx[size * batch:]
        batch_output, _ = model(batch_edges)
    # num_batch = len(data_idx) // size
    # for batch in range(num_batch):
    #     print(batch + 1)
    #     batch_edges = data_idx[size * batch:size * (batch + 1)]
    #     batch_output, _ = model(batch_edges)

        if alg == "sage":
            batch_output = batch_output.data.numpy()
            batch_prob = torch.softmax(torch.tensor(batch_output), dim=1).numpy() # 概率用于绘制ROC
            # batch_output = batch_output.data.numpy().argmax(axis=1)
            batch_output = batch_output.argmax(axis=1)
            batch_loss = model.loss(batch_edges,
                              Variable(torch.LongTensor(label[np.array(batch_edges)])))
            losses += batch_loss.item()
            acc = accuracy_score(label[batch_edges],
                                 batch_output)
            accs += acc.item()
            
            recall = recall_score(label[batch_edges],
                                        batch_output,
                                        average="weighted")
            recalls += recall.item()
            macro_recall = recall_score(label[batch_edges],
                                        batch_output,
                                        average = 'macro')
            macro_recalls += macro_recall.item()

            precision = precision_score(label[batch_edges],
                                 batch_output,
                                 average="weighted")
            precisions += precision.item()
            macro_precision = precision_score(label[batch_edges],
                                 batch_output,
                                 average="macro")
            macro_precisions += macro_precision.item()

            f1 = f1_score(label[batch_edges],
                                        batch_output,
                                        average="weighted")
            f1s += f1.item()
            macro_f1 = f1_score(label[batch_edges],
                                        batch_output,
                                        average="macro")
            macro_f1s += macro_f1.item()

        else:
            _, out, _, idx = batch_output
            batch_output = out.index_select(0, idx)
            batch_loss = loss_fn(batch_output, label[batch_edges])
            losses += batch_loss.item()
            batch_prob = torch.softmax(batch_output.cpu(), dim=1).detach().numpy() # 概率用于绘制ROC
            acc = accuracy_score(label[batch_edges].cpu(),
                                 torch.argmax(batch_output.cpu(), dim=-1))
            accs += acc.item()

            recall = recall_score(label[batch_edges].cpu(),
                                 torch.argmax(batch_output.cpu(), dim=-1),
                                 average="weighted")
            recalls += recall.item()
            macro_recall = recall_score(label[batch_edges].cpu(),
                                 torch.argmax(batch_output.cpu(), dim=-1),
                                 average="macro")
            macro_recalls += macro_recall.item()

            precision = precision_score(label[batch_edges].cpu(),
                        torch.argmax(batch_output.cpu(), dim=-1),
                        average="weighted")
            precisions += precision.item()
            macro_precision = precision_score(label[batch_edges].cpu(),
                        torch.argmax(batch_output.cpu(), dim=-1),
                        average="macro")
            macro_precisions += macro_precision.item()

            f1 = f1_score(label[batch_edges].cpu(),
                                 torch.argmax(batch_output.cpu(), dim=-1),
                                 average="weighted")
            f1s += f1.item()
            macro_f1 = f1_score(label[batch_edges].cpu(),
                                 torch.argmax(batch_output.cpu(), dim=-1),
                                 average="macro")
            macro_f1s += macro_f1.item()

            batch_output = torch.argmax(batch_output.cpu(), dim=-1)
            #print(batch_output)
            #print(acc_train)

            # batch_output = out.index_select(0, idx)
            # batch_loss = loss_fn(batch_output, label[batch_edges])
        #acc_output.append(acc_train)
        predict_output.extend(batch_output)

        all_labels.append(label[batch_edges])
        all_probabilities.append(batch_prob)
    
    losses /= num_batch
    accs /= num_batch
    precisions /= num_batch
    recalls /= num_batch
    f1s /= num_batch
    macro_precisions /= num_batch
    macro_recalls /= num_batch
    macro_f1s /= num_batch

    # 保存数据和图片
    if save_data:
        if alg == "gat":
            all_labels = [label.cpu().numpy() for label in all_labels]

        # Flatten the collected lists
        all_labels = np.concatenate(all_labels)
        all_probabilities = np.concatenate(all_probabilities)    

        # Compute ROC curve and ROC area for each class
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        
        if binary:
            fpr[0], tpr[0], _ = roc_curve(all_labels, all_probabilities[:, 1])
            roc_auc[0] = auc(fpr[0], tpr[0])
            n_classes = 1
        else:
            # Binarize the labels for multi-class ROC calculation
            all_labels_bin = label_binarize(all_labels, classes=np.unique(all_labels))
            n_classes = all_labels_bin.shape[1]
            for i in range(n_classes):
                fpr[i], tpr[i], _ = roc_curve(all_labels_bin[:, i], all_probabilities[:, i])
                roc_auc[i] = auc(fpr[i], tpr[i]) # OVR 一对多
            
            # micro roc
            # fpr["micro"], tpr["micro"], _ = roc_curve(all_labels.ravel(), all_probabilities.ravel())
            # roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

            # Compute macro-average ROC curve and ROC area
            all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
            mean_tpr = np.zeros_like(all_fpr)
            for i in range(n_classes):
                mean_tpr += interp(all_fpr, fpr[i], tpr[i])
            mean_tpr /= n_classes
            fpr["macro"] = all_fpr
            tpr["macro"] = mean_tpr
            roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

            # Compute weighted-average ROC curve and ROC area
            weights = np.bincount(all_labels) / len(all_labels)
            weighted_tpr = np.zeros_like(all_fpr)
            for i in range(n_classes):
                weighted_tpr += weights[i] * interp(all_fpr, fpr[i], tpr[i])
            fpr["weighted"] = all_fpr
            tpr["weighted"] = weighted_tpr
            roc_auc["weighted"] = auc(fpr["weighted"], tpr["weighted"])

        # Plot all ROC curves
        plt.figure()
        # colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
        # for i, color in zip(range(n_classes), colors):
        #     plt.plot(fpr[i], tpr[i], color=color, lw=2,
        #              label='ROC curve of class {0} (area = {1:0.2f})'
        #              ''.format(i, roc_auc[i]))
        if binary:
            plt.plot(fpr[0], tpr[0], color='darkorange', lw=2,
                    label='ROC curve (area = {0:0.2f})'.format(roc_auc[0]))
        else:
            plt.plot(fpr["macro"], tpr["macro"], color='navy', linestyle='-.', lw=2,
                    label='Macro-average ROC curve (area = {0:0.2f})'.format(roc_auc["macro"]))
            plt.plot(fpr["weighted"], tpr["weighted"], color='red', linestyle='-', lw=2,
                    label='Weighted-average ROC curve (area = {0:0.2f})'.format(roc_auc["weighted"]))

        plt.plot([0, 1], [0, 1], 'k--', lw=2)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic - Multi-class')
        plt.legend(loc="lower right")
        plot_path = './runs/' + alg + '/roc_curves.png'
        plt.savefig(plot_path)
        # plt.show()

        # AUC values
        metrics_path = './runs/' + alg + '/metrics.txt'
        with open(metrics_path, 'w') as f:
            # 计算并保存每个类别的 recall, precision, f1
            # report = classification_report(all_labels, predict_output, target_names=[f'Class {i}' for i in range(n_classes)])
            report = classification_report(all_labels, predict_output, target_names=[f'Class {i}' for i in np.unique(all_labels)], digits=6)
            f.write('\nClassification Report:\n')
            f.write(report)
            if binary:
                f.write(f'ROC AUC: {roc_auc[0]:.2f}\n')
            else:
                f.write(f'Macro-average AUC-ROC: {roc_auc["macro"]:.2f}\n')
                f.write(f'Weighted-average AUC-ROC: {roc_auc["weighted"]:.2f}\n')
                for i in range(n_classes):
                    f.write(f'AUC-ROC for class {i}: {roc_auc[i]:.2f}\n')
        if binary:
            return losses, accs, recalls, precisions, f1s, roc_auc[0], macro_recalls, macro_precisions,  macro_f1s, predict_output
        return losses, accs, recalls, precisions, f1s, roc_auc["weighted"], macro_recalls, macro_precisions,  macro_f1s, roc_auc["macro"], predict_output
    else:
        return losses, accs, recalls, precisions, f1s, macro_recalls, macro_precisions,  macro_f1s, predict_output

# 绘制曲线
def plot_picture(lists, alg, name):
    plt.plot(range(len(lists)), lists, label=name)
    plt.xlabel('Epoch')
    plt.ylabel(name)
    plt.legend()
    path = './runs/' + alg + '/' + name + '.png'
    plt.savefig(path)
    #  plt.show()

# 绘制混淆矩阵
def plot_confusion_matrix(cm,
                          target_names,
                          recall,
                          macro_recall,
                          title='Confusion matrix',
                          alg = 'sage',
                          path = 'cm.png',
                          cmap=None,
                          normalize=True):
    # accuracy = np.trace(cm) / float(np.sum(cm))
    # misclass = 1 - accuracy
    

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)

    if normalize:
        # cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        # 四舍五入到指定的小数位数，例如2位
        decimal_places = 4
        cm = np.round(cm_normalized, decimal_places)
        # mappable = ScalarMappable(norm=Normalize(vmin=0, vmax=1), cmap=cmap)
        # plt.colorbar(mappable) # 背景从0到1颜色变深
        # 调整每行的最后一个元素，使每行的和为1
        # for i in range(cm.shape[0]):
        #     cm[i, -1] = 1 - np.sum(cm[i, :-1])
        
    if cmap is None:
        cmap = plt.get_cmap('Blues')

    plt.figure(figsize=(12, 12))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.4f}".format(cm[i, j]), # 显示百分比和个数
                     horizontalalignment="center",
                     color="black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     # color="white" if cm[i, j] > thresh else "black")
                     color = "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label\nrecall={:0.4f}; macro_recall={:0.4f}'.format(recall, macro_recall))
    paths = './runs/' + alg + '/' + path
    plt.savefig(paths)
    # plt.show()

# 定义早停法类
class EarlyStopping:
    def __init__(self, patience=20, verbose=False, delta=0, path='./runs/checkpoint.pt'):
        """
        初始化早停法参数

        参数:
            patience (int): 验证损失没有改善时，允许的额外时期数量
            verbose (bool): 如果为True，则打印日志消息
            delta (float): 验证损失之间的最小变化，认为是改进
            path (str): 保存最佳模型权重的文件路径
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path

    def __call__(self, val_loss, model): # 当一个类的实例被当作函数那样调用时，该类的__call__方法会被自动执行
        """
        检查是否应该停止早期，并保存最佳模型

        参数:
            val_loss (float): 当前时期的验证损失
            model (torch.nn.Module): 当前模型
        """
        # score = -val_loss # 损失越小，score越大
        score = val_loss # 精度

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score <= self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience} best score: {self.best_score}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        """
        保存模型权重

        参数:
            val_loss (float): 当前时期的验证损失
            model (torch.nn.Module): 当前模型
        """
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss

# -----------------------------------------------------------------------------

if __name__ == '__main__':
    def str2bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Boolean value expected.')

    ALG = ['sage', 'gat']
    DATA = ['Darknet', 'CES-CIC', 'iot', 'tor', 'unsw']

    p = argparse.ArgumentParser()
    p.add_argument('--alg',
                   help='algorithm to use.',
                   default='gat',
                   choices=ALG)
    p.add_argument('--dataset',
                   help='Experimental dataset.',
                   type=str,
                   default='Darknet',
                   choices=DATA)
    p.add_argument('--binary',
                   help='Perform binary or muticlass task',
                   type=str2bool,
                   default=False)
    p.add_argument('--residual',
                   help='Apply modified model with residuals or not',
                   type=str2bool,
                   default=False)
    p.add_argument('--GTO',
                   help='Apply Graph Topology Optimization or not',
                   type=str2bool,
                   default=False)
    p.add_argument('--RWR',
                   help='Apply RWR algorithm or not',
                   type=str2bool,
                   default=False)
    # Parse and validate script arguments.

    args = p.parse_args()
    print(args)

    # Training and testing
    fit(args)
