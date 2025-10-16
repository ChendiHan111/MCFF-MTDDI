import numpy as np
from sklearn.model_selection import KFold,StratifiedKFold
import time
import torch
import csv
import os
import random
import torch.nn as nn
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,  average_precision_score, cohen_kappa_score
import torch.utils.data as Data
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR
import logging
import torch.nn.functional as F
import warnings
warnings.filterwarnings("ignore")

def get_drugpair_info(dir,list,drugs):
    with open(dir) as raw_input_file:
        data = csv.reader(raw_input_file, delimiter=',')
        header=next(data)
        for p, r in data:
            list.append([eval(p), eval(r)])
            if eval(p)[0] not in drugs:
                drugs.append(eval(p)[0])
            if eval(p)[1] not in drugs:
                drugs.append(eval(p)[1])
        return list,drugs
        

def feature_vector(feature_dir,drugs):
    feature={}
    with open(feature_dir) as file1:
        data=csv.reader(file1)
        if feature_dir!=filename[3]and feature_dir!=filename[5]:
            header=next(data)
        if feature_dir!=filename[5]:
            for d, emb in data:
                if d in drugs:
                    feature[d]=eval(emb)
        else:
            for d,emb in data:
                feature[eval(d)]=eval(emb)
    return feature


def train_test_data1(data_lis):
    train_X_data=[]
    train_Y_data=[]
    test_X_data=[]
    test_Y_data=[]
    
    data_lis=np.array(data_lis,dtype=object)
    drug_pair=data_lis[:,0]
    Y=data_lis[:,1]
    label=np.array(list(map(int,Y)))

    kfold=StratifiedKFold(n_splits=5,shuffle=True,random_state=3)
    
    for train,test in kfold.split(drug_pair,label):
        train_X_data.append(drug_pair[train])
        train_Y_data.append(label[train])
        test_X_data.append(drug_pair[test])
        test_Y_data.append(label[test])
    train_X=np.array(train_X_data,dtype=object)
    train_Y=np.array(train_Y_data,dtype=object)
    test_X=np.array(test_X_data,dtype=object)
    test_Y=np.array(test_Y_data,dtype=object)
    return train_X,train_Y,test_X,test_Y


def create_log_id(dir_path):
    log_count=0
    file_path=os.path.join(dir_path,'log{:d}'.format(log_count))
    while os.path.exists(file_path):
        log_count+=1
        file_path=os.path.join(dir_path,'log{:d}'.format(log_count))
    return log_count


def logging_config(folder=None, name=None, level=logging.DEBUG,console_level=logging.DEBUG,no_console=True):
    if not os.path.exists(folder):
        os.makedirs(folder)
    for handler in logging.root.handlers:
        logging.root.removeHandler(handler)
    logging.root.handlers = []
    logpath = os.path.join(folder, name + ".txt")
    print("All logs will be saved to %s" % logpath)

    logging.root.setLevel(level)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logfile = logging.FileHandler(logpath)
    logfile.setLevel(level)
    logfile.setFormatter(formatter)
    logging.root.addHandler(logfile)

    if not no_console:
        logconsole = logging.StreamHandler()
        logconsole.setLevel(console_level)
        logconsole.setFormatter(formatter)
        logging.root.addHandler(logconsole)
    return folder

def early_stopping(recall_list, stopping_steps,min_epoch):
    best_recall = max(recall_list)
    best_step = recall_list.index(best_recall)
    if len(recall_list) - best_step - 1 >= stopping_steps and min_epoch>60 :
        should_stop = True
    else:
        should_stop = False
    return best_recall, should_stop

class M2(nn.Module):
    def __init__(self):
        super(M2, self).__init__()

        self.fea_dim = 86
        self.fc1 = nn.Sequential(nn.Linear(800, 2048), nn.BatchNorm1d(2048), nn.Dropout(0.2), nn.ReLU(True))
        
        self.gru1 = nn.GRUCell(800, 800)
        self.gru2 = nn.GRUCell(800, 800)

        self.fc = nn.Sequential(nn.Linear(2048, 800), nn.BatchNorm1d(800), nn.Dropout(0.3), nn.ReLU(True))
        self.fce = nn.Sequential(nn.Linear(400, 400), nn.BatchNorm1d(400), nn.Dropout(0.1), nn.ReLU(True))
        self.fc_1 = nn.Sequential(nn.Linear(1200, 800), nn.BatchNorm1d(800), nn.Dropout(0.2), nn.ReLU(True))

        self.softmax=nn.Softmax(dim=1)

        self.fc2 = nn.Sequential(nn.Linear(2048, self.fea_dim))

    def fusion(self, batch_data):
        emb1_1, emb1_2 = [], []
        emb2= []
        emb3_1, emb3_2 = [], []
        emb4_1, emb4_2 = [], []

        for i in batch_data:
            emb1_1.append(drug_emb1[i[0]])
            emb1_2.append(drug_emb1[i[1]])
            emb2.append([*drug_emb2[i[0]],*drug_emb2[i[1]]])
            emb3_1.append(drug_emb3[i[0]])
            emb3_2.append(drug_emb3[i[1]])
            emb4_1.append(drug_emb4[i[0]])
            emb4_2.append(drug_emb4[i[1]])
        emb1_1t = torch.Tensor(emb1_1)
        emb1_2t = torch.Tensor(emb1_2)
        emb2t = torch.Tensor(emb2)
        emb3_1t = torch.Tensor(emb3_1)
        emb3_2t = torch.Tensor(emb3_2)
        emb4_1t = torch.Tensor(emb4_1)
        emb4_2t = torch.Tensor(emb4_2)

        size=emb1_1t.size(0)
        ft_1=torch.cat((emb1_1t,emb3_1t,emb4_1t),0).to('cuda')
        ft_2=torch.cat((emb1_2t,emb3_2t,emb4_2t),0).to('cuda')
        ft_1=self.fce(ft_1)
        ft_2=self.fce(ft_2)

        ft1_1=torch.cat((ft_1[:size],ft_2[:size]-ft_1[:size],ft_1[size:2*size]),1)#n*1200
        ft1_2=torch.cat((ft_2[:size],ft_2[:size]-ft_2[:size],ft_2[size:2*size]),1)#n*1200

        ft2_1 = torch.cat((ft_1[:size], ft_2[:size] - ft_1[:size], ft_1[2*size:]), 1)#n*1200
        ft2_2 = torch.cat((ft_2[:size], ft_2[:size] - ft_2[:size], ft_2[2*size:]), 1)#n*1200

        ft1=ft1_1+ft1_2
        ft2=ft2_1+ft2_2
        ft1 = self.fc_1(ft1)
        ft2=self.fc_1(ft2)#n*1200->n*800

        sf = self.fc(emb2t.to('cuda'))
        ft=torch.stack((ft1,ft2),1)#n*2*800
        ft_0=torch.sum(ft,dim=1)#n*800
        ft_1=self.gru1(ft_0,ft1)
        ft_2=self.gru1(sf,ft_1)
        ft_3=self.gru2(ft_0,ft2)
        ft_4=self.gru2(sf,ft_3)
        ft3=torch.stack((ft_2,ft_4),1)
        attention=self.softmax(ft3)
        feature=torch.sum(ft*attention,dim=1)#n*2*800->n*800
        out=self.fc2(self.fc1(feature))
        return out
        

    def train_DDI_data(self, mode, train_data):
        x = self.fusion(train_data)
        return x

    def test_DDI_data(self, mode, test_data):
        x = self.fusion(test_data)
        sm = nn.Softmax(dim=1)
        pre = sm(x)
        return pre

    def forward(self, mode, *input):
        if mode == 'train':
            return self.train_DDI_data(mode, *input)
        if mode == 'test':
            return self.test_DDI_data(mode, *input)


def calc_metrics(y_true, y_pred, pred_score):

    acc = accuracy_score(y_true, y_pred)
    macro_precision = precision_score(y_true, y_pred, average='macro')
    macro_recall = recall_score(y_true, y_pred, average='macro')
    macro_f1 = f1_score(y_true, y_pred, average='macro')
    kappa = cohen_kappa_score(y_true, y_pred)
    y_true_bi=F.one_hot(y_true.to(torch.int64),num_classes=86)

    auc_ = roc_auc_score(y_true_bi, pred_score)
    aupr = average_precision_score(y_true_bi, pred_score)
    return acc, macro_precision, macro_recall, macro_f1, kappa, auc_, aupr
    

def pred_tru(loader_test, model):
    with torch.no_grad():
        for i, data in enumerate(loader_test):
            test_x_map = data[0]
            test_x = []
            for k in range(len(test_x_map[0])):
                dp = (test_x_map[0][k], test_x_map[1][k])
                test_x.append(dp)

            if i == 0:
                test_y = data[1]
            else:
                test_y = torch.cat((test_y, data[1]), 0)
            
            out1 = model('test', test_x)
            if i == 0:
                out = out1
            else:
                out = torch.cat((out, out1), 0)
    return out, test_y

def evaluate(loader_test, model):
    model.eval()

    with torch.no_grad():
        out, test_y = pred_tru(loader_test, model)

        prediction = torch.max(out, 1)[1]
        prediction = prediction.cuda().data.cpu().numpy()
        out = out.cuda().data.cpu().numpy()

        acc, macro_precision, macro_recall, macro_f1, kappa, auc_, aupr = calc_metrics(test_y, prediction, out)
        return macro_precision, macro_recall, macro_f1, acc, kappa, auc_, aupr
        

def pos_weight():
    data1 = []
    with open(filename[0]) as f2:
        data2 = csv.reader(f2)
        header=next(data2)
        for i, j in data2:
            data1.append(eval(j))
    data3 = torch.Tensor(data1)
    num = data3.size(0)
    posn = torch.sum(data3, 0)
    numn = torch.full_like(posn, num)
    pos_weight = torch.div(numn - posn, posn).to('cuda')
    return pos_weight

def Train(batch_size=2000,n_epoch=200):
    random.seed(2021)
    np.random.seed(2021)
    torch.manual_seed(2021)

    save_dir = 'log\\'

    logging_config(folder=save_dir, name='mc_log', no_console=False)

    use_cuda = torch.cuda.is_available()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    n_gpu = torch.cuda.device_count()
    if n_gpu > 0:
        torch.cuda.manual_seed_all(2021)

    macro_precision_list = []
    macro_recall_list = []
    macro_f1_list = []
    acc_list = []
    kappa_list=[]
    auc_list = []
    aupr_list = []

   #5-fold cross validation：
    for i in range(5):
        time0 = time.time()
        #'''
        model=M2()
        model.to(device)
        logging.info(model)

        optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
        scheduler = MultiStepLR(optimizer, milestones=[50], gamma=0.1)

        trainset=[]
        trainx = trainX[i]
        trainy = trainY[i]
        for j in range(len(trainx)):
            trainset.append([trainx[j],trainy[j]])
            
        loader_train=Data.DataLoader(dataset=trainset,batch_size=batch_size,shuffle=True)
        
        testset = []
        testx = testX[i]
        testy = testY[i]
        for j in range(len(testx)):
            testset.append([testx[j], testy[j]])
            
        loader_test = Data.DataLoader(dataset=testset, batch_size=batch_size, shuffle=True)

        best_epoch = -1
        val_list = []
        for epoch in range(1,n_epoch+1):
            model.train()

            ddi_total_loss=0
            for step,tdata in enumerate(loader_train):
                iter=step+1
                time2=time.time()
                train_x_map = tdata[0]
                train_y=tdata[1]               

                if use_cuda:
                    train_y = train_y.to(device)
                train_x = []
                for ii in range(len(train_x_map[0])):
                    dp = (train_x_map[0][ii], train_x_map[1][ii])
                    train_x.append(dp)
                out=model('train',train_x)

                loss_func = torch.nn.CrossEntropyLoss()
                loss = loss_func(out, train_y.long())

                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                ddi_total_loss+=loss.item()
                if (iter%100)==0:
                    logging.info('DDI Training: Epoch {:04d} Iter {:04d} | Time {:.1f}s | Iter Loss {:.4f} | Iter Mean Loss {:.4f}'
                            .format(epoch, iter, time.time() - time2, loss.item(), ddi_total_loss / iter))
            scheduler.step()

            time3 = time.time()

            macro_precision, macro_recall, macro_f1, acc, kappa, auc_, aupr = evaluate(loader_test, model)
            logging.info(
                'DDI Evaluation:Total Time {:.1f}s | Macro Precision {:.4f} | Macro Recall {:.4f} | Macro F1 {:.4f} | ACC {:.4f} | Kappa {:.4f} | AUC {:.4f} | AUPR {:.4f}'
                    .format(time.time() - time3, macro_precision, macro_recall, macro_f1, acc, kappa, auc_, aupr))
            val_list.append(macro_precision+macro_recall+macro_f1+acc+ kappa+ auc_+ aupr)
            best_acc, should_stop = early_stopping(val_list, 10,epoch)

            if should_stop:
                PATH ='best_model_epoch.pth'
                model.load_state_dict(torch.load(PATH))
                model.to(device)
                macro_precision, macro_recall, macro_f1, acc, kappa, auc_, aupr = evaluate(loader_test, model)
                logging.info(
                    'Final DDI Evaluation:Macro Precision {:.4f} | Macro Recall {:.4f} | Macro F1 {:.4f} | ACC {:.4f} | Kappa {:.4f} | AUC {:.4f} | AUPR {:.4f}'
                        .format(macro_precision, macro_recall, macro_f1, acc, kappa, auc_, aupr))
                break
            if val_list.index(best_acc) == len(val_list) - 1:
                PATH = 'best_model_epoch.pth'
                torch.save(model.state_dict(), PATH)
                best_epoch = epoch


        #test:
        PATH = 'best_model_epoch.pth'
        
        model.load_state_dict(torch.load(PATH))
        model.to(device)
        
        time5 = time.time()
        macro_precision, macro_recall, macro_f1, acc, kappa, auc_, aupr = evaluate(loader_test, model)
        logging.info(
            'DDI Test:Total Time {:.1f}s | Macro Precision {:.4f} | Macro Recall {:.4f} | Macro F1 {:.4f} | ACC {:.4f} | Kappa {:.4f} | AUC {:.4f} | AUPR {:.4f}'
                .format(time.time() - time5, macro_precision, macro_recall, macro_f1, acc, kappa, auc_, aupr))
        macro_precision_list.append(macro_precision)
        macro_recall_list.append(macro_recall)
        macro_f1_list.append(macro_f1)
        acc_list.append(acc)
        kappa_list.append(kappa)
        auc_list.append(auc_)
        aupr_list.append(aupr)

        logging.info('Training+Evaluation:Total Time {:.1f}s '.format(time.time() - time0))

    mean_macro_precision=np.mean(macro_precision_list)
    mean_macro_recall=np.mean(macro_recall_list)
    mean_macro_f1=np.mean(macro_f1_list)
    mean_acc = np.mean(acc_list)
    mean_kappa=np.mean(kappa_list)
    mean_auc = np.mean(auc_list)
    mean_aupr = np.mean(aupr_list)
    logging.info('5-fold cross validation DDI Mean Evaluation: Macro Precision {:.4f} | Macro Recall {:.4f} | Macro F1 {:.4f} | ACC {:.4f} | Kappa {:.4f} | AUC {:.4f} | AUPR {:.4f}'
            .format(mean_macro_precision, mean_macro_recall, mean_macro_f1,mean_acc,mean_kappa,mean_auc,mean_aupr))



if __name__=='__main__':


    filename = ['Multi-Class-data1\Multi-Class Dataset.csv',
                     'input-features\drug initial embedding representations.csv','input-features\Morgan fingerprint vectors.csv',
                    'input-features\drug subgraph mean representations.csv','input-features\drug subgraph frequency representations.csv','input-features\extra label vectors.csv']
    

    data_list, drugs = get_drugpair_info(filename[0], [], [])
    drug_emb1 = feature_vector(filename[1], drugs)
    drug_emb2 = feature_vector(filename[2], drugs)
    drug_emb3 = feature_vector(filename[3], drugs)
    drug_emb4 = feature_vector(filename[4], drugs)

    trainX, trainY, testX, testY = train_test_data1(data_list)

    Train(batch_size=1024, n_epoch=1000)
