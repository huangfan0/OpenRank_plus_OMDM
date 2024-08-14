import os
import torch
import numpy as np
from torch_geometric.utils import negative_sampling
from github_dataset import GithubDataset
from torch_geometric.transforms import ToUndirected, RandomLinkSplit
from torch_geometric.loader import NeighborLoader,LinkNeighborLoader
# from gcn_model import HeteroGNN
# from hgt_model import HeteroGNN
from model import HeteroGNN
# from HAN_model_modify_linkhead import HeteroGNN
# from HAT_model import HeteroGNN
import pandas as pd
from tqdm import tqdm
from utility import custom_negative_sampling,combineloss,edge_generation
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.metrics import precision_recall_curve, average_precision_score
from sklearn.metrics import f1_score
# os.environ["CUDA_VISIBLE_DEVICES"] = "6"
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

print('han link prediction')
dataset=GithubDataset('/home/ubuntu/MYCODE/DATA/data')
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
data = dataset[0].to(device)

edges=None
for edge in data.edge_index_dict:
    if edges is None:
        edges=data.edge_index_dict[edge]
    elif edge != ('dev', 'follow', 'dev') :
        edges=torch.cat((edges,data.edge_index_dict[edge]),dim=1)
edges=edges.t().detach().cpu().numpy()
edges_lists = edges.tolist()
edges_tuples = [tuple(row) for row in edges_lists]
edges=set(edges_tuples)
# edge=pd.DataFrame(edges)
# edge=edge.sort_values(by=[0,1],ascending=[True,True])
# edge.to_csv('/home/flbi/GNN/mycode/baseline/dataset/process_data/current_dev_repo_edge.csv',header=False,index=False)


train_data, val_data, test_data = RandomLinkSplit(
    num_val=0.1,
    num_test=0.1,
    neg_sampling_ratio=0.0,
    edge_types=[('dev', 'pr', 'repo'), ('dev', 'issue', 'repo'),
                ('dev','folk','repo'),('dev','push','repo'),('dev','release','repo'),
                ('dev','follow','dev'),
                ('repo','re_pr','dev'),('repo','re_issue','dev'),('repo','re_folk','dev'),
                ('repo','re_push','dev'),('repo','re_release','dev')],
)(data)
loader = NeighborLoader(
    train_data,
    num_neighbors=[20, 18],  
    input_nodes=('dev', None), 
    batch_size=1024, 
    shuffle=True,  
)
val_loader=NeighborLoader(
    data=val_data,
    num_neighbors=[30, 20],  # Reduce neighbor count
    input_nodes=('dev', None),
    batch_size=1024,
    shuffle=True,
)
test_loader=NeighborLoader(
    data=test_data,
    num_neighbors=[30, 20],  # Reduce neighbor count
    input_nodes=('dev', None),
    batch_size=1024,
    shuffle=True,
)
hidden_channels=512
out_channels=128 
num_layers=3
lambda_l2 =0.5
lr=0.01
model=HeteroGNN(data.metadata(), hidden_channels, out_channels, num_layers)
model=model.to(device)

optimizer = torch.optim.SGD(model.parameters(), lr=lr)
criterion = combineloss()
print("hidden_channels: "+str(hidden_channels), 'out_channels: '+str(out_channels), 'lr: '+str(lr))
def test(test_loader):
    for i,data in enumerate(test_loader):
        #print(data)
        auc_scores=[]
        pr_auc_scores=[]
        f1s=[]
        for edge_name in data.edge_types:
            pos_edge,neg_edge,label=edge_generation(data,edge_name,edges,device)
            label=label.detach().cpu().numpy()
            if edge_name != ('dev', 'follow', 'dev'):
                openrank_dev,openrank_repo,score=model(data.x_dict, data.edge_index_dict,pos_edge,neg_edge)
            else:
                openrank_dev,openrank_repo,score=model(data.x_dict, data.edge_index_dict,pos_edge,neg_edge,follow=True)
            sorted_score=score.detach().cpu().numpy()[:]
            sorted_score.sort()
            true_num=len(pos_edge[0])
            threshod=sorted_score[-true_num]
            y_pred=torch.zeros(len(score),dtype=torch.float32)
            for i in range(len(score)):
                if score[i]>=threshod:
                    y_pred[i]=1
            fpr, tpr, _ = roc_curve(label, y_pred)
            auc_score = roc_auc_score(label,y_pred)

            precision, recall, _ = precision_recall_curve(label, y_pred)
            pr_auc_score = average_precision_score(label,y_pred)

            threshold = 0.5
            pred_labels = [1 if scores > threshold else 0 for scores in y_pred]

            f1 = f1_score(label, y_pred)
            auc_scores.append(auc_score)
            pr_auc_scores.append(pr_auc_score)
            f1s.append(f1)
    print(f' roc:  {auc_scores}, pr: {pr_auc_scores}, f1 {f1s}')

def train():
    for epoch in range(100):
        total_loss=0
        l2_reg=0
        for data in loader:
            for edge_name in data.edge_types:
                if edge_name != ('dev', 'follow', 'dev'):
                    optimizer.zero_grad()
                    y_dev=data['dev'].y
                    y_repo=data['repo'].y
                    pos_edge,neg_edge,true_label=edge_generation(data,edge_name,edges,device)
                    openrank_dev,openrank_repo,score=model(data.x_dict, data.edge_index_dict,pos_edge,neg_edge)
                    loss=criterion(openrank_dev,openrank_repo,y_dev,y_repo,score,true_label)
                    loss.backward()
                    optimizer.step()
                    total_loss+=loss.item()
                    loss=loss.detach()
                else:
                    optimizer.zero_grad()
                    y_dev=data['dev'].y
                    y_repo=data['repo'].y
                    pos_edge,neg_edge,true_label=edge_generation(data,edge_name,edges,device)
                    openrank_dev,openrank_repo,score=model(data.x_dict, data.edge_index_dict,pos_edge,neg_edge,follow=True)
                    loss=criterion(openrank_dev,openrank_repo,y_dev,y_repo,score,true_label)
                    loss.backward()
                    optimizer.step()
                    total_loss+=loss.item()
                    loss=loss.detach()

        print(f'Epoch:{epoch},losss:{total_loss/len(loader)}')

        if epoch%30==0:
            test(val_loader)
# train()
train()
test(test_loader)
