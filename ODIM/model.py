import torch
import numpy as np
import torch.nn.functional as F
from torch_geometric.nn import HeteroConv, Linear, SAGEConv,HGTConv,HANConv
import torch.nn.init as init
from torch_geometric.nn import MessagePassing

class Hetero_emb1(torch.nn.Module):
    def __init__(self, metadata, hidden_channels, out_channels, num_layers,num_head=8):
        super().__init__()

        self.convs = torch.nn.ModuleList()
        for _ in range(num_layers):
            conv = HGTConv(-1,hidden_channels,metadata,num_head)
            self.convs.append(conv)

        self.lin = Linear(hidden_channels, out_channels)


    def forward(self, x_dict, edge_index_dict):
        for conv in self.convs:
            x_dict = conv(x_dict, edge_index_dict)
            x_dict = {key: F.leaky_relu(x) for key, x in x_dict.items()}
        # x_dict = {key: self.lin(x) for key, x in x_dict.items()}
        # x1=self.regress(self.lin(x_dict['dev']))
       
        return x_dict
    
class Hetero_emb2(torch.nn.Module):
    def __init__(self, metadata, hidden_channels, out_channels, num_layers,num_head=8):
        super().__init__()

        self.convs = torch.nn.ModuleList()
        for _ in range(num_layers):
            conv = HGTConv(-1,hidden_channels,metadata,num_head)
            self.convs.append(conv)

        self.lin = Linear(hidden_channels, out_channels)


    def forward(self, x_dict, edge_index_dict):
        for conv in self.convs:
            x_dict = conv(x_dict, edge_index_dict)
            x_dict = {key: F.leaky_relu(x) for key, x in x_dict.items()}
        # x_dict = {key: self.lin(x) for key, x in x_dict.items()}
        # x1=self.regress(self.lin(x_dict['dev']))
        return x_dict
    
class LinkPredictor(torch.nn.Module):
    def __init__(self, hidden_channels,head_channel=256):
        super(LinkPredictor, self).__init__()
        self.lin_src = Linear(hidden_channels, head_channel)
        self.lin_dst = Linear(hidden_channels, head_channel)
        self.final = Linear(head_channel, 1)

    def forward(self, z_src, z_dst):
        h_src = self.lin_src(z_src)
        h_dst = self.lin_dst(z_dst)
        h = h_src * h_dst #/ (torch.norm(h_src) * torch.norm(h_dst))
        # x=torch.sigmoid((torch.sum(h,1)))#torch.log
        x=torch.sigmoid(self.final(h))
        return x
    

class nodeembedding(torch.nn.Module):
    def  __init__(self, hidden_channels,out_channels) :
        super(nodeembedding,self).__init__()
        self.lin = Linear(hidden_channels, out_channels)
        self.regress=Linear(out_channels,1)
    def forward(self,x_dict):
        x=self.regress(self.lin(x_dict))
        return x
    

class HeteroGNN(torch.nn.Module):
    def __init__(self, metadata, hidden_channels, out_channels, num_layers):
        super().__init__()
        self.embedding_res=Hetero_emb1(metadata,hidden_channels,out_channels,num_layers)
        self.embedding_link=Hetero_emb2(metadata,hidden_channels,out_channels,num_layers)
        self.reg=nodeembedding(hidden_channels,out_channels)
        self.link=LinkPredictor(hidden_channels)
        self._reset_parameters()
    def _reset_parameters(self):
        for module in self.modules():
            if isinstance(module, torch.nn.Linear):
         
                init.kaiming_uniform_(module.weight, a=0, mode='fan_in', nonlinearity='leaky_relu')
                if module.bias is not None:
                    init.zeros_(module.bias)
            elif isinstance(module, MessagePassing):
            
                if hasattr(module, 'lin') and module.lin is not None:
               
                    init.kaiming_uniform_(module.lin.weight, a=0, mode='fan_in', nonlinearity='leaky_relu')
                    if module.lin.bias is not None:
                        init.zeros_(module.lin.bias)
                elif hasattr(module, 'root_weight') and isinstance(module.root_weight, torch.Tensor):
              
                    init.kaiming_uniform_(module.root_weight, a=0, mode='fan_in', nonlinearity='leaky_relu')
                    if module.bias is not None:
                        init.zeros_(module.bias)

    def forward(self,x_dict, edge_index_dict,pos_edge,neg_edge=None,follow=False):
        x_emb=self.embedding_res(x_dict,edge_index_dict)
        x_emb_link=self.embedding_link(x_dict,edge_index_dict)

        x_emb_dev=x_emb['dev']
        x_emb_repo=x_emb['repo']
        x_emb_link_dev=x_emb_link['dev']
        x_emb_link_repo=x_emb_link['repo']
        emb_dev=x_emb_dev+x_emb_link_dev
        emb_repo=x_emb_repo+x_emb_link_repo

        fit_openrank_dev=self.reg(emb_dev).squeeze()
        fit_openrank_repo=self.reg(emb_repo).squeeze()
        pos_src_emb=emb_dev[pos_edge[0]]
        pos_dst_emb=emb_repo[pos_edge[1]]
        pos_score=self.link(pos_src_emb,pos_dst_emb).squeeze()
        if neg_edge is None:
            return emb_dev,emb_repo,pos_score
        if follow is True:
            pos_src_emb=emb_dev[pos_edge[0]]
            pos_dst_emb=emb_dev[pos_edge[1]]
            pos_score=self.link(pos_src_emb,pos_dst_emb).squeeze()
            neg_src_emb=emb_dev[neg_edge[0]]
            neg_dst_emb=emb_dev[neg_edge[1]]
            neg_score=self.link(neg_src_emb,neg_dst_emb).squeeze()
            score=torch.cat([pos_score,neg_score],dim=0)
            return fit_openrank_dev,fit_openrank_repo,score
        else:        
            
            neg_src_emb=emb_dev[neg_edge[0]]
            neg_dst_emb=emb_repo[neg_edge[1]]
            neg_score=self.link(neg_src_emb,neg_dst_emb).squeeze()
            score=torch.cat([pos_score,neg_score],dim=0)

            return fit_openrank_dev,fit_openrank_repo,score