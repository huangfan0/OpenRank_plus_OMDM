import torch
import pandas as pd
from torch_geometric.data import HeteroData
from torch_geometric.data import InMemoryDataset, download_url

class GithubDataset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None):
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])
        # For PyG<2.4:
        #self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ['pr.csv', 'issue.csv', 'push.csv','folk.csv','release.csv','following.csv','follower.csv',
                'actor_activity.csv','repo_embedding.csv','repo_label.csv','actor_label.csv']

    @property
    def processed_file_names(self):
        return ['data.pt']

    def download(self):
        # Download to `self.raw_dir`.
        pass
        ...

    def process(self):
        # Read data into huge `Data` list.
        data_list = []
        path='/home/huangfan/GNN/a1_myode/dataset/data/'
        dev_data = pd.read_csv(path+'actor_activity.csv', header=None)
        repo_data = pd.read_csv(path+'repo_embedding.csv', header=None)
        pr_data = pd.read_csv(path+'pr.csv', header=None)
        issue_data = pd.read_csv(path+'issue.csv', header=None)
        folk=pd.read_csv(path+'folk.csv',header=None)
        push=pd.read_csv(path+'push.csv',header=None)
        releases=pd.read_csv(path+'release.csv',header=None)
        follow=pd.read_csv(path+'follow.csv',header=None)
        repo_label=pd.read_csv(path+'repo_label.csv',header=None)
        dev_label=pd.read_csv(path+'actor_label.csv',header=None)
        pr_count=pd.read_csv(path+'pr_count_modify3.csv',header=None)
        issue_count=pd.read_csv(path+'issue_count_modify3.csv',header=None)
        folk_count=pd.read_csv(path+'folk_count_modify3.csv',header=None)
        push_count=pd.read_csv(path+'push_count_modify3.csv',header=None)
        release_count=pd.read_csv(path+'release_count_modify3.csv',header=None)
        follow_count=pd.read_csv(path+'follow_edge_prob.csv',header=None)

        

        repo_label=repo_label.sort_values(by=repo_label.columns[0])[1].values
        dev_label=dev_label.sort_values(by=dev_label.columns[0])[1].values
        
        node_types = {'developer': 0, 'repository': 1}
        node_index = {'developer': {}, 'repository': {}}

        
        edges = {('developer', 'pr', 'repository'): [], ('developer', 'issue', 'repository'): [],
                    ('developer','folk','repository'):[],('developer','push','repository'):[],
                    ('developer','release','repository'):[],('developer','following','developer'):[]}
        edge_count={'pr':[],'issue':[],'folk':[],'push':[],'release':[],'follow':[]}
        
        for dev_id in dev_data[0]:
            if dev_id not in node_index['developer']:
                node_index['developer'][dev_id] = len(node_index['developer'])
            
            #nodes.append(('developer', dev_id))

        
        for repo_id in repo_data[0]:
            if repo_id not in node_index['repository']:
                node_index['repository'][repo_id] = len(node_index['repository'])
            
            #nodes.append(('repository', repo_id))

        
        i=0
        pr_count=pr_count.values.tolist()
        for row in pr_data.itertuples(index=False):
            dev_id, repo_id = row
            if dev_id in node_index['developer'] and repo_id in node_index['repository']:
                edges[('developer', 'pr', 'repository')].append((node_index['developer'][dev_id], node_index['repository'][repo_id]))
                edge_count['pr'].append(pr_count[i])
            i+=1
            # else:
            #     print('pr ',dev_id,repo_id)
       
        i=0
        issue_count=issue_count.values.tolist()
        for row in issue_data.itertuples(index=False):
            dev_id, repo_id = row
            if dev_id in node_index['developer'] and repo_id in node_index['repository']:
                edges[('developer', 'issue', 'repository')].append((node_index['developer'][dev_id], node_index['repository'][repo_id]))
                edge_count['issue'].append(issue_count[i])
            i+=1
            # else:
            #     print('issue ',dev_id,repo_id)
        i=0
        folk_count=folk_count.values.tolist()
        for row in folk.itertuples(index=False):
            dev_id, repo_id = row
            if dev_id in node_index['developer'] and repo_id in node_index['repository']:
                edges[('developer', 'folk', 'repository')].append((node_index['developer'][dev_id], node_index['repository'][repo_id]))
                edge_count['folk'].append(folk_count[i])
            i+=1
            # else:
            #     print('folk ',dev_id,repo_id)
        i=0
        push_count=push_count.values.tolist()
        for row in push.itertuples(index=False):
            dev_id, repo_id = row
            if dev_id in node_index['developer'] and repo_id in node_index['repository']:
                edges[('developer', 'push', 'repository')].append((node_index['developer'][dev_id], node_index['repository'][repo_id]))
                edge_count['push'].append(push_count[i])
            i+=1
            # else:
            #     print('push ',dev_id,repo_id)
        i=0
        release_count=release_count.values.tolist()
        for row in releases.itertuples(index=False):
            dev_id, repo_id = row
            if dev_id in node_index['developer'] and repo_id in node_index['repository']:
                edges[('developer', 'release', 'repository')].append((node_index['developer'][dev_id], node_index['repository'][repo_id]))
                edge_count['release'].append(release_count[i])
            i+=1
            # else:
            #     print('release ',dev_id,repo_id)
        i=0
        follow_count=follow_count.values.tolist()
        for row in follow.itertuples(index=False):
            dev_id1, dev_id2 = row
            if dev_id1 in node_index['developer'] and dev_id2 in node_index['developer']:
                edges[('developer', 'following', 'developer')].append((node_index['developer'][dev_id1], node_index['developer'][dev_id2]))
                edge_count['follow'].append(follow_count[i])
            i+=1


        
        
        x_dict = {'developer': torch.tensor(dev_data.iloc[:,1:].values),
                'repository': torch.tensor(repo_data.iloc[:,1:].values)}

        
        #edge_index_dict = {key: SparseTensor(row=value[0],col=value[1],value=value) for key, value in edges.items()}
        edge_index_dict = {key: torch.tensor(value, dtype=torch.long).t() for key, value in edges.items()}

        reverse_edge={}
        relations_to_convert = [
        ('developer', 'pr', 'repository'),('developer', 'issue', 'repository'),('developer','folk','repository'),
        ('developer','push','repository'),('developer','release','repository'),
        ]
        new_relation=[('repo','re_pr','dev'),('repo','re_issue','dev'),('repo','re_folk','dev'),('repo','re_push','dev'),('repo','re_release','dev')]
        for i,relation in enumerate(relations_to_convert):
            edge= edge_index_dict[relation]
            reverse_edge_index = torch.stack([edge[1], edge[0]], dim=0) 
            reverse_edge[new_relation[i]]=reverse_edge_index

        
        
        data=HeteroData()
        data['repo'].x=x_dict['repository'].to(torch.float32)
        data['dev'].x=x_dict['developer'].to(torch.float32)
        data['repo'].y=torch.from_numpy(repo_label).to(torch.float32)
        data['dev'].y=torch.from_numpy(dev_label).to(torch.float32)
        data['dev', 'pr', 'repo'].edge_index=edge_index_dict[('developer', 'pr', 'repository')]
        data['dev', 'issue', 'repo'].edge_index=edge_index_dict[('developer', 'issue', 'repository')]
        data['dev','folk','repo'].edge_index=edge_index_dict[('developer','folk','repository')]
        data['dev','push','repo'].edge_index=edge_index_dict[('developer','push','repository')]
        data['dev','release','repo'].edge_index=edge_index_dict[('developer','release','repository')]
        data['dev','follow','dev'].edge_index=edge_index_dict[('developer','following','developer')]
        data['repo','re_pr','dev'].edge_index=reverse_edge[(('repo','re_pr','dev'))]
        data['repo','re_issue','dev'].edge_index=reverse_edge[('repo','re_issue','dev')]
        data['repo','re_folk','dev'].edge_index=reverse_edge[('repo','re_folk','dev')]
        data['repo','re_push','dev'].edge_index=reverse_edge[('repo','re_push','dev')]
        data['repo','re_release','dev'].edge_index=reverse_edge[('repo','re_release','dev')]

        data['dev', 'pr', 'repo'].edge_attr=torch.tensor(edge_count['pr'],dtype=torch.float32)
        data['dev', 'issue', 'repo'].edge_attr=torch.tensor(edge_count['issue'],dtype=torch.float32)
        data['dev','folk','repo'].edge_attr=torch.tensor(edge_count['folk'],dtype=torch.float32)
        data['dev','push','repo'].edge_attr=torch.tensor(edge_count['push'],dtype=torch.float32)
        data['dev','release','repo'].edge_attr=torch.tensor(edge_count['release'],dtype=torch.float32)
        data['dev','follow','dev'].edge_attr=torch.tensor(edge_count['follow'],dtype=torch.float32)
        data['repo','re_pr','dev'].edge_attr=torch.tensor(edge_count['pr'],dtype=torch.float32)
        data['repo','re_issue','dev'].edge_attr=torch.tensor(edge_count['issue'],dtype=torch.float32)
        data['repo','re_folk','dev'].edge_attr=torch.tensor(edge_count['folk'],dtype=torch.float32)
        data['repo','re_push','dev'].edge_attr=torch.tensor(edge_count['push'],dtype=torch.float32)
        data['repo','re_release','dev'].edge_attr=torch.tensor(edge_count['release'],dtype=torch.float32)
       
        #print(data)
        data_list.append(data)

        #self.save(data_list, self.processed_paths[0])
        # For PyG<2.4:
        torch.save(self.collate(data_list), self.processed_paths[0])
# dataset=GithubDataset('/root/data')
# print(dataset[0])