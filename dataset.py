from tqdm import tqdm
import numpy as np, networkx as nx, time
import pickle, argparse
import os, spacy
import scipy.sparse as sp
from gensim.corpora import Dictionary as gensim_dico

import torch
import torch.nn.functional as F

from torch_geometric.data import Data, DataLoader
from torch_geometric.data import Batch as GeometricBatch

from torch_scatter import scatter_add

from rgcn import RGCN
from utils import (get_dataset,
                    get_dataset_path,
                    get_domain_dataset,
                    parse_processed_amazon_dataset,
                    spacy_seed_concepts,
                    spacy_seed_concepts_list)
from utils_graph import conceptnet_graph, subgraph_dense, unique_rows
from train_and_extract_graph_features import (negative_sampling,
                                              edge_normalization)

pkl_path = '/media/disk1/jennybae/data/kingdom/pkl_files/data2000'
sess_path = '/media/disk1/jennybae/kingdom/'

norm = {'booksdvd': 4.18, 'bookskitchen': 4.13, 'bookselectronics': 4.13,
        'electronicskitchen': 3.56, 'electronicsdvd': 4.18, 'electronicsbooks': 4.45,
        'kitchenbooks': 4.45, 'kitchenelectronics': 3.5, 'kitchendvd': 4.18,
        'dvdelectronics': 3.62, 'dvdkitchen': 3.62, 'dvdbooks': 4.45}

def generate_sampled_graph_and_labels(triplets, sample_size, split_size, num_entity, num_rels, negative_rate):
    """
        Get training graph and labels with negative sampling.
    """
    edges = triplets
    src, rel, dst = edges.transpose()
    uniq_entity, edges = np.unique((src, dst), return_inverse=True)
    src, dst = np.reshape(edges, (2, -1)) # unique node index in mini-batch
    relabeled_edges = np.stack((src, rel, dst)).transpose()

    # Negative sampling
    samples, labels = negative_sampling(relabeled_edges, len(uniq_entity), negative_rate)

    # further split graph, only half of the edges will be used as graph
    # structure, while the rest half is used as unseen positive samples
    split_size = int(sample_size * split_size)
    graph_split_ids = np.random.choice(np.arange(sample_size),
                                       size=split_size, replace=False)

    src = torch.tensor(src[graph_split_ids], dtype = torch.long).contiguous()
    dst = torch.tensor(dst[graph_split_ids], dtype = torch.long).contiguous()
    rel = torch.tensor(rel[graph_split_ids], dtype = torch.long).contiguous()

    # Create bi-directional graph
    src, dst = torch.cat((src, dst)), torch.cat((dst, src))
    rel = torch.cat((rel, rel + num_rels))

    edge_index = torch.stack((src, dst))
    edge_type = rel

    data = Data(edge_index = edge_index)
    data.entity = torch.from_numpy(uniq_entity)
    data.num_nodes = len(uniq_entity)
    data.edge_type = edge_type
    data.edge_norm = edge_normalization(edge_type, edge_index, len(uniq_entity), num_rels)
    # len(uniq_entity): subgraph 에서 다시 node labeling 을 했으므로 subgraph 에 한하여 normalization
    data.samples = torch.from_numpy(samples)
    data.labels = torch.from_numpy(labels)

    return data, uniq_entity

def pyg_collate(data):
    graphs = []
    for id, d in enumerate(data):
        graphs.append(d["graph_data"])
    source_nodes = [ins["source_node"] for ins in data]
    target_nodes = [ins["target_node"] for ins in data]

    source_bow = torch.tensor([ins["source_bow"] for ins in data])
    source_label = torch.tensor([ins["source_label"] for ins in data])
    target_bow = torch.tensor([ins["target_bow"] for ins in data])
    return GraphBatch(
        graphs=graphs, s_nodes=source_nodes, t_nodes=target_nodes,
        s_bow=source_bow, s_label=source_label, t_bow=target_bow
    )

class GraphBatch:
    def __init__(self, graphs, s_nodes, t_nodes, s_bow, s_label, t_bow):
        self.num_nodes = [g.num_nodes for g in graphs]
        self.graphs = GeometricBatch.from_data_list(graphs)
        self.s_nodes = s_nodes
        self.t_nodes = t_nodes
        self.s_bow = torch.tensor(s_bow)
        self.s_label = torch.tensor(s_label).long()
        self.t_bow = torch.tensor(t_bow)

    def to(self, device):
        self.graphs = self.graphs.to(device)
        self.s_bow = self.s_bow.to(device)
        self.s_label = self.s_label.to(device)
        self.t_bow = self.t_bow.to(device)


def pyg_collate(data):
    graphs = []
    for id, d in enumerate(data):
        graphs.append(d["graph_data"])
    source_nodes = [ins["source_node"] for ins in data if "source_node" in ins]
    target_nodes = [ins["target_node"] for ins in data if "target_node" in ins]

    source_bow = torch.tensor([ins["source_bow"] for ins in data if "source_bow" in ins])
    source_label = torch.tensor([ins["source_label"] for ins in data if "source_label" in ins])
    target_bow = torch.tensor([ins["target_bow"] for ins in data if "target_bow" in ins])
    return GraphBatch(
        graphs=graphs, s_nodes=source_nodes, t_nodes=target_nodes,
        s_bow=source_bow, s_label=source_label, t_bow=target_bow
    )


class TrainDataset(torch.utils.data.Dataset):
    def __init__(self, domain1, domain2, kg_name, bow_size=5000, transform=True):
        self.domain1 = domain1
        self.domain2 = domain2
        self.kg_name = kg_name
        self.bow_size = bow_size
        self.pkl_path = os.path.join(pkl_path, self.kg_name)

        self.all_seeds = pickle.load(open(os.path.join(self.pkl_path, 'all_seeds.pkl'), 'rb'))
        self.concept_map = pickle.load(open(os.path.join(self.pkl_path, 'concept_map.pkl'), 'rb'))
        self.unique_nodes_mapping = pickle.load(open(os.path.join(self.pkl_path, 'unique_nodes_mapping.pkl'), 'rb'))
        self.concept_graphs = pickle.load(open(os.path.join(self.pkl_path, 'concept_graphs.pkl'), 'rb'))
        self.relation_map = pickle.load(open(os.path.join(self.pkl_path, 'relation_map.pkl'), 'rb'))
        # self.word_index = pickle.load(open(os.path.join(self.pkl_path, 'word_index.pkl'), 'rb'))
        # train_triplets = np.load(open(os.path.join(self.kl_path, 'triplets.np'), 'rb'), allow_pickle=True)

        self.source, self.s_dico = get_domain_dataset(self.domain1, exp_type='small')
        self.target1, self.t_dico = get_domain_dataset(self.domain2, exp_type='small')

        self.X_s, self.Y_s, self.X_t1, self.Y_t1, self.X_t2, self.Y_t2, _ = get_dataset(self.domain1, self.domain2,
                                                                        max_words=self.bow_size, exp_type='small')
        if transform:
            c = norm[self.domain1 + self.domain2]
            self.X_s = np.log(1 + np.array(self.X_s.todense()).astype('float32'))/c
            self.X_t1 = np.log(1 + np.array(self.X_t1.todense()).astype('float32'))/c
            self.X_t2 = np.log(1 + np.array(self.X_t2.todense()).astype('float32'))/c

        else:
            self.X_s = np.array(self.X_s.todense()).astype('float32')
            self.X_t1 = np.array(self.X_t1.todense()).astype('float32')
            self.X_t2 = np.array(self.X_t2.todense()).astype('float32')

    def __len__(self):
        return len(self.X_s)

    def __getitem__(self, idx):
        s_c = [self.s_dico[item] for item in np.where(self.source[idx] != 0)[0]]  # dico 기준 word list
        s_n = list(spacy_seed_concepts_list(s_c).intersection(set(self.all_seeds)))

        t_c = [self.t_dico[item] for item in np.where(self.target1[idx] != 0)[0]]
        t_n = list(spacy_seed_concepts_list(t_c).intersection(set(self.all_seeds)))

        n = list(set(s_n).union(set(t_n)))

        xg = np.concatenate([self.concept_graphs[item] for item in n])

        xg = xg[~np.all(xg == 0, axis=1)]

        absent1 = set(xg[:, 0]) - self.unique_nodes_mapping.keys()
        absent2 = set(xg[:, 2]) - self.unique_nodes_mapping.keys()
        absent = absent1.union(absent2)

        for item in absent:
            xg = xg[~np.any(xg == item, axis=1)]

        xg[:, 0] = np.vectorize(self.unique_nodes_mapping.get)(xg[:, 0])
        xg[:, 2] = np.vectorize(self.unique_nodes_mapping.get)(xg[:, 2])

        xg = unique_rows(xg).astype('int64')
        graph_data, uniq_entity = generate_sampled_graph_and_labels(xg, sample_size=5000, split_size=0.5,
                                                       num_entity=len(self.unique_nodes_mapping),
                                                       num_rels=len(self.relation_map),
                                                       negative_rate=1)

        x_s1, y_s, x_t1, x_t2 = self.X_s[idx], self.Y_s[idx], self.X_t1[idx], self.X_t2[idx]

        feature = {'graph_data': graph_data,
                   'source_node': np.concatenate([np.where(uniq_entity==self.unique_nodes_mapping[self.concept_map[item]])[0]
                                                  for item in s_n if item in self.concept_map]),
                   'target_node':  np.concatenate([np.where(uniq_entity==self.unique_nodes_mapping[self.concept_map[item]])[0]
                                                  for item in t_n if item in self.concept_map]),
                   'source_bow': x_s1,
                   'source_label': y_s,
                   'target_bow': x_t1}

        return feature




class EvalDataset(torch.utils.data.Dataset):
    def __init__(self, domain1, domain2, kg_name, bow_size=5000, transform=True):
        self.domain1 = domain1
        self.domain2 = domain2
        self.kg_name = kg_name
        self.bow_size = bow_size
        self.pkl_path = os.path.join(pkl_path, self.kg_name)

        self.all_seeds = pickle.load(open(os.path.join(self.pkl_path, 'all_seeds.pkl'), 'rb'))
        self.concept_map = pickle.load(open(os.path.join(self.pkl_path, 'concept_map.pkl'), 'rb'))
        self.unique_nodes_mapping = pickle.load(open(os.path.join(self.pkl_path, 'unique_nodes_mapping.pkl'), 'rb'))
        self.concept_graphs = pickle.load(open(os.path.join(self.pkl_path, 'concept_graphs.pkl'), 'rb'))
        self.relation_map = pickle.load(open(os.path.join(self.pkl_path, 'relation_map.pkl'), 'rb'))
        # self.word_index = pickle.load(open(os.path.join(self.pkl_path, 'word_index.pkl'), 'rb'))
        # train_triplets = np.load(open(os.path.join(self.kl_path, 'triplets.np'), 'rb'), allow_pickle=True)

        self.target2, self.t_dico = get_domain_dataset(self.domain2, exp_type='test')

        self.X_s, self.Y_s, self.X_t1, self.Y_t1, self.X_t2, self.Y_t2, _ = get_dataset(self.domain1, self.domain2,
                                                                                        max_words=self.bow_size,
                                                                                        exp_type='small')
        if transform:
            c = norm[self.domain1 + self.domain2]
            self.X_t2 = np.log(1 + np.array(self.X_t2.todense()).astype('float32'))/c

        else:
            self.X_t2 = np.array(self.X_t2.todense()).astype('float32')

    def __len__(self):
        return len(self.X_t2)

    def __getitem__(self, idx):
        t_c = [self.t_dico[item] for item in np.where(self.target2[idx] != 0)[0]]
        n = list(spacy_seed_concepts_list(t_c).intersection(set(self.all_seeds)))


        xg = np.concatenate([self.concept_graphs[item] for item in n])

        xg = xg[~np.all(xg == 0, axis=1)]

        absent1 = set(xg[:, 0]) - self.unique_nodes_mapping.keys()
        absent2 = set(xg[:, 2]) - self.unique_nodes_mapping.keys()
        absent = absent1.union(absent2)

        for item in absent:
            xg = xg[~np.any(xg == item, axis=1)]

        xg[:, 0] = np.vectorize(self.unique_nodes_mapping.get)(xg[:, 0])
        xg[:, 2] = np.vectorize(self.unique_nodes_mapping.get)(xg[:, 2])

        xg = unique_rows(xg).astype('int64')
        graph_data, uniq_entity = generate_sampled_graph_and_labels(xg, sample_size=len(xg), split_size=0.5,
                                                       num_entity=len(self.unique_nodes_mapping),
                                                       num_rels=len(self.relation_map),
                                                       negative_rate=1)

        x_t2, y_t2 = self.X_t2[idx], self.Y_t2[idx]

        feature = {'graph_data': graph_data,
                   'target_node': np.concatenate([np.where(uniq_entity==self.unique_nodes_mapping[self.concept_map[item]])[0]
                                                  for item in n if item in self.concept_map]),
                   'target_bow': x_t2,
                   'target_label': y_t2}

        return feature
