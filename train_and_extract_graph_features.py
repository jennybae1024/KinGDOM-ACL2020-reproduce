from tqdm import tqdm
from utils_graph import unique_rows
from utils import get_domain_dataset, spacy_seed_concepts_list
import numpy as np, pickle, argparse

import os
import random
import torch
import torch.nn.functional as F
from rgcn import RGCN
from torch_scatter import scatter_add
from torch_geometric.data import Data
from torch.utils.tensorboard import SummaryWriter

# graph_feat_dir = '/media/disk1/jennybae/data/kingdom/pkl_files'
data_dir = '/media/disk1/jennybae/data/kingdom'
sess_dir = '/media/disk1/jennybae/kingdom/gae'

filename={"conceptnet": "conceptnet_english.txt",
          "wordnet18": "wordnet18.txt"}


def sample_edge_uniform(n_triples, sample_size):
    """Sample edges uniformly from all the edges."""
    all_edges = np.arange(n_triples)
    return np.random.choice(all_edges, sample_size, replace=False)

def negative_sampling(pos_samples, num_entity, negative_rate):
    size_of_batch = len(pos_samples)
    num_to_generate = size_of_batch * negative_rate
    neg_samples = np.tile(pos_samples, (negative_rate, 1))
    labels = np.zeros(size_of_batch * (negative_rate + 1), dtype=np.float32)
    labels[: size_of_batch] = 1
    values = np.random.choice(num_entity, size=num_to_generate)
    choices = np.random.uniform(size=num_to_generate)
    subj = choices > 0.5
    obj = choices <= 0.5
    neg_samples[subj, 0] = values[subj]
    neg_samples[obj, 2] = values[obj]

    return np.concatenate((pos_samples, neg_samples)), labels

def edge_normalization(edge_type, edge_index, num_entity, num_relation):
    """
        Edge normalization trick
        - one_hot: (num_edge, num_relation)
        - deg: (num_node, num_relation)
        - index: (num_edge)
        - deg[edge_index[0]]: (num_edge, num_relation)
        - edge_norm: (num_edge)
    """
    one_hot = F.one_hot(edge_type, num_classes = 2 * num_relation).to(torch.float)
    deg = scatter_add(one_hot, edge_index[0], dim = 0, dim_size = num_entity)
    index = edge_type + torch.arange(len(edge_index[0])) * (2 * num_relation)
    edge_norm = 1 / deg[edge_index[0]].view(-1)[index]

    return edge_norm

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
    data.edge_type = edge_type
    data.edge_norm = edge_normalization(edge_type, edge_index, len(uniq_entity), num_rels)
    # len(uniq_entity): subgraph 에서 다시 node labeling 을 했으므로 subgraph 에 한하여 normalization
    data.samples = torch.from_numpy(samples)
    data.labels = torch.from_numpy(labels)

    return data

def generate_graph(triplets, num_rels):
    """
        Get feature extraction graph without negative sampling.
    """
    edges = triplets
    src, rel, dst = edges.transpose()
    uniq_entity, edges = np.unique((src, dst), return_inverse=True)
    src, dst = np.reshape(edges, (2, -1))
    relabeled_edges = np.stack((src, rel, dst)).transpose()
    
    src = torch.tensor(src, dtype = torch.long).contiguous()
    dst = torch.tensor(dst, dtype = torch.long).contiguous()
    rel = torch.tensor(rel, dtype = torch.long).contiguous()

    # Create bi-directional graph
    src, dst = torch.cat((src, dst)), torch.cat((dst, src))
    rel = torch.cat((rel, rel + num_rels))

    edge_index = torch.stack((src, dst))
    edge_type = rel

    data = Data(edge_index = edge_index)
    data.entity = torch.from_numpy(uniq_entity)
    data.edge_type = edge_type
    data.edge_norm = edge_normalization(edge_type, edge_index, len(uniq_entity), num_rels)

    return data

def sentence_features(model, domain, split, all_seeds, concept_graphs,
                      relation_map, unique_nodes_mapping, args):
    """
        Graph features for each sentence (document) instance in a domain.
    """
    x, dico = get_domain_dataset(domain, exp_type=split)
    d = list(dico.values())

    sent_features = np.zeros((len(x), 100))
    
    for j in tqdm(range(len(x)), position=0, leave=False):
        c = [dico.id2token[item] for item in np.where(x[j] != 0)[0]]
        n = list(spacy_seed_concepts_list(c).intersection(set(all_seeds)))

        try:
            xg = np.concatenate([concept_graphs[item] for item in n])
            xg = xg[~np.all(xg == 0, axis=1)]
        
            absent1 = set(xg[:, 0]) - unique_nodes_mapping.keys()
            absent2 = set(xg[:, 2]) - unique_nodes_mapping.keys()
            absent = absent1.union(absent2)

            for item in absent:
                xg = xg[~np.any(xg == item, axis=1)]
        
            xg[:, 0] = np.vectorize(unique_nodes_mapping.get)(xg[:, 0])
            xg[:, 2] = np.vectorize(unique_nodes_mapping.get)(xg[:, 2])

            if args.kg_corruption:
                corrupt_indicies = random.sample(range(len(xg)), k=int(len(xg) * args.kg_corruption_rate))
                xg[corrupt_indicies, 2]  = random.choices(list(unique_nodes_mapping.values()),
                                                          k=int(len(xg) * args.kg_corruption_rate))

            xg = unique_rows(xg).astype('int64')
            # print(len(xg))
            # shuffle
            np.random.shuffle(xg)
            if len(xg) > args.kg_size:
                xg = xg[:args.kg_size, :]

            features = []
            # if len(xg) > eval_batch_size:
            #     for i in range(int(len(xg)/eval_batch_size)):
            #         inputs = xg[i*eval_batch_size:(i+1)*eval_batch_size]
            #         sg = generate_graph(inputs, len(relation_map)).to(torch.device('cuda'))
            #         seg_feat = model(sg.entity, sg.edge_index, sg.edge_type, sg.edge_norm)
            #         features.append(seg_feat.cpu().detach().numpy())
            # features = np.concatenate(np.array(features))
            # sent_features[j] = features.mean(axis=0)
            sg = generate_graph(xg, len(relation_map)).to(torch.device('cuda'))
            features = model(sg.entity, sg.edge_index, sg.edge_type, sg.edge_norm)
            sent_features[j] = features.cpu().detach().numpy().mean(axis=0)
            torch.cuda.empty_cache()
            
        except ValueError:
            pass
    
    return sent_features


def train(train_triplets, model, batch_size, split_size, negative_sample, reg_ratio, num_entities, num_relations):

    train_data = generate_sampled_graph_and_labels(train_triplets, batch_size, split_size, 
                                                   num_entities, num_relations, negative_sample)

    train_data.to(torch.device('cuda'))

    entity_embedding = model(train_data.entity, train_data.edge_index, train_data.edge_type, train_data.edge_norm)
    score, loss = model.score_loss(entity_embedding, train_data.samples, train_data.labels) 
    loss += reg_ratio * model.reg_loss(entity_embedding)
    return score, loss


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', type=int, default=5000, help='graph batch size')
    parser.add_argument('--split-size', type=float, default=0.5, help='what fraction of graph edges used in training')
    parser.add_argument('--ns', type=int, default=1, help='negative sampling ratio')
    parser.add_argument('--epochs', type=int, default=1500, help='number of epochs')
    parser.add_argument('--save', type=int, default=100, help='save after how many epochs')
    parser.add_argument('--lr', type=float, default=1e-2, help='learning rate')
    parser.add_argument('--dropout', type=float, default=0.25, help='learning rate')
    parser.add_argument('--reg', type=float, default=1e-2, help='regularization coefficient')
    parser.add_argument('--grad-norm', type=float, default=1.0, help='grad norm')
    parser.add_argument('--eval_only', action="store_true")
    parser.add_argument('--dataset_type', type=str, default='data2000')
    parser.add_argument('--eval_batch_size', type=int, default=1000)
    parser.add_argument('--checkpoint', type=str, default=None)
    parser.add_argument('--kg_name', type=str, default='conceptnet')
    parser.add_argument('--kg_size', type=int, default=30000)
    parser.add_argument('--kg_corruption', type=bool, default=False)
    parser.add_argument('--kg_corruption_rate', type=float, default=0.2)

    args = parser.parse_args()
    print(args)
    
    graph_batch_size = args.batch_size
    graph_split_size = args.split_size
    negative_sample = args.ns
    n_epochs = args.epochs
    save_every = args.save
    lr = args.lr
    dropout = args.dropout
    regularization = args.reg
    grad_norm = args.grad_norm

    pkl_path = os.path.join(data_dir, 'pkl_files', args.dataset_type, args.kg_name)
    all_seeds = pickle.load(open(os.path.join(pkl_path, 'all_seeds.pkl'), 'rb'))
    relation_map = pickle.load(open(os.path.join(pkl_path, 'relation_map.pkl'), 'rb'))
    unique_nodes_mapping = pickle.load(open(os.path.join(pkl_path, 'unique_nodes_mapping.pkl'), 'rb'))
    concept_graphs = pickle.load(open(os.path.join(pkl_path, 'concept_graphs.pkl'), 'rb'))
    train_triplets = np.load(open(os.path.join(pkl_path, 'triplets.np'), 'rb'), allow_pickle=True)

    n_bases = 4
    model = RGCN(len(unique_nodes_mapping), len(relation_map), num_bases=n_bases, dropout=dropout).cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)


    sess_path = os.path.join(sess_dir, args.dataset_type, args.kg_name)
    if not os.path.exists(sess_path):
        os.makedirs(sess_path)

    if args.checkpoint:
        model.load_state_dict(torch.load(args.checkpoint))

    writer = SummaryWriter(sess_path)

    if not args.eval_only:
        for epoch in tqdm(range(1, (n_epochs + 1)), desc='Epochs', position=0):

            permutation = torch.randperm(len(train_triplets))
            losses = []

            for i in range(0, len(train_triplets), graph_batch_size):

                model.train()
                optimizer.zero_grad()

                indices = permutation[i:i+graph_batch_size]

                score, loss = train(train_triplets[indices], model, batch_size=len(indices), split_size=graph_split_size,
                                    negative_sample=negative_sample, reg_ratio = regularization,
                                    num_entities=len(unique_nodes_mapping), num_relations=len(relation_map))

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_norm)
                optimizer.step()
                losses.append(loss.item())

            avg_loss = round(sum(losses)/len(losses), 4)

            writer.add_scalar("Loss/train", avg_loss, epoch)
            if epoch%save_every == 0:
                tqdm.write("Epoch {} Train Loss: {}".format(epoch, avg_loss))

                torch.save(model.state_dict(), os.path.join(sess_path,
                                                            'model_epoch' + str(epoch) +'.pt'))
    writer.close()
    model.eval()

    if args.dataset_type=="data2000":
        splits = ['test', 'small']
    elif args.dataset_type=="data1000":
        splits = ['test', 'd1000']
    elif args.dataset_type == "data500":
        splits = ['test', 'd500']

    for domain in ['books', 'dvd', 'electronics', 'kitchen']:
        print ('Extracting features for', domain)
        for split in splits:
            sf = sentence_features(model, domain, split, all_seeds, concept_graphs,
                                   relation_map, unique_nodes_mapping, args)
            graph_feat_path = os.path.join(data_dir, 'graph_features', args.dataset_type, args.kg_name)
            if not os.path.exists(graph_feat_path):
                os.makedirs(graph_feat_path)
            if args.kg_corruption:
                np.ndarray.dump(sf, open(os.path.join(graph_feat_path,
                                                      'sf_' + domain + '_' + split + '_bow5000_kg{}k_cor{}.np'.format(
                                                          int(args.kg_size / 1000), int(args.kg_corruption_rate*10))), 'wb'))
            else:
                np.ndarray.dump(sf, open(os.path.join(graph_feat_path, 'sf_'+ domain + '_' + split +
                                                      '_bow5000_kg{}k.np'.format(int(args.kg_size/1000))), 'wb'))

    print ('Done.')
    