from tqdm import tqdm
import numpy as np, os, gc, argparse, math
import pickle, argparse
import os, spacy
import scipy.sparse as sp
from gensim.corpora import Dictionary as gensim_dico
from sklearn.metrics import accuracy_score

import torch, torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.tensorboard import SummaryWriter


from dataset import TrainDataset, EvalDataset, pyg_collate
from rgcn import RGCN, mmRGCN
from models import LinearModel
from train import loss_ae

pkl_path = '/media/disk1/jennybae/data/kingdom/pkl_files'
sess_path = '/media/disk1/jennybae/kingdom/'



class E2EModel(nn.Module):
    def __init__(self, num_entities, num_relations, num_bases, input_dim1, input_dim2, dropout=0.25):
        super(E2EModel, self).__init__()

        self.gae_model = mmRGCN(num_entities, num_relations, num_bases, dropout)
        self.sa_model = LinearModel(input_dim1, input_dim2, dropout)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--no-cuda', action='store_true', default=False, help='does not use GPU')
    parser.add_argument('--batch-size', type=int, default=4, metavar='BS', help='batch size')
    parser.add_argument('--eval_batch_size', type=int, default=10, metavar='BS', help='batch size')
    parser.add_argument('--epochs', type=int, default=100, metavar='E', help='number of epochs')
    parser.add_argument('--lr', type=float, default=1e-4, metavar='LR', help='learning rate')
    parser.add_argument('--model_type', type=str)
    parser.add_argument('--dataset_type', type=str)
    parser.add_argument('--source_domain', type=str)
    parser.add_argument('--target_domain', type=str)
    parser.add_argument('--kg_name', type=str, help="conceptnet or wordnet18")
    parser.add_argument('--kg_seed_type', type=str, help="seed from dataXXXX")
    parser.add_argument('--kg_size', type=int, default=30000)
    parser.add_argument('--kg_corruption', type=bool, default=False)
    parser.add_argument('--kg_corruption_rate', type=float, default=0.2)
    parser.add_argument('--dropout', type=float)
    parser.add_argument('--alpha', type=float)
    args = parser.parse_args()
    print(args)

    n_epochs = args.epochs
    batch_size = args.batch_size
    len_dataloader = 2000/batch_size
    dr = args.dropout
    al = args.alpha
    lr = args.lr
    bow_size = 5000
    graph_size = 100
    n_bases = 4
    transform = True

    if args.kg_corruption:
        kg_spec = 'kg{}k_cor{}'.format(int(args.kg_size / 1000), int(args.kg_corruption_rate * 10))
    else:
        kg_spec = 'kg{}k'.format(int(args.kg_size / 1000))



    global use_cuda

    pkl_path = os.path.join(pkl_path, args.dataset_type, args.kg_name)
    all_seeds = pickle.load(open(os.path.join(pkl_path, 'all_seeds.pkl'), 'rb'))
    relation_map = pickle.load(open(os.path.join(pkl_path, 'relation_map.pkl'), 'rb'))
    unique_nodes_mapping = pickle.load(open(os.path.join(pkl_path, 'unique_nodes_mapping.pkl'), 'rb'))
    concept_graphs = pickle.load(open(os.path.join(pkl_path, 'concept_graphs.pkl'), 'rb'))
    train_triplets = np.load(open(os.path.join(pkl_path, 'triplets.np'), 'rb'), allow_pickle=True)


    if torch.cuda.is_available() and not args.no_cuda:
        use_cuda = True
    else:
        use_cuda = False

    loss_class = torch.nn.CrossEntropyLoss()
    loss_domain = torch.nn.CrossEntropyLoss()

    if use_cuda:
        loss_class = loss_class.cuda()
        loss_domain = loss_domain.cuda()

    train_dataset = TrainDataset(args.source_domain, args.target_domain, args.kg_name, transform=transform)
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.batch_size,
                                  collate_fn=pyg_collate)

    eval_dataset = EvalDataset(args.source_domain, args.target_domain, args.kg_name, transform=transform)
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size,
                                 collate_fn=pyg_collate)

    all_accs = []
    maxa = 0

    for iter_turn in range(5):
        base_path = os.path.join(sess_path, args.model_type, args.dataset_type,
                                 '{}-{}'.format(args.source_domain, args.target_domain), args.kg_name + "_" + kg_spec)
        sess_path = os.path.join(base_path, 'dr{}_al{}_lr{}/trial{}'.format(int(-math.log(dr, 2)),
                                                                            int(al), int(-math.log(lr, 10)), iter_turn))
        writer = SummaryWriter(sess_path)

        model = E2EModel(len(unique_nodes_mapping), len(relation_map), n_bases, bow_size, graph_size, dr)
        if use_cuda:
            model = model.cuda()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        for p in model.parameters():
            p.requires_grad = True


        eval_accs, eval_loss = [], []
        max_acc = 0

        for epoch in tqdm(range(1, (n_epochs+1))):

            for step, batch in enumerate(train_dataloader):
                model.train()
                optimizer.zero_grad()

                batch.to(torch.device('cuda'))
                #
                # x_s1, y_s, x_t1, gr, source_nodes, target_nodes = batch
                # x_s1, y_s, x_t1 = x_s1.cuda(), y_s.cuda(), x_t1.cuda()

                entity_embeddings = model.gae_model(batch.graphs)
                score, gae_loss = model.gae_model.score_loss(entity_embeddings, batch.graphs.samples, batch.graphs.labels)

                num_nodes = [len(g.entity) for g in batch.graphs.to_data_list()]
                s_xg = torch.zeros(batch.graphs.num_graphs, 100)
                t_xg = torch.zeros(batch.graphs.num_graphs, 100)
                for i in range(batch.graphs.num_graphs):
                    s_xg[i] = torch.mean(model.gae_model(batch.graphs)[:sum(num_nodes[:(i + 1)])][batch.s_nodes[0]], 0)
                    t_xg[i] = torch.mean(model.gae_model(batch.graphs)[:sum(num_nodes[:(i + 1)])][batch.t_nodes[0]], 0)

                y_s_domain = torch.zeros(batch.graphs.num_graphs).long().cuda()
                y_t_domain = torch.ones(batch.graphs.num_graphs).long().cuda()

                p = float(step*args.batch_size + epoch * len_dataloader) / n_epochs / len_dataloader

                if al == 1:
                    alpha = 4. / (1. + np.exp(-10 * p)) - 1
                elif al == 2:
                    alpha = 2. / (1. + np.exp(-10 * p)) - 1

                recon_s2, y_s_pred, y_s_domain_pred = model.sa_model(batch.s_bow, s_xg, alpha)
                recon_t2, _, y_t_domain_pred = model.sa_model(batch.t_bow, t_xg, alpha)

                loss_class_s = loss_class(y_s_pred, batch.s_label)
                loss_domain_s = loss_domain(y_s_domain_pred, y_s_domain)
                loss_domain_t = loss_domain(y_t_domain_pred, y_t_domain)
                loss_recon_s = loss_ae(recon_s2, s_xg)
                loss_recon_t = loss_ae(recon_t2, t_xg)
                sa_loss = loss_class_s + loss_recon_s + loss_recon_t + loss_domain_s + loss_domain_t

                loss = sa_loss + gae_loss

                loss.backward()
                optimizer.step()


            model.eval()
            Y_t_pred, preds, labels = [], [], []

            for step, batch in enumerate(eval_dataloader):
                batch.to(torch.device('cuda'))
                # x_t2, y_t2, graph_list, target_nodes = batch
                # x_t2, y_t2 = x_t2.cuda(), y_t2.cuda()

                sent_features = []

                entity_embeddings = model.gae_model(batch.graphs)
                sent_features.append(torch.mean(entity_embeddings, axis=0))
                sent_features = torch.stack(sent_features).to(torch.device('cuda'))

                recon_t2, y_t2_pred, y_t_domain_pred = model.sa_model(batch.t_bow, sent_features, 0)
                Y_t_pred.append(y_t2_pred)

                preds.append(torch.argmax(y_t2_pred, 1).data.cpu().numpy())
                labels.append(batch.t_label)

            Y_t_pred = torch.stack(Y_t_pred, dim=0)
            labels = torch.stack(labels, dim=0)
            e_loss = round(loss_class(Y_t_pred, labels).item(), 4)

            preds = np.concatenate(preds)
            labels = labels.data.cpu().numpy()
            e_acc = round(accuracy_score(labels, preds)*100, 2)

            max_acc = max(max_acc, e_acc)
            eval_accs.append(e_acc)
            eval_loss.append(e_loss)

            writer.add_scalar("Loss/eval/e2e_model/{}-{}".format(args.source_domain, args.target_domain), e_loss, epoch)
            writer.add_scalar("Acc/eval/e2e_model/{}-{}".format(args.source_domain, args.target_domain), e_acc, epoch)

        all_accs.append(max_acc)
        del model, optimizer

        writer.close()

        gc.collect()
