import numpy as np, os, gc, argparse, math
from tqdm import tqdm
import torch
import torch.optim as optim
from torch.nn import functional as F
from utils import get_dataset
from models import LinearModel
from sklearn.metrics import accuracy_score

from torch.utils.tensorboard import SummaryWriter



norm = {'booksdvd': 4.18, 'bookskitchen': 4.13, 'bookselectronics': 4.13,
        'electronicskitchen': 3.56, 'electronicsdvd': 4.18, 'electronicsbooks': 4.45,
        'kitchenbooks': 4.45, 'kitchenelectronics': 3.5, 'kitchendvd': 4.18,
        'dvdelectronics': 3.62, 'dvdkitchen': 3.62, 'dvdbooks': 4.45}


sess_dir = '/media/disk1/jennybae/kingdom/sa_model'
data_dir = '/media/disk1/jennybae/data/kingdom'

filename={"conceptnet": "conceptnet_english.txt",
          "wordnet18": "wordnet18.txt"}

data_type2exp_type ={"data2000": "small",
                     "data1000": "d1000",
                     "data500": "d500"}

def loss_ae(recon_x, x):
    dim = x.size(1)
    MSE = F.mse_loss(recon_x, x.view(-1, dim), reduction='mean')
    return MSE


def train_model(model, optimizer, loss_class, loss_domain, X_s1, X_s2, Y_s, X_t1, X_t2, al=1):
    
    loss1, loss2, loss3, loss4, loss5 = [], [], [], [], []
    preds1, preds2, preds3 = [], [], []
    labels1, labels2, labels3 = [], [], []
    
    permutation = torch.randperm(X_s.size(0))
    # print(permutation.shape)

    for i in range(0, X_s.size(0), batch_size):
        
        p = float(i + epoch * len_dataloader) / n_epochs / len_dataloader
        
        if al == 1:
            alpha = 4./ (1. + np.exp(-10 * p)) - 1
        elif al == 2:
            alpha = 2./ (1. + np.exp(-10 * p)) - 1
        
        model.train()
        optimizer.zero_grad()

        indices = permutation[i:i+batch_size]
        
        x_s1, x_s2, y_s, x_t1, x_t2 = X_s1[indices], X_s2[indices], Y_s[indices], X_t1[indices], X_t2[indices]
        
        y_s_domain = torch.zeros_like(permutation[i:i+batch_size]).long()
        y_t_domain = torch.ones_like(permutation[i:i+batch_size]).long()
        
        if use_cuda:
            y_s_domain = y_s_domain.cuda()
            y_t_domain = y_t_domain.cuda()
        
        recon_s2, y_s_pred, y_s_domain_pred = model(x_s1, x_s2, alpha)
        recon_t2, _, y_t_domain_pred = model(x_t1, x_t2, alpha)
        
        loss_class_s = loss_class(y_s_pred, y_s)
        loss_domain_s = loss_domain(y_s_domain_pred, y_s_domain)
        loss_domain_t = loss_domain(y_t_domain_pred, y_t_domain)
        loss_recon_s = loss_ae(recon_s2, x_s2)
        loss_recon_t = loss_ae(recon_t2, x_t2)
        
        loss = loss_class_s + loss_recon_s + loss_recon_t + loss_domain_s + loss_domain_t
        loss.backward()
        optimizer.step()
        
        preds1.append(torch.argmax(y_s_pred, 1).data.cpu().numpy())
        preds2.append(torch.argmax(y_s_domain_pred, 1).data.cpu().numpy())
        preds3.append(torch.argmax(y_t_domain_pred, 1).data.cpu().numpy())
        
        labels1.append(y_s.data.cpu().numpy())
        labels2.append(y_s_domain.data.cpu().numpy())
        labels3.append(y_t_domain.data.cpu().numpy())
        
        loss1.append(loss_class_s.item())
        loss2.append(loss_domain_s.item())
        loss3.append(loss_domain_t.item())
        loss4.append(loss_recon_s.item())
        loss5.append(loss_recon_t.item())
        
    preds1, preds2, preds3  = np.concatenate(preds1), np.concatenate(preds2), np.concatenate(preds3)
    labels1, labels2, labels3 = np.concatenate(labels1), np.concatenate(labels2), np.concatenate(labels3)
    
    avg_acc1 = round(accuracy_score(labels1, preds1)*100, 2)
    avg_acc2 = round(accuracy_score(labels2, preds2)*100, 2)
    avg_acc3 = round(accuracy_score(labels3, preds3)*100, 2)
    
    avg_loss1 = round(np.mean(np.array(loss1)), 4)
    avg_loss2 = round(np.mean(np.array(loss2)), 4)
    avg_loss3 = round(np.mean(np.array(loss3)), 4)
    avg_loss4 = round(np.mean(np.array(loss4)), 4)
    avg_loss5 = round(np.mean(np.array(loss5)), 4)
        
    # print ('Source sentiment loss: {a}, domain loss: {b}, recons loss: {c}, sentiment acc: {d}, domain acc: {e}'.format(
    #        a = avg_loss1, b = avg_loss2, c = avg_loss4, d = avg_acc1, e = avg_acc2))
        
    # print ('Target domain loss: {a}, recons loss: {b}, domain acc: {c}'.format(
    #        a = avg_loss3, b = avg_loss5, c = avg_acc3))
        
        
def eval_model(model, loss_class, loss_domain, X_t1, X_t2, Y_t, max_acc):
        
    model.eval()
    Y_t_domain = torch.ones(len(Y_t)).long()
    
    if use_cuda:
        Y_t_domain = Y_t_domain.cuda()
    
    recon_t2, Y_t_pred, Y_t_domain_pred = model(X_t1, X_t2, 0)
        
    loss1 = round(loss_class(Y_t_pred, Y_t).item(), 4)
    loss2 = round(loss_domain(Y_t_domain_pred, Y_t_domain).item(), 4)
    loss3 = round(loss_ae(recon_t2, X_t2).item(), 4)
    
    preds1 = torch.argmax(Y_t_pred, 1).data.cpu().numpy()
    preds2 = torch.argmax(Y_t_domain_pred, 1).data.cpu().numpy()
        
    labels1 = Y_t.data.cpu().numpy()
    labels2 = Y_t_domain.data.cpu().numpy()
    
    avg_acc1 = round(accuracy_score(labels1, preds1)*100, 2)
    avg_acc2 = round(accuracy_score(labels2, preds2)*100, 2)
    
    # print ('Target sentiment loss: {a}, domain loss: {b}, recons loss: {c}, sentiment acc: {d}, domain acc: {e}'.format(
    #      a = loss1, b = loss2, c = loss3, d = avg_acc1, e = avg_acc2))

    if avg_acc1>max_acc:
        np.savetxt(os.path.join(sess_path, 'output.txt'), preds1, fmt = '%2d', delimiter='\n')
        np.savetxt(os.path.join(sess_path, 'label.txt'), labels1, fmt = '%2d', delimiter='\n')

    return avg_acc1, loss1


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--no-cuda', action='store_true', default=False, help='does not use GPU')
    parser.add_argument('--batch-size', type=int, default=40, metavar='BS', help='batch size')
    parser.add_argument('--epochs', type=int, default=100, metavar='E', help='number of epochs')
    parser.add_argument('--lr', type=float, default=1e-4, metavar='LR', help='learning rate')
    parser.add_argument('--model_type', type=str)
    parser.add_argument('--dataset_type', type=str)
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
    # dropouts = [0.25, 0.5]
    # alphas = [1, 2]

    bow_size = 5000
    graph_size = 100
    transform = True

    global use_cuda

    if torch.cuda.is_available() and not args.no_cuda:
        use_cuda = True
    else:
        use_cuda = False

    loss_class = torch.nn.CrossEntropyLoss()
    loss_domain = torch.nn.CrossEntropyLoss()

    if use_cuda:
        loss_class = loss_class.cuda()
        loss_domain = loss_domain.cuda()


    domains = ['books', 'dvd', 'electronics', 'kitchen']

    for d1 in domains:
        for d2 in domains:
        
            if d1 == d2:
                continue

            # BOW features and sentiment labels
            X_s, Y_s, X_t1, Y_t1, X_t2, Y_t2, _ = get_dataset(d1, d2, max_words=bow_size, exp_type=data_type2exp_type[args.dataset_type])

            Y_s = torch.LongTensor(Y_s)
            Y_t1 = torch.LongTensor(Y_t1)
            Y_t2 = torch.LongTensor(Y_t2)

            # graph_feat_path = os.path.join(data_dir, 'graph_features', args.dataset_type, args.kg_name)
            # (sf, open(os.path.join(graph_feat_path, 'sf_' + domain + '_' + split + '_bow5000_kg{}k_cor{}.np'.format(
            #                            int(args.kg_size / 1000), int(args.kg_corruption_rate * 10))), 'wb'))

            if args.kg_corruption:
                kg_spec = 'kg{}k_cor{}'.format(int(args.kg_size/1000), int(args.kg_corruption_rate*10))
            else:
                kg_spec = 'kg{}k'.format(int(args.kg_size/1000))

            # Graph features
            X_s_ = np.load(open(os.path.join(data_dir, 'graph_features', args.kg_seed_type, args.kg_name,
                                     'sf_' + d1 +'_{}_bow5000_'.format(data_type2exp_type[args.kg_seed_type]) + kg_spec + ".np"), 'rb'), allow_pickle=True)
            X_t1_ = np.load(open(os.path.join(data_dir, 'graph_features',  args.kg_seed_type, args.kg_name,
                                     'sf_' + d2 + '_{}_bow5000_'.format(data_type2exp_type[args.kg_seed_type]) + kg_spec + ".np"), 'rb'), allow_pickle = True)

            X_t2_ = np.load(open(os.path.join(data_dir, 'graph_features',  args.kg_seed_type, args.kg_name,
                                    'sf_' + d2 + '_test_bow5000_' + kg_spec + ".np"), 'rb'), allow_pickle = True)

            if transform:

                c = norm[d1+d2]
                X_s  = torch.tensor(np.log(1 + np.array(X_s.todense()).astype('float32'))/c)
                X_t1 = torch.tensor(np.log(1 + np.array(X_t1.todense()).astype('float32'))/c)
                X_t2 = torch.tensor(np.log(1 + np.array(X_t2.todense()).astype('float32'))/c)
            
                X_s_ = torch.sigmoid(torch.tensor(X_s_)).type(torch.FloatTensor)
                X_t1_ = torch.sigmoid(torch.tensor(X_t1_)).type(torch.FloatTensor)
                X_t2_ = torch.sigmoid(torch.tensor(X_t2_)).type(torch.FloatTensor)
        
            else:
            
                X_s  = torch.tensor(np.array(X_s.todense()).astype('float32'))
                X_t1 = torch.tensor(np.array(X_t1.todense()).astype('float32'))
                X_t2 = torch.tensor(np.array(X_t2.todense()).astype('float32'))
            
                X_s_ = torch.tensor(X_s_).type(torch.FloatTensor)
                X_t1_ = torch.tensor(X_t1_).type(torch.FloatTensor)
                X_t2_ = torch.tensor(X_t2_).type(torch.FloatTensor)
            
            if use_cuda:
                X_s, X_t1, X_t2,  = X_s.cuda(), X_t1.cuda(), X_t2.cuda()
                Y_s, Y_t1, Y_t2,  = Y_s.cuda(), Y_t1.cuda(), Y_t2.cuda()
                X_s_, X_t1_, X_t2_ = X_s_.cuda(), X_t1_.cuda(), X_t2_.cuda()
        
            all_accs = []
            maxa = 0


            for iter_turn in range(5):
                base_path = os.path.join(sess_dir, args.model_type, args.dataset_type,
                                         '{}-{}'.format(d1, d2), args.kg_name+"_"+kg_spec)
                sess_path = os.path.join(base_path, 'dr{}_al{}_lr{}/trial{}'.format(int(-math.log(dr,2)),
                                                                            int(al), int(-math.log(lr, 10)), iter_turn))
                # print(base_path)
                # print(sess_path)
                # 1/0
                if not os.path.exists(sess_path):
                    os.makedirs(sess_path)

                writer = SummaryWriter(sess_path)

                model = LinearModel(bow_size, graph_size, dr)
                if use_cuda:
                    model = model.cuda()
                optimizer = optim.Adam(model.parameters(), lr=lr)
                for p in model.parameters():
                    p.requires_grad = True

                accs, loss = [], []
                max_acc = 0

                for epoch in tqdm(range(1, (n_epochs+1)), desc='Epochs', position=0):
                    train_model(model, optimizer, loss_class, loss_domain, X_s, X_s_, Y_s, X_t1, X_t1_, al)
                    acc, l = eval_model(model, loss_class, loss_domain, X_t2, X_t2_, Y_t2, max_acc)
                    max_acc = max(max_acc, acc)
                    accs.append(acc)
                    loss.append(l)
                    writer.add_scalar("Loss/eval/sa_model/{}-{}".format(d1, d2), l, epoch)
                    writer.add_scalar("Acc/eval/sa_model/{}-{}".format(d1, d2), acc, epoch)

                all_accs.append(max_acc)

                del model, optimizer
                writer.close()

                gc.collect()

                # if max_acc > maxa:
                params = {'lr': lr, 'dr': dr, 'alpha': al}
                maxa = max_acc
                print ('Results: Acc: {a}, Loss: {b}, LR: {c}, Dropout: {d}, alpha: {e}'
                        .format(a = max_acc, b = loss[accs.index(max_acc)], c = lr, d = dr, e = al))
                f = open(os.path.join(base_path, 'eval_summary.txt'), 'a')
                f.write('Acc:{a}\tLoss:{b}\tLR:{c}\tDropout:{d}\talpha:{e}\n'
                        .format(a = max_acc, b = loss[accs.index(max_acc)], c = lr, d = dr, e = al))
                f.close()

            print (d1, d2, str(max(all_accs)))
            # print ('Best results at:', params)
            print ('-'*70)
            del maxa
            