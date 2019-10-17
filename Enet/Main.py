import matplotlib
matplotlib.use('Agg')

import torch
import sys
import os.path
sys.path.append('%s/../../pytorch_DGCNN' % os.path.dirname(os.path.realpath(__file__)))
from main import *
from util_functions import *
import networkx as nx
from scipy.sparse.linalg import svds, eigs
from scipy.sparse import csr_matrix
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn import metrics
import cPickle as pickle
import scipy.stats as ss
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import *
# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "2"

parser = argparse.ArgumentParser(description='E-Net')
# general settings
parser.add_argument('--save-name', default='0', help='save model name')
parser.add_argument('--task', default='missing', help='task name')
parser.add_argument('--data-name', default='cora', help='network name')
parser.add_argument('--data-name-noise', default='0', help='noisy rate name')
parser.add_argument('--train-name', default=None, help='train name')
parser.add_argument('--test-name', default=None, help='test name')
parser.add_argument('--max-train-num', type=int, default=100000,
                    help='set maximum number of train links (to fit into memory)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--test-ratio', type=float, default=0.1,
                    help='ratio of test links')
parser.add_argument('--missing-ratio', type=float, default=0.005,   # to tune
                    help='ratio of missing links')
parser.add_argument('--split-ratio',type=str,default='0.8,0.1,0.1',
                    help='ratio of train, val and test links')
parser.add_argument('--neg-pos-ratio', type=float, default=5,
                    help='ratio of negative/positive links')
# model settings
parser.add_argument('--hop', default=2, metavar='S',
                    help='enclosing subgraph hop number, \
                    options: 1, 2,..., "auto"')
parser.add_argument('--max-nodes-per-hop', default=20,
                    help='if > 0, upper bound the # nodes per hop by subsampling')
parser.add_argument('--use-embedding', action='store_true', default=False,   # to tune
                    help='whether to use node2vec node embeddings')
parser.add_argument('--embedding-size', default=128,   # to tune
                    help='embedding size of node2vec')
parser.add_argument('--use-attribute', action='store_true', default=True,
                    help='whether to use node attributes')
parser.add_argument('--add-noise', default=True, help='whether to use add noise')
parser.add_argument('--noisy-perc', type=float, default=0.2, help='noise percentage')
parser.add_argument('--use-svd', action='store_true', default=False,
                    help='whether to use svd to reduce feature dimension')
parser.add_argument('--lazy-subgraph', action='store_true', default=True,   # to tune
                    help='whether to use lazy subgraph extraction')
parser.add_argument('--multi-subgraph', default=3,   # to tune
                    help='number of subgraphs to extract for each query nodes')
parser.add_argument('--num-walks', default=5,   # to tune
                    help='number of walks for each node')
parser.add_argument('--num-node-to-walks', default=5,   # to tune
                    help='number of walks for each node')
parser.add_argument('--use-pos', default=False, help='whether to use subgraph position')
# earlystopping setting
parser.add_argument('--early-stop', default=True, help='whether to use early stopping')   # to tune
parser.add_argument('--early-stop-patience', type=int, default=7, help='patience for early stop')
parser.add_argument('--early-stop-index', type=int, default=0, help='early stop index')
parser.add_argument('--early-stop-delta', type=float, default=1e-3, help='early stop delta')
# GCN setting
parser.add_argument('--learning-rate', type=float, default=1e-4, help='GCN learning rate')  # to tune
parser.add_argument('--smooth-coef', type=float, default=1e-4, help='smooth regularization coefficient')
parser.add_argument('--noise-hidden-dim', type=int, default=300, help='noise detector hidden dimension')
parser.add_argument('--reg-smooth', action='store_true', default=True,   # to tune
                    help='whether to use smooth regularization')
parser.add_argument('--disable-train', action='store_true', help='disable train mode')
parser.add_argument('--eval', action='store_true', help='disable eval mode')
parser.add_argument('--load-save-graph', action='store_true', help='whether to save or load graph')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)
print(args)

random.seed(cmd_args.seed)
np.random.seed(cmd_args.seed)
torch.manual_seed(cmd_args.seed)
if args.hop != 'auto':
    args.hop = int(args.hop)
if args.max_nodes_per_hop is not None:
    args.max_nodes_per_hop = int(args.max_nodes_per_hop)
if args.num_walks is not None:
    args.num_walks = int(args.num_walks)
if args.num_node_to_walks is not None:
    args.num_node_to_walks = int(args.num_node_to_walks)
args.embedding_size = int(args.embedding_size)
args.split_ratio = [float(_) for _ in args.split_ratio.split(',')]

class EarlyStop(object):
    def __init__(self, patience=args.early_stop_patience, index=args.early_stop_index, delta=args.early_stop_delta):
        self.patience = patience
        self.delta = delta
        self.index = index
        self.best_loss = 1e15
        self.test_loss = 1e15
        self.wait = 0
        self.finish = 1

    def check(self, val_loss, test_loss, noise_test_loss, epoch):
        if val_loss[self.index] - self.best_loss < - self.delta:
            self.best_loss = val_loss[self.index]
            self.test_loss = test_loss
            self.noise_test_loss = noise_test_loss
            self.epoch = epoch
            # self.noise_test_loss = noise_test_loss
            self.wait = 1
        else:
            self.wait += 1
        return self.wait > self.patience

'''Prepare data'''
args.file_dir = os.path.dirname(os.path.realpath('__file__'))
args.res_dir = os.path.join(args.file_dir, 'results/{}'.format(args.data_name))

if args.train_name is None:
    if args.data_name == 'ppd_rule_new':
        args.data_dir = os.path.join(args.file_dir, 'data/{}_clean_wlabel.mat'.format(args.data_name))
    if args.data_name == 'cora' or args.data_name == 'citeseer' or args.data_name == 'pubmed':
        args.data_dir = os.path.join(args.file_dir, 'data/{}.mat'.format(args.data_name))
    print("input: {}".format(args.data_dir))
    data = sio.loadmat(args.data_dir)
    # label = data['label']
    net_c = data['net']   # input: clean network (not consider missing links)
    label = data['label']
    if label.shape[0] == 1:
        label = label[0]
    else:
        label = np.argmax(label, axis=1)
    print('shape of net: {}, shape of label: {}'.format(net_c.shape, label.shape))

    if args.add_noise:
        if args.data_name == 'cora' or args.data_name == 'citeseer' or args.data_name == 'pubmed':
            data = sio.loadmat(os.path.join(args.file_dir, 'data/{}_n0.1.mat'.format(args.data_name)))
        else:
            data = sio.loadmat(os.path.join(args.file_dir, 'data/{}.mat'.format(args.data_name)))
        net = data['net']   # input: flawed network
        node_num = net.shape[0]

        # edges; clean edges; noisy edges
        edges = nx.from_scipy_sparse_matrix(net).edges()
        edges_clean = nx.from_scipy_sparse_matrix(net_c).edges()
        # edges_noisy = [e for e in edges if e not in edges_clean and e[::-1] not in edges_clean]
        print('# of nodes: {}'.format(node_num))
        print('# of clean/all edges: {}/{}'.format(len(edges_clean), len(edges)))
        num_noisy_edge = len(edges) - len(edges_clean)
    else:
        M_noise = None
    if data.has_key('group'):
        # load node attributes (here a.k.a. node classes)
        try:
            attributes = data['group'].toarray().astype('float32')
        except AttributeError:
            attributes = data['group'].astype('float32')
        if args.use_svd:
            attributes = csr_matrix(attributes, dtype=float)
            u, s, vt = svds(attributes, k=100)
            attributes = u * s
    else:
        attributes = None
    print('attribute dimension:', attributes.shape)
    # check whether net is symmetric (for small nets only)
    if False:
        net_ = net.toarray()
        assert(np.allclose(net_, net_.T, atol=1e-8))
    #Sample train and test links
    train_val_test = sample_train_val_test(edges, edges_clean, net, args.split_ratio, args.missing_ratio,
                                           args.neg_pos_ratio, max_train_num=args.max_train_num, task=args.task)
else:
    args.train_dir = os.path.join(args.file_dir, 'data/{}'.format(args.train_name))
    args.test_dir = os.path.join(args.file_dir, 'data/{}'.format(args.test_name))
    train_idx = np.loadtxt(args.train_dir, dtype=int)
    test_idx = np.loadtxt(args.test_dir, dtype=int)
    max_idx = max(np.max(train_idx), np.max(test_idx))
    net = ssp.csc_matrix((np.ones(len(train_idx)), (train_idx[:, 0], train_idx[:, 1])), shape=(max_idx+1, max_idx+1))
    net[train_idx[:, 1], train_idx[:, 0]] = 1  # add symmetric edges
    net[np.arange(max_idx+1), np.arange(max_idx+1)] = 0  # remove self-loops
    #Sample negative train and test links
    train_pos = (train_idx[:, 0], train_idx[:, 1])
    test_pos = (test_idx[:, 0], test_idx[:, 1])
    train_val_test = sample_neg(net, train_pos=train_pos, test_pos=test_pos, max_train_num=args.max_train_num)
if args.task == 'missing':
    print('MISSING LINK PREDICTION-- # train_pos: %d, # train_neg: %d, # val_pos: %d, # val_neg: %d, # test_pos: %d, # test_neg: %d' % (
        len(train_val_test['train'][0][0]), len(train_val_test['train'][1][0]), len(train_val_test['val'][0][0]),
        len(train_val_test['val'][1][0]), len(train_val_test['test'][0][0]), len(train_val_test['test'][1][0])))
else:
    print('MISSING LINK PREDICTION-- # train_pos: %d, # train_neg: %d, # test_pos: %d, # test_neg: %d' % (
        len(train_val_test['train'][0][0]), len(train_val_test['train'][1][0]),
        len(train_val_test['test'][0][0]), len(train_val_test['test'][1][0])))

'''Train and apply classifier'''
A = net.copy()  # the observed network
# mask missing links
for key, value in train_val_test.items():
    A[value[0][0], value[0][1]] = 0
    A[value[0][1], value[0][0]] = 0

node_information = None
if args.use_embedding:
    embedding_fn = 'embeddings/{}_{}.npy'.format(args.data_name, args.embedding_size)
    if not os.path.exists('embeddings'):
        os.mkdir('embeddings')
    if os.path.exists(embedding_fn):
        print('load embeddings')
        embeddings = np.load(embedding_fn)
        print('finish loading embeddings')
    else:
        print('generate embedding')
        embeddings = generate_node2vec_embeddings(A, args.embedding_size, True, train_val_test['train'][1])
        embeddings = np.array(embeddings)
        np.save(embedding_fn, embeddings)
    node_information = embeddings
if args.use_attribute and attributes is not None:
    if node_information is not None:
        node_information = np.concatenate([node_information, attributes], axis=1)
    else:
        node_information = attributes

''' Construct data for noisy link detection '''
target_noise = [1 if e not in edges_clean and e[::-1] not in edges_clean else 0 for e in edges]
edges = np.array(edges)
data_to_pred_noise = np.concatenate([node_information[edges[:,0]], node_information[edges[:,1]]], 1)
data_to_pred_noise = torch.from_numpy(data_to_pred_noise).type('torch.FloatTensor').cuda()

graph_fn = 'graph/{}_{}_{}_{}.pkl'.format(args.data_name, args.hop, args.max_nodes_per_hop, args.lazy_subgraph)
if not os.path.exists('graph'):
    os.mkdir('graph')
if os.path.exists(graph_fn) and args.load_save_graph:
    print('load subgraphs')
    graphs, max_n_label = pickle.load(open(graph_fn, 'r'))
    print('finish loading subgraphs')
else:
    graphs, max_n_label = links2subgraphs(A, train_val_test, args.hop,
                                          args.max_nodes_per_hop, node_information,
                                          None, args.lazy_subgraph, args.multi_subgraph,
                                          args.num_node_to_walks, args.num_walks)
    if args.load_save_graph:
        pickle.dump([graphs, max_n_label], open(graph_fn, 'w'))
if args.task == 'missing':
    train_graphs, val_graphs, test_graphs = graphs['train'], graphs['val'], graphs['test']
    print('# train graph: %d, #val graph: %d, # test graph: %d' % (len(train_graphs), len(val_graphs), len(test_graphs)))
else:
    train_graphs, test_graphs = graphs['train'], graphs['test']
    print('# train graph: %d, # test graph: %d' % (len(train_graphs),len(test_graphs)))

# print(test_graphs)
# DGCNN configurations
cmd_args.gm = 'PPDGCNN'
cmd_args.sortpooling_k = 0.6
cmd_args.latent_dim = [32, 32, 32, 1]
cmd_args.hidden = 128   # to tune
cmd_args.out_dim = 0
cmd_args.dropout = True
cmd_args.num_class = 2
cmd_args.mode = 'gpu'
cmd_args.num_epochs = 100
cmd_args.learning_rate = args.learning_rate   # to tune
cmd_args.batch_size = 50
cmd_args.printAUC = True
cmd_args.feat_dim = max_n_label + 1
cmd_args.attr_dim = 0
cmd_args.decay_learning_rate = False
cmd_args.sumsort = False
cmd_args.pretrain = False

cmd_args.noise_matrix = True
cmd_args.PP = False
cmd_args.GT = False
cmd_args.reg_nd = False
cmd_args.reg_smooth = args.reg_smooth
cmd_args.reg_l1 = False
cmd_args.loss_missing = True

cmd_args.smooth_coef = args.smooth_coef  # to tune
cmd_args.l1_coef = 1
cmd_args.noise_coef = 1   # to tune
cmd_args.pr_threshold = 0
cmd_args.attention_mode = ''
cmd_args.noise_hidden_dim = args.noise_hidden_dim   # to tune
cmd_args.total_num_nodes = net.shape[0]
cmd_args.use_pos = args.use_pos
cmd_args.softmax = False
cmd_args.nodefeat_lp = True
print(cmd_args)

if node_information is not None:
    cmd_args.attr_dim = node_information.shape[1]
if cmd_args.sortpooling_k <= 1:
    if args.task == 'missing':
        num_nodes_list = sorted([g.num_nodes for g in train_graphs + test_graphs + val_graphs])
    else:
        num_nodes_list = sorted([g.num_nodes for g in train_graphs + test_graphs])
    cmd_args.sortpooling_k = num_nodes_list[int(math.ceil(cmd_args.sortpooling_k * len(num_nodes_list))) - 1]
    cmd_args.sortpooling_k = max(10, cmd_args.sortpooling_k)
    print('k used in SortPooling is: ' + str(cmd_args.sortpooling_k))
# suffix = 'PP-{}.GT-{}.nm-{}.pre-{}.ndim-{}.thr-{}.att-{}.svd-{}.reg_smooth-{}.reg_l1-{}.{}.GTcleannet'.format(
#     cmd_args.PP, cmd_args.GT, cmd_args.noise_matrix, cmd_args.pretrain, cmd_args.noise_hidden_dim,
#     cmd_args.pr_threshold, cmd_args.attention_mode, args.use_svd, cmd_args.reg_smooth,
#     cmd_args.reg_l1, args.data_name)

suffix = '{}.nw-{}.l-{}.s-{}.nd-{}.lr-{}'.format(args.data_name, args.num_walks, args.num_node_to_walks,
                                                 args.smooth_coef, args.noise_hidden_dim, args.learning_rate)

def eval_noise(classifier, data_to_pred_noise, target_noise):
    params = classifier.state_dict()
    w1 = params['gnn.noise_params1.weight']
    b1 = params['gnn.noise_params1.bias']
    w2 = params['gnn.noise_params2.weight']
    b2 = params['gnn.noise_params2.bias']
    print("w1:", w1)
    input_dim = w1.shape[-1]
    noise_hidden_dim = w1.shape[0]
    noise_activation = eval('nn.{}()'.format('Sigmoid'))
    noise_params1 = torch.nn.Linear(input_dim, noise_hidden_dim)
    noise_params2 = torch.nn.Linear(noise_hidden_dim, 1)

    # weights_init(self)
    param = getattr(noise_params1, 'weight')
    param.data = w1
    param = getattr(noise_params1, 'bias')
    param.data = b1
    param = getattr(noise_params2, 'weight')
    param.data = w2
    param = getattr(noise_params2, 'bias')
    param.data = b2

    noise_predict_layer = noise_activation(noise_params2(
        noise_params1(data_to_pred_noise))).squeeze()

    e_prob = noise_predict_layer.cpu().detach().numpy()
    # e_class = np.where(e_prob < 0.5, 1, 0)
    ranks = ss.rankdata(e_prob)
    # th = int(0.1*len(e_prob))
    th = num_noisy_edge
    e_class = [1 if r <= th else 0 for r in ranks]
    acc = accuracy_score(target_noise, e_class)
    p = precision_score(target_noise, e_class)
    r = recall_score(target_noise, e_class)
    f1 = f1_score(target_noise, e_class)
    fpr, tpr, _ = metrics.roc_curve(target_noise, e_prob, pos_label=1)
    auc = metrics.auc(fpr, tpr)
    record = [acc, auc, p, r, f1]

    return record

if not args.disable_train:
    classifier = Classifier()
    if cmd_args.mode == 'gpu':
        classifier = classifier.cuda()

    optimizer = optim.Adam(classifier.parameters(), lr=cmd_args.learning_rate)

    train_idxes = list(range(len(train_graphs)))
    early_stop = EarlyStop()

    for epoch in range(cmd_args.num_epochs):
        t0 = time.time()

        random.shuffle(train_idxes)
        classifier.train()
        avg_loss, _, _ = loop_dataset(train_graphs, classifier, train_idxes, optimizer=optimizer)
        if not cmd_args.printAUC:
            avg_loss[2] = 0.0
        print('\033[92maverage training of epoch %d: loss %.5f acc %.5f auc %.5f pre %.5f rec %.5f f1 %.5f '
              'reg_smooth %.5f reg_l1 %.5f time %.5f\033[0m' %
              (epoch, avg_loss[0], avg_loss[1], avg_loss[-1], avg_loss[2], avg_loss[3], avg_loss[4],
               avg_loss[5], avg_loss[6], time.time() - t0))

        classifier.eval()
        if args.task == 'missing':
            val_loss, _, _ = loop_dataset(val_graphs, classifier, list(range(len(val_graphs))), test=True)
            # print('\033[93maverage val of epoch %d: loss %.5f acc %.5f auc %.5f pre %.5f rec %.5f f1 %.5f '
            #       'reg_smooth %.5f reg_l1 %.5f time %.5f\033[0m' %
            #       (epoch, val_loss[0], val_loss[1], val_loss[-1], val_loss[2], val_loss[3], val_loss[4],
            #        val_loss[5], val_loss[6], time.time() - t0))

        test_loss, _, _ = loop_dataset(test_graphs, classifier, list(range(len(test_graphs))), test=True)
        if not cmd_args.printAUC:
            test_loss[2] = 0.0
        print('\033[93maverage test of epoch %d: loss %.5f acc %.5f auc %.5f pre %.5f rec %.5f f1 %.5f '
              'reg_smooth %.5f reg_l1 %.5f time %.5f\033[0m' %
              (epoch, test_loss[0], test_loss[1], test_loss[-1], test_loss[2], test_loss[3], test_loss[4],
               test_loss[5], test_loss[6], time.time() - t0))

        # get noisy link detection performance
        noise_test_loss = eval_noise(classifier, data_to_pred_noise, target_noise)
        print('\033[93mnoise test of epoch %d: acc %.5f auc %.5f pre %.5f rec %.5f f1 %.5f \033[0m' %
              (epoch, noise_test_loss[0], noise_test_loss[1], noise_test_loss[2], noise_test_loss[3],
               noise_test_loss[4]))

        record_loss = val_loss if args.task == 'missing' else noise_val_loss
        if early_stop.check(record_loss, test_loss, noise_test_loss, epoch):
            print("------ early stopping ------")
            test_loss = early_stop.test_loss
            noise_test_loss = early_stop.noise_test_loss
            print('\033[93mearlystop missing test of epoch %d: loss %.5f acc %.5f auc %.5f pre %.5f rec %.5f f1 %.5f '
                  'reg_smooth %.5f reg_l1 %.5f time %.5f\033[0m' %
                  (epoch, test_loss[0], test_loss[1], test_loss[-1], test_loss[2], test_loss[3], test_loss[4],
                   test_loss[5], test_loss[6], time.time() - t0))
            print('\033[93mearlystop noisy test of epoch %d: acc %.5f auc %.5f pre %.5f rec %.5f f1 %.5f \033[0m' %
                  (epoch, noise_test_loss[0], noise_test_loss[1], noise_test_loss[2], noise_test_loss[3],
                   noise_test_loss[4]))
            # torch.save(classifier, 'model/{}.pkl'.format(suffix))
            break

    torch.save(classifier, 'model/{}.pkl'.format(suffix+args.save_name))
if args.eval:
    classifier = torch.load('model/{}.pkl'.format(suffix+args.save_name))

    ''' Construct data for noisy link detection '''
    target_noise = [1 if e not in edges_clean and e[::-1] not in edges_clean else 0 for e in edges]
    edges = np.array(edges)
    print(edges)
    # data_to_pred_noise = preprocessing.normalize(np.concatenate([node_information[edges[:,0]], node_information[edges[:,1]]], 1))
    data_to_pred_noise = np.concatenate([node_information[edges[:,0]], node_information[edges[:,1]]], 1)
    data_to_pred_noise = torch.from_numpy(data_to_pred_noise).type('torch.FloatTensor').cuda()

    # classifier = torch.load('model/'+args.model_name)

    params = classifier.state_dict()
    w1 = params['gnn.noise_params1.weight']
    b1 = params['gnn.noise_params1.bias']
    w2 = params['gnn.noise_params2.weight']
    b2 = params['gnn.noise_params2.bias']

    input_dim = w1.shape[-1]
    noise_hidden_dim = w1.shape[0]
    noise_activation = eval('nn.{}()'.format('Sigmoid'))
    noise_params1 = nn.Linear(input_dim, noise_hidden_dim)
    noise_params2 = nn.Linear(noise_hidden_dim, 1)

    # weights_init(self)
    param = getattr(noise_params1, 'weight')
    param.data = w1
    param = getattr(noise_params1, 'bias')
    param.data = b1
    param = getattr(noise_params2, 'weight')
    param.data = w2
    param = getattr(noise_params2, 'bias')
    param.data = b2

    noise_predict_layer = noise_activation(noise_params2(
        noise_params1(data_to_pred_noise))).squeeze()


    e_prob = noise_predict_layer.cpu().detach().numpy()

    if not os.path.exists('network'):
        os.mkdir('network')

    t0 = time.time()
    result = dict()
    '''Train and apply attr classifier'''
    B = net.copy()  # the observed network


    test =  train_val_test['test']
    B[test[0][0], test[0][1]] = 0
    B[test[0][1], test[0][0]] = 0
    save_graph(B, label, 'network/{}_{}'.format(args.data_name, 'noisy'))

    embeddings = generate_node2vec_embeddings(B, 128, False)

    print('start training on noisy graph')
    mlp = MLPClassifier([64,16,8,1], early_stopping=True)
    X_train, X_test, y_train, y_test = train_test_split(embeddings, label, test_size=0.2, random_state=42)
    model = mlp.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    result['noisy'] = [f1_score(y_test, y_pred, average='macro'), f1_score(y_test, y_pred, average='micro'), f1_score(y_test, y_pred, average='weighted')]

    classifier.eval()
    # result = classifier(test_graphs)
    # print(test_graphs)
    # print(type(test_graphs))
    predicts, targets = loop_dataset(test_graphs, classifier, list(range(len(test_graphs))), test=True, score=True)


    ###predicted noise link
    noise_thres = 0.1
    predicted_noise_link = [[],[]]
    for i in range(len(e_prob)):
        if e_prob[i] <= noise_thres:
            predicted_noise_link[0].append(edges[i][0])
            predicted_noise_link[1].append(edges[i][1])

    predicted_missing_link = [[],[]]
    test_links = train_val_test['test']
    test_links = [test_links[0][0] + test_links[1][0], test_links[0][1] + test_links[1][1]]
    missing_thres = 0.5
    print(len(test_links[0]),len(predicts))
    for i in range(len(predicts)):
        if predicts[i] == 1:
            predicted_missing_link[0].append(test_links[0][i])
            predicted_missing_link[1].append(test_links[1][i])
    print(predicts)

    B[predicted_noise_link[0], predicted_noise_link[1]] = 0
    B[predicted_noise_link[1], predicted_noise_link[0]] = 0
    B[predicted_missing_link[0], predicted_missing_link[1]] = 1
    B[predicted_missing_link[1], predicted_missing_link[0]] = 1
    save_graph(B, label, 'network/{}_{}'.format(args.data_name, 'clean'))

    embeddings = generate_node2vec_embeddings(B, 128, False)
    print('start training on cleaned graph')
    mlp = MLPClassifier([64,16,8,1], early_stopping=True)
    X_train, X_test, y_train, y_test = train_test_split(embeddings, label, test_size=0.2, random_state=42)
    model = mlp.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    result['clean'] = [f1_score(y_test, y_pred, average='macro'), f1_score(y_test, y_pred, average='micro'), f1_score(y_test, y_pred, average='weighted')]


    embeddings = generate_node2vec_embeddings(net_c, 64, False)
    print('start training on clean graph')
    mlp = MLPClassifier([64,16,8,1], early_stopping=True)
    X_train, X_test, y_train, y_test = train_test_split(embeddings, label, test_size=0.2, random_state=42)
    model = mlp.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    result['origin'] = [f1_score(y_test, y_pred, average='macro'), f1_score(y_test, y_pred, average='micro'), f1_score(y_test, y_pred, average='weighted')]
    save_graph(net_c, label, 'network/{}_{}'.format(args.data_name, 'origin'))

    if not os.path.exists('downstream'):
        os.mkdir('downstream')
    print(predicted_noise_link, predicted_missing_link)
    with open('downstream/{}.txt'.format(args.data_name), 'w') as f:
        print('data:{}, num of noisy:{}, num of missing:{}, total size:{}'.format(args.data_name, len(predicted_noise_link[0]), len(predicted_missing_link[0]), len(y_test)))
        f.write('data:{}, num of noisy:{}, num of missing:{}, total size:{}\n'.format(args.data_name, len(predicted_noise_link[0]), len(predicted_missing_link[0]), len(y_test)))
        for key,value in result.items():
            print("{}\tmacro:{}, micro:{}, weighted:{}".format(key, value[0],value[1],value[2]))
            f.write("{}\tmacro:{}, micro:{}, weighted:{}\n".format(key, value[0],value[1],value[2]))





