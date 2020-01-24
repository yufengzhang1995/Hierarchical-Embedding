from __future__ import division
from __future__ import print_function

import time
import os
import sys
import pandas as pd
sys.path.append('.')

# Train on CPU (hide GPU) due to memory constraints
os.environ['CUDA_VISIBLE_DEVICES'] = ""

import tensorflow as tf
import numpy as np
import scipy.sparse as sp
import networkx as nx

from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score

from optimizer import OptimizerAE, OptimizerVAE
from input_data import load_data
from model import GCNModelAE, GCNModelVAE
from preprocessing import preprocess_graph, construct_feed_dict, sparse_to_tuple, mask_test_edges

# Settings
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_float('learning_rate', 0.01, 'Initial learning rate.')
flags.DEFINE_integer('epochs',200, 'Number of epochs to train.')
flags.DEFINE_integer('pow1',20, 'N-hop neighbor to aggregate.')
flags.DEFINE_float('lambd', 0.005, 'Lambd for group lasso.')
flags.DEFINE_integer('hidden1', 32, 'Number of units in hidden layer 1.')
flags.DEFINE_integer('hidden2', 16, 'Number of units in hidden layer 2.')
#flags.DEFINE_list('hidden1', [32,32,32], 'Number of units in hidden layer 1.')
#flags.DEFINE_list('hidden2', [16,16,16], 'Number of units in hidden layer 2.')
flags.DEFINE_integer('budget_hidden1', 96, 'neuron architecure for hidden1 layer')
flags.DEFINE_integer('budget_embed', 48, 'neuron architecure for embedding layer')
flags.DEFINE_float('weight_decay', 0., 'Weight for L2 loss on embedding matrix.')
flags.DEFINE_float('dropout', 0., 'Dropout rate (1 - keep probability).')
flags.DEFINE_float('cut_off', 0.0045, 'Threshold to throw away which columns in weight matrices')

flags.DEFINE_string('model', 'gcn_ae', 'Model string.')
flags.DEFINE_string('filename_adj', 'chrx_adj.txt', 'Dataset with txt extension string.')
flags.DEFINE_string('filename_feat', 'chrx_feat.txt', 'Dataset with txt extension string.')
flags.DEFINE_integer('features', 0, 'Whether to use features (1) or not (0).')

model_str = FLAGS.model
filename_adj = FLAGS.filename_adj
filename_feat = FLAGS.filename_feat
# Load data
# adj, features = load_data(dataset_str)
G1 = nx.read_edgelist(filename_adj,
                     nodetype = int,
                     data = (('weight',float),),
                     create_using = nx.DiGraph())
adj = nx.to_scipy_sparse_matrix(G1)

G2 = nx.read_edgelist(filename_feat,
                     nodetype = int,
                     data = (('weight',int),),
                     create_using = nx.DiGraph())
features = nx.to_scipy_sparse_matrix(G2)

# Store original adjacency matrix (without diagonal entries) for later
adj_orig = adj
adj_orig = adj_orig - sp.dia_matrix((adj_orig.diagonal()[np.newaxis, :], [0]), shape=adj_orig.shape)
adj_orig.eliminate_zeros()

adj_train, train_edges, val_edges, val_edges_false, test_edges, test_edges_false = mask_test_edges(adj)
adj = adj_train

if FLAGS.features == 0:
    features = sp.identity(features.shape[0])  # featureless

# Some preprocessing
adj_norm = preprocess_graph(adj)

# Define placeholders
placeholders = {
    'features': tf.sparse_placeholder(tf.float32),
    'adj': tf.sparse_placeholder(tf.float32),
    'adj_orig': tf.sparse_placeholder(tf.float32),
    'dropout': tf.placeholder_with_default(0., shape=())
}

num_nodes = adj.shape[0]

features = sparse_to_tuple(features.tocoo())
num_features = features[2][1]
features_nonzero = features[1].shape[0]

# Create model
model = None
if model_str == 'gcn_ae':
    model = GCNModelAE(placeholders, num_features, features_nonzero)
elif model_str == 'gcn_vae':
    model = GCNModelVAE(placeholders, num_features, num_nodes, features_nonzero)

pos_weight = float(adj.shape[0] * adj.shape[0] - adj.sum()) / adj.sum()
norm = adj.shape[0] * adj.shape[0] / float((adj.shape[0] * adj.shape[0] - adj.sum()) * 2)

# Optimizer
with tf.name_scope('optimizer'):
    if model_str == 'gcn_ae':
        opt = OptimizerAE(preds=model.reconstructions,
                          labels=tf.reshape(tf.sparse_tensor_to_dense(placeholders['adj_orig'],
                                                                      validate_indices=False), [-1]),
                          model = model,
                          pos_weight=pos_weight,
                          norm=norm,
                          lambd = FLAGS.lambd,
                          base = False)
    elif model_str == 'gcn_vae':
        opt = OptimizerVAE(preds=model.reconstructions,
                           labels=tf.reshape(tf.sparse_tensor_to_dense(placeholders['adj_orig'],
                                                                       validate_indices=False), [-1]),
                           model=model, num_nodes=num_nodes,
                           pos_weight=pos_weight,
                           norm=norm)

# Initialize session
sess = tf.Session()
sess.run(tf.global_variables_initializer())

cost_val = []
acc_val = []


def get_roc_score(edges_pos, edges_neg, emb=None):
    if emb is None:
        feed_dict.update({placeholders['dropout']: 0})
        emb = sess.run(model.z_mean, feed_dict=feed_dict)

    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    # Predict on test set of edges
    adj_rec = np.dot(emb, emb.T)
    preds = []
    pos = []
    for e in edges_pos:
        preds.append(sigmoid(adj_rec[e[0], e[1]]))
        pos.append(adj_orig[e[0], e[1]])

    preds_neg = []
    neg = []
    for e in edges_neg:
        preds_neg.append(sigmoid(adj_rec[e[0], e[1]]))
        neg.append(adj_orig[e[0], e[1]])

    preds_all = np.hstack([preds, preds_neg])
    labels_all = np.hstack([np.ones(len(preds)), np.zeros(len(preds_neg))])
    roc_score = roc_auc_score(labels_all, preds_all)
    ap_score = average_precision_score(labels_all, preds_all)

    return roc_score, ap_score


cost_val = []
acc_val = []
val_roc_score = []

adj_label = adj_train + sp.eye(adj_train.shape[0])
adj_label = sparse_to_tuple(adj_label)

# Train model
for epoch in range(FLAGS.epochs):

    t = time.time()
    # Construct feed dictionary
    feed_dict = construct_feed_dict(adj_norm, adj_label, features, placeholders)
    feed_dict.update({placeholders['dropout']: FLAGS.dropout})
    # Run single weight update
    outs = sess.run([opt.opt_op, opt.cost, opt.accuracy], feed_dict=feed_dict)

    # Compute average loss
    avg_cost = outs[1]
    avg_accuracy = outs[2]

    roc_curr, ap_curr = get_roc_score(val_edges, val_edges_false)
    val_roc_score.append(roc_curr)

    print("Epoch:", '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(avg_cost),
          "train_acc=", "{:.5f}".format(avg_accuracy), "val_roc=", "{:.5f}".format(val_roc_score[-1]),
          "val_ap=", "{:.5f}".format(ap_curr),
          "time=", "{:.5f}".format(time.time() - t))

print("Optimization round one Finished!")

name_list = ['weights0','weights1','weights2']
layer_sizes = dict()
layer_sizes["hidden1"] = []
layer_sizes["embedding"] = []

def evaluate_architecture(model):
    # Making a choice about the optimal layer sizes.
    for i in range(3):
        norms = tf.norm(model.weight_matrix_hidden1[name_list[i]], ord = 2, axis = 0)
        norms = sess.run(norms)
        norms = norms[norms>FLAGS.cut_off]
        layer_sizes["hidden1"].append(norms.shape[0])
    for i in range(3):
        norms = tf.norm(norms = tf.norm(model.weight_matrix_embeddings[name_list[i]], ord = 2, axis = 0))
        norms = sess.run(norms)
        norms = norms[norms>FLAGS.cut_off]
        layer_sizes["embedding"].append(norms.shape[0])

    layer_sizes["hidden1"] = [int(FLAGS.budget_hidden1*layer_size/sum(layer_sizes["hidden1"]))  for layer_size in layer_sizes["hidden1"]]
    layer_sizes["embedding"] = [int(FLAGS.budget_embed*layer_size/sum(layer_sizes["embedding"]))  for layer_size in layer_sizes["embedding"]]

    dict_temp_1 = pd.DataFrame(layer_sizes)
    dict_temp_1.to_csv(f'layer_sizes_1_{FLAGS.pow1}.csv',index = True, header = True, sep = ',')

    print("layer resizing finished!")
evaluate_architecture(model)
"""
def reset_architecture(self):# Changing the weight sizes.
    FLAGS.hidden1 = self.layer_sizes["hidden1"]
    FLAGS.hidden2 = self.layer_sizes["embedding"]
    return self.args


evaluate_architecture(model)

# Train model
for epoch in range(FLAGS.epochs):
    
    t = time.time()
    # Construct feed dictionary
    feed_dict = construct_feed_dict(adj_norm, adj_label, features, placeholders)
    feed_dict.update({placeholders['dropout']: FLAGS.dropout})
    # Run single weight update
    outs = sess.run([opt.opt_op, opt.cost, opt.accuracy], feed_dict=feed_dict)
    
    # Compute average loss
    avg_cost = outs[1]
    avg_accuracy = outs[2]
    
    roc_curr, ap_curr = get_roc_score(val_edges, val_edges_false)
    val_roc_score.append(roc_curr)
    
    print("Epoch:", '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(avg_cost),
          "train_acc=", "{:.5f}".format(avg_accuracy), "val_roc=", "{:.5f}".format(val_roc_score[-1]),
          "val_ap=", "{:.5f}".format(ap_curr),
          "time=", "{:.5f}".format(time.time() - t))

print("Optimization Finished!")
"""








final_emb = sess.run(model.z_mean, feed_dict=feed_dict)
final_embdf = pd.DataFrame(final_emb)
final_embdf.to_csv(f'embeddings_{FLAGS.pow1}.csv',index = True, header = True, sep = ',')

# write out six weight matrix
name_list = ['weights0','weights1','weights2']

final_weight_hidden1 = sess.run(model.weight_matrix_hidden1, feed_dict=feed_dict)
for i in range(3):
    temp = pd.DataFrame(final_weight_hidden1[name_list[i]])
    temp.to_csv(f'hidden1_wieght_{i}_{FLAGS.pow1}.csv',index = True, header = True, sep = ',')

final_weight_embed = sess.run(model.weight_matrix_embeddings, feed_dict=feed_dict)
for i in range(3):
    temp = pd.DataFrame(final_weight_hidden1[name_list[i]])
    temp.to_csv(f'embed_wieght_{i}_{FLAGS.pow1}.csv',index = True, header = True, sep = ',')

roc_score, ap_score = get_roc_score(test_edges, test_edges_false)
print('Test ROC score: ' + str(roc_score))
print('Test AP score: ' + str(ap_score))
