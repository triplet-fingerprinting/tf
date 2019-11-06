import os
import random
import numpy as np
import cPickle as pickle
import keras.backend as K
from keras.models import Model
from keras.layers import Input, Lambda, Dot
from keras import optimizers
from keras.callbacks import CSVLogger
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from keras.backend.tensorflow_backend import set_session
import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
config.log_device_placement = False  # to log device placement (on which device the operation ran)
sess = tf.Session(config=config)
set_session(sess)  # set this TensorFlow session as the default session for Keras

# This is tuned hyper-parameters
alpha = 0.1
batch_size_value = 128
emb_size = 64
number_epoch = 30

description = 'Triplet_Model'
Training_Data_PATH = '../../dataset/extracted_AWF775/'
print Training_Data_PATH
print "with parameters, Alpha: %s, Batch_size: %s, Embedded_size: %s, Epoch_num: %s"%(alpha, batch_size_value, emb_size, number_epoch)


alpha_value = float(alpha)
print description

# ================================================================================
# This part is to prepare the files' index for geenrating triplet examples
# and formulating each epoch inputs

# Extract all folders' names
dirs = sorted(os.listdir(Training_Data_PATH))

# Each given folder name (URL of each class), we assign class id
# e.g. {'adp.com' : 23, ...}
name_to_classid = {d:i for i,d in enumerate(dirs)}

# Just reverse from previous step
# Each given class id, show the folder name (URL of each class)
# e.g. {23 : 'adp.com', ...}
classid_to_name = {v:k for k,v in name_to_classid.items()}

num_classes = len(name_to_classid)
print "number of classes: "+str(num_classes)

# Each directory, there are n traces corresponding to the identity
# We map each trace path with an integer id, then build dictionaries
# We are mapping
#   path_to_id and id_to_path
#   classid_to_ids and id_to_classid

# read all directories
# c is class
# name_to_classid.items() contains [(directory, classid), ('slickdeals.net', 547), ...]

trace_paths = {c:[directory + "/" + img for img in sorted(os.listdir(Training_Data_PATH + directory))]
         for directory,c in name_to_classid.items()}
# trace_paths --> {0: ['104.com.tw/104.com.tw_0001.pkl', '104.com.tw/104.com.tw_0002.pkl',...] ,....}

# retreive all traces
# to create the list of all traces paths
all_traces_path = []
for trace_list in trace_paths.values():
    all_traces_path += trace_list
# all_trace_path --> ['104.com.tw/104.com.tw_0001.pkl', '104.com.tw/104.com.tw_0002.pkl',...]
# len(all_trace_path = num_class * num_examples e.g. 700 * 25

# map to integers
# just map each path to sequence of ID (from 1 to len(all_trace_path)
path_to_id = {v: k for k, v in enumerate(all_traces_path)}
# path_to_id --> {'chron.com/chron.com_0006.pkl': 1185, 'habrahabr.ru/habrahabr.ru_0001.pkl': 2680, ...}
id_to_path = {v: k for k, v in path_to_id.items()}
# id_to_path --> {0: '104.com.tw/104.com.tw_0001.pkl', 1: '104.com.tw/104.com.tw_0002.pkl', ...}

# build mapping between traces and class
classid_to_ids = {k: [path_to_id[path] for path in v] for k, v in trace_paths.items()}
# classid_to_ids --> {0: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], 1: [10, 11, 12, 13, 14, 15, 16, 17, 18, 19],...]
id_to_classid = {v: c for c, traces in classid_to_ids.items() for v in traces}
# id_to_classid --> {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0, 10: 1,...]

# open trace
all_traces = []
for path in id_to_path.values():
    each_path = Training_Data_PATH + path
    with open(each_path, 'rb') as handle:
        each_trace = pickle.load(handle)
    all_traces += [each_trace]

all_traces = np.vstack((all_traces))
all_traces = all_traces[:, :, np.newaxis]
print "Load traces with ",all_traces.shape
print "Total size allocated on RAM : ", str(all_traces.nbytes / 1e6) + ' MB'

def build_pos_pairs_for_id(classid): # classid --> e.g. 0
    traces = classid_to_ids[classid]
    # pos_pairs is actually the combination C(10,2)
    # e.g. if we have 10 example [0,1,2,...,9]
    # and want to create a pair [a, b], where (a, b) are different and order does not matter
    # e.g. [(0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (0, 6), (0, 7), (0, 8), (0, 9),
    # (1, 2), (1, 3), (1, 4), (1, 5), (1, 6), (1, 7), (1, 8), (1, 9)...]
    # C(10, 2) = 45
    pos_pairs = [(traces[i], traces[j]) for i in range(len(traces)) for j in range(i+1, len(traces))]
    random.shuffle(pos_pairs)
    return pos_pairs

def build_positive_pairs(class_id_range):
    # class_id_range = range(0, num_classes)
    listX1 = []
    listX2 = []
    for class_id in class_id_range:
        pos = build_pos_pairs_for_id(class_id)
        # -- pos [(1, 9), (0, 9), (3, 9), (4, 8), (1, 4),...] --> (anchor example, positive example)
        for pair in pos:
            listX1 += [pair[0]] # identity
            listX2 += [pair[1]] # positive example
    perm = np.random.permutation(len(listX1))
    # random.permutation([1,2,3]) --> [2,1,3] just random
    # random.permutation(5) --> [1,0,4,3,2]
    # In this case, we just create the random index
    # Then return pairs of (identity, positive example)
    # that each element in pairs in term of its index is randomly ordered.
    return np.array(listX1)[perm], np.array(listX2)[perm]

Xa_train, Xp_train = build_positive_pairs(range(0, num_classes))

# Gather the ids of all network traces that are used for training
# This just union of two sets set(A) | set(B)
all_traces_train_idx = list(set(Xa_train) | set(Xp_train))
print "X_train Anchor: ", Xa_train.shape
print "X_train Positive: ", Xp_train.shape

# Build a loss which doesn't take into account the y_true, as# Build
# we'll be passing only 0
def identity_loss(y_true, y_pred):
    return K.mean(y_pred - 0 * y_true)

# The real loss is here
def cosine_triplet_loss(X):
    _alpha = alpha_value
    positive_sim, negative_sim = X

    losses = K.maximum(0.0, negative_sim - positive_sim + _alpha)
    return K.mean(losses)

# ------------------- Hard Triplet Mining -----------
# Naive way to compute all similarities between all network traces.

def build_similarities(conv, all_imgs):
    embs = conv.predict(all_imgs)
    embs = embs / np.linalg.norm(embs, axis=-1, keepdims=True)
    all_sims = np.dot(embs, embs.T)
    return all_sims

def intersect(a, b):
    return list(set(a) & set(b))

def build_negatives(anc_idxs, pos_idxs, similarities, neg_imgs_idx, num_retries=50):
    # If no similarities were computed, return a random negative
    if similarities is None:
        return random.sample(neg_imgs_idx,len(anc_idxs))
    final_neg = []
    # for each positive pair
    for (anc_idx, pos_idx) in zip(anc_idxs, pos_idxs):
        anchor_class = id_to_classid[anc_idx]
        #positive similarity
        sim = similarities[anc_idx, pos_idx]
        # find all negatives which are semi(hard)
        possible_ids = np.where((similarities[anc_idx] + alpha_value) > sim)[0]
        possible_ids = intersect(neg_imgs_idx, possible_ids)
        appended = False
        for iteration in range(num_retries):
            if len(possible_ids) == 0:
                break
            idx_neg = random.choice(possible_ids)
            if id_to_classid[idx_neg] != anchor_class:
                final_neg.append(idx_neg)
                appended = True
                break
        if not appended:
            final_neg.append(random.choice(neg_imgs_idx))
    return final_neg


class SemiHardTripletGenerator():
    def __init__(self, Xa_train, Xp_train, batch_size, all_traces, neg_traces_idx, conv):
        self.batch_size = batch_size

        self.traces = all_traces
        self.Xa = Xa_train
        self.Xp = Xp_train
        self.cur_train_index = 0
        self.num_samples = Xa_train.shape[0]
        self.neg_traces_idx = neg_traces_idx
        self.all_anchors = list(set(Xa_train))
        self.mapping_pos = {v: k for k, v in enumerate(self.all_anchors)}
        self.mapping_neg = {k: v for k, v in enumerate(self.neg_traces_idx)}
        if conv:
            self.similarities = build_similarities(conv, self.traces)
        else:
            self.similarities = None

    def next_train(self):
        while 1:
            self.cur_train_index += self.batch_size
            if self.cur_train_index >= self.num_samples:
                self.cur_train_index = 0

            # fill one batch
            traces_a = self.Xa[self.cur_train_index:self.cur_train_index + self.batch_size]
            traces_p = self.Xp[self.cur_train_index:self.cur_train_index + self.batch_size]
            traces_n = build_negatives(traces_a, traces_p, self.similarities, self.neg_traces_idx)

            yield ([self.traces[traces_a],
                    self.traces[traces_p],
                    self.traces[traces_n]],
                   np.zeros(shape=(traces_a.shape[0]))
                   )

# Training the Triplet Model
from DF_model import DF
shared_conv2 = DF(input_shape=(5000,1), emb_size=emb_size)

anchor = Input((5000, 1), name='anchor')
positive = Input((5000, 1), name='positive')
negative = Input((5000, 1), name='negative')

a = shared_conv2(anchor)
p = shared_conv2(positive)
n = shared_conv2(negative)

# The Dot layer in Keras now supports built-in Cosine similarity using the normalize = True parameter.
# From the Keras Docs:
# keras.layers.Dot(axes, normalize=True)
# normalize: Whether to L2-normalize samples along the dot product axis before taking the dot product.
#  If set to True, then the output of the dot product is the cosine proximity between the two samples.
pos_sim = Dot(axes=-1, normalize=True)([a,p])
neg_sim = Dot(axes=-1, normalize=True)([a,n])

# customized loss
loss = Lambda(cosine_triplet_loss,
              output_shape=(1,))(
             [pos_sim,neg_sim])

model_triplet = Model(
    inputs=[anchor, positive, negative],
    outputs=loss)
print model_triplet.summary()

opt = optimizers.SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)

model_triplet.compile(loss=identity_loss, optimizer=opt)

batch_size = batch_size_value
# At first epoch we don't generate hard triplets
gen_hard = SemiHardTripletGenerator(Xa_train, Xp_train, batch_size, all_traces, all_traces_train_idx, None)
nb_epochs = number_epoch
csv_logger = CSVLogger('log/Training_Log_%s.csv'%description, append=True, separator=';')
for epoch in range(nb_epochs):
    print("built new hard generator for epoch "+str(epoch))
    model_triplet.fit_generator(generator=gen_hard.next_train(),
                    steps_per_epoch=Xa_train.shape[0] // batch_size,
                    epochs=1, verbose=1, callbacks=[csv_logger])
    gen_hard = SemiHardTripletGenerator(Xa_train, Xp_train, batch_size, all_traces, all_traces_train_idx, shared_conv2)
    #For no semi-hard_triplet
    #gen_hard = HardTripletGenerator(Xa_train, Xp_train, batch_size, all_traces, all_traces_train_idx, None)
shared_conv2.save('trained_model/%s.h5'%description)
