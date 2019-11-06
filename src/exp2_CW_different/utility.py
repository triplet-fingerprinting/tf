import cPickle as pickle
import os
import random
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def create_test_set_Wang_disjoint(features_model, n_instance, max_n, n_shot, type_exp):

    # Load data
    site_dict = {}
    dataset_dir = '../../dataset/extracted_Wang100/'
    sites = os.listdir(dataset_dir)
    random.shuffle(sites)

    for s in sites:
        site_dict[s] = []
        site_random = list(range(1, n_instance+1))
        random.shuffle(site_random)
        site_random = site_random[:n_instance]
        for ins in site_random:
            ins_num = '{:04}'.format(ins)
            file_name = '%s_%s.pkl' % (s, ins_num)
            with open(dataset_dir + '/' + s + '/' + file_name, 'rb') as handle:
                test_set = pickle.load(handle)
                site_dict[s].append(test_set)

    # create_signature and test_set
    signature_dict = {}
    test_dict = {}
    for s in sites:
        data_set = site_dict[s]
        random.shuffle(data_set)
        signature = []
        for i in range(n_shot):
            signature.append(data_set.pop(0))
        signature_dict[s] = signature
        test_dict[s] = data_set[:n_instance - max_n]

    # Feed signature vector to the model to create embedded signature feature's vectors
    signature_vector_dict = {}
    for i in sites:
        signature_instance = signature_dict[i]
        signature_instance = np.array(signature_instance)
        signature_instance = signature_instance.astype('float32')
        signature_instance = signature_instance[:, :, np.newaxis]
        signature_vector = features_model.predict(signature_instance)
        if type_exp == "N-MEV":
            signature_vector = np.array([signature_vector.mean(axis=0)])
        signature_vector_dict[i] = signature_vector

    # Feed test vector to the model to create embedded test feature's vectors
    test_vector_dict = {}
    for i in sites:
        test_instance = test_dict[i]
        test_instance = np.array(test_instance)
        test_instance = test_instance.astype('float32')
        test_instance = test_instance[:, :, np.newaxis]
        test_vector = features_model.predict(test_instance)
        test_vector_dict[i] = test_vector

    return signature_vector_dict, test_vector_dict

def kNN_accuracy(signature_vector_dict, test_vector_dict, size_of_problem, n_shot):
    X_train = []
    y_train = []

    # print "Size of problem :", size_of_problem
    site_labels = signature_vector_dict.keys()
    random.shuffle(site_labels)
    tested_sites = site_labels[:size_of_problem]
    for s in tested_sites:
        for each_test in range(len(signature_vector_dict[s])):
            X_train.append(signature_vector_dict[s][each_test])
            y_train.append(s)

    X_test = []
    y_test = []
    for s in tested_sites:
        for i in range(len(test_vector_dict[s])):
            X_test.append(test_vector_dict[s][i])
            y_test.append(s)

    knn = KNeighborsClassifier(n_neighbors=n_shot, weights='distance',p=2, metric='cosine', algorithm='brute')
    knn.fit(X_train, y_train)

    acc_knn_top1 = accuracy_score(y_test, knn.predict(X_test))
    acc_knn_top1 = float("{0:.15f}".format(round(acc_knn_top1,6)))
    #print 'KNN accuracy Top1 = ', (acc_knn_top1)

    # Top-2
    count_correct = 0
    for s in range(len(X_test)):
        test_example = X_test[s]
        class_label = y_test[s]
        predict_prob = knn.predict_proba([test_example])
        best_n = np.argsort(predict_prob[0])[-2:]
        class_mapping = knn.classes_
        top_n_list = []
        for p in best_n:
            top_n_list.append(class_mapping[p])
        if class_label in top_n_list:
            count_correct = count_correct + 1
    #print float(count_correct), float(len(X_test))
    acc_knn_top2 = float(count_correct) / float(len(X_test))
    acc_knn_top2 = float("{0:.15f}".format(round(acc_knn_top2, 6)))
    #print 'KNN accuracy Top2 = ', (acc_knn_top2)

    # Top 5
    count_correct = 0
    for s in range(len(X_test)):
        test_example = X_test[s]
        class_label = y_test[s]
        predict_prob = knn.predict_proba([test_example])
        best_n = np.argsort(predict_prob[0])[-5:]
        class_mapping = knn.classes_
        top_n_list = []
        for p in best_n:
            top_n_list.append(class_mapping[p])
        if class_label in top_n_list:
            count_correct = count_correct + 1
    #print float(count_correct), float(len(X_test))
    acc_knn_top5 = float(count_correct) / float(len(X_test))
    acc_knn_top5 = float("{0:.15f}".format(round(acc_knn_top5, 6)))
    #print 'KNN accuracy Top2 = ', (acc_knn_top2)
    return acc_knn_top1, acc_knn_top2, acc_knn_top5
