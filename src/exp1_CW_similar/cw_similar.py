from keras.models import load_model
from utility import create_test_set_AWF_disjoint, kNN_accuracy, create_test_set_AWF_training_included
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from keras.backend.tensorflow_backend import set_session
import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
config.log_device_placement = False  # to log device placement (on which device the operation ran)
sess = tf.Session(config=config)
set_session(sess)  # set this TensorFlow session as the default session for Keras



def AWF_Disjoint_Experment():
    '''
    This function aims to experiment the performance of TF attack
    when the model is trained on the dataset with similar distribution.
    The model is trained on AWF777 and tested on AWF100 and the set of websites
    in the training set and the testing set are mutually exclusive.
    '''
    model_path = '../model_training/trained_model/Triplet_Model.h5'
    features_model = load_model(model_path)
    type_exp = 'N-MEV'
    # N-MEV is the use of mean of embedded vectors as mentioned in the paper
    SOP_list = [100]
    # SOP_list is the size of problem (how large the closed world is)
    # You can run gird search for various sizes of problems
    # SOP_list = [100, 75, 50, 25, 10]
    n_shot_list = [5]
    # n_shot_list is the number of n examples (n-shot)
    # You can run grid search for various sizes of n-shot
    # n_shot_list = [1, 5, 10, 15, 20]

    for size_of_problem in SOP_list:
        print "SOP:", size_of_problem
        for n_shot in n_shot_list:
            acc_list_Top1 = []
            acc_list_Top2 = []
            acc_list_Top5 = []
            for i in range(10):
                signature_vector_dict, test_vector_dict = create_test_set_AWF_disjoint(features_model=features_model,
                                                                                       n_instance=90, max_n=20,
                                                                                       n_shot=n_shot, type_exp=type_exp)
                # Measure the performance (accuracy)
                acc_knn_top1, acc_knn_top2, acc_knn_top5 = kNN_accuracy(signature_vector_dict, test_vector_dict, size_of_problem, n_shot=n_shot)
                acc_list_Top1.append(float("{0:.15f}".format(round(acc_knn_top1, 5))))
                acc_list_Top2.append(float("{0:.15f}".format(round(acc_knn_top2, 5))))
                acc_list_Top5.append(float("{0:.15f}".format(round(acc_knn_top5, 5))))
            print "N_shot:", n_shot
            print str(acc_list_Top1).strip('[]')
            print str(acc_list_Top2).strip('[]')
            print str(acc_list_Top5).strip('[]')


def AWF_TrainingIncluded_Experment():
    '''
    This function aims to experiment the performance of TF attack
    when the model is trained on the dataset with similar distribution.
    The model is trained on AWF777 and tested on AWF100 and the set of websites
    in the training set are partially-to-fully included with the test set.
    '''
    model_path = '../triplet_training/trained_model/Triplet_Model.h5'
    features_model = load_model(model_path)
    type_exp = 'N-MEV'
    # N-MEV is the use of mean of embedded vectors as mentioned in the paper
    training_included_list = [25, 50, 75, 100]
    # training_included_list is the percentatges of the websites in the test set
    # will be included in the training set
    # You can run gird search for various percentages of inclusion
    # training_included_list = [25, 50, 75, 100]
    SOP_list = [100]
    # SOP_list is the size of problem (how large the closed world is)
    # You can run gird search for various sizes of problems
    # SOP_list = [100, 75, 50, 25, 10]
    n_shot_list = [5]
    # n_shot_list is the number of n examples (n-shot)
    # You can run grid search for various sizes of n-shot
    # n_shot_list = [1, 5, 10, 15, 20]
    for training_included in training_included_list:
        print "Training included:", training_included
        for size_of_problem in SOP_list:
            print "SOP:", size_of_problem
            for n_shot in n_shot_list:
                acc_list_Top1 = []
                acc_list_Top2 = []
                acc_list_Top5 = []
                for i in range(10):
                    signature_vector_dict, test_vector_dict = create_test_set_AWF_training_included(features_model=features_model,
                                                                                                    n_instance=90, max_n=20,
                                                                                                    n_shot=n_shot, type_exp=type_exp,
                                                                                                    training_included=training_included)
                    # Measure the performance (accuracy)
                    acc_knn_top1, acc_knn_top2, acc_knn_top5 = kNN_accuracy(signature_vector_dict, test_vector_dict, size_of_problem, n_shot=n_shot)
                    acc_list_Top1.append(float("{0:.15f}".format(round(acc_knn_top1, 5))))
                    acc_list_Top2.append(float("{0:.15f}".format(round(acc_knn_top2, 5))))
                    acc_list_Top5.append(float("{0:.15f}".format(round(acc_knn_top5, 5))))
                print "N_shot:", n_shot

                print str(acc_list_Top1).strip('[]')
                print str(acc_list_Top2).strip('[]')
                print str(acc_list_Top5).strip('[]')


AWF_Disjoint_Experment()
