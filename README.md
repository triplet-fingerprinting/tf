# Triplet Fingerprinting Attack
:warning: experimental - PLEASE BE CAREFUL. Intended for reasearch purposes only.

The source code and dataset are used to demonstrate the Triplet Fingerprinting (TF) model, and reproduce the results of the ACM CCS2019 paper:

## ACM Reference Format
```
Payap Sirinam, Nate Mathews, Mohammad Saidur Rahman, and Matthew Wright. 2019. 
Triplet Fingerprinting: More Practical and Portable Website Fingerprinting with N-shot Learning. 
In 2019 ACM SIGSAC Conference on Computer and Communications Security (CCS ’19), 
November 11–15, 2019, London, United Kingdom. ACM, New York, NY, USA, 18 pages. 
https://doi.org/10.1145/3319535.3354217
```
## Dataset
We perform our experiments using datasets provided by other re-searchers and used in the previous literature. We label these datasetsas follows:
1. AWF dataset[1]: This dataset includes monitored websites fromthe 1,200 Alexa Top sites, and unmonitored websites from the 400,000 Alexa Top sites. The dataset was collected in 2016.
2. Wang dataset[2].This dataset contains a set of monitored web-sites selected from a list of sites blocked in China, the UK, and SaudiArabia. The unmonitored websites were chosen from the Alexa Topsites. The dataset was collected in 2013.
3. DF dataset[3].This dataset consists of both monitored andunmonitored websites crawled from the Alexa Top sites. As withthe AWF dataset, this dataset was collected in 2016 

Please find the full detail of the categorization of datasets and their names used in each experiment in our published paper. Moreover, please properly cite these papers in your work as we use their datasets. 

```
[1] Vera Rimmer, Davy Preuveneers, Marc Juarez, Tom Van Goethem, and WouterJoosen. 2018.  
Automated Website Fingerprinting through Deep Learning. In Proceedings of the 25nd Network 
and Distributed System Security Symposium (NDSS 2018). Internet Society.
[2] Tao Wang, Xiang Cai, Rishab Nithyanand, Rob Johnson, and Ian Goldberg. 2014.
Effective attacks and provable defenses for website fingerprinting. In USENIX 
Security Symposium. USENIX Association, 143–157.
[3] Payap Sirinam, Mohsen Imani, Marc Juarez, and Matthew Wright. 2018. 
DeepFingerprinting: Undermining Website Fingerprinting Defenses with Deep Learning.
The 25th ACM SIGSAC Conference on Computer and Communications Security(CCS ’18)(2018).
```

## Data Representation.
The data used for training and testing the model consists of network traffic examples from various sources of data as mentioned above. All examples are converted to sequences in which we ignore packet size and timestamps and only store the traffic direction of each packet, with +1 and -1 representing outgoing and incoming packets, respectively.

The sequences are trimmed or padded with 0’s as need to reach a fixed length of 5,000 packets. Thus, the input forms a 1-D array of [1 x 5000].

## Dataset Format
In all datasets, we use the same data structure as following:
Each folder's name represents the name of website e.g. abc.com
     In each folder, it contains files, each files contains the array of network traffic sequences
     e.g.  abc.com_<sequence_#) where sequence_# represent the sequence of network traffic's examples for each website.
           The dimension of each network sequence follows the data representation above.

# Reproduce Results
- We publish the datasets of web traffic traces produced for each experiment. However, due to the limitation on the size of uploaded files set by GitHub, we upload our dataset to google drive repository instead. The dataset can be downloaded from this [link](https://drive.google.com/drive/folders/1Rp1m7yWsoEP4ax79BO6uaUSMmVkrWdtY?usp=sharing). For the source codes, you can directly download from the GitHub's files.
- You will need to download the corresponding source codes and datasets and places them into your machine, please make sure that the tree of folders are consistent with what is shown in the GitHub files.

## Similar but mutually exclusive datasets
The first experiment evaluates the attack scenario in which the attacker pre-trains the feature extractor on one dataset and performs
classification on a different dataset with different classes (Disjointed websites). More precisely, the websites’ URLs used during the pretraining phase and the attack phase are mutually exclusive. In this scenario, the training and testing datasets have a similar distribution in that they are both collected with from the same period of time (2016) using the same version of TBB (6.X).

We train the feature extractor by using the AWF775 dataset and test classification performance on AWF100. During the training phase, we randomly sampled 25 examples for each website in the AWF775 dataset using the semi-hard-negative mining strategy to formulate 232,500 triplets

### Training TF Model
We first train the TF model from the AWF777 dataset

```
python src/model_training/model_training.py
```

The output will be

```
../../dataset/extracted_AWF775/
with parameters, Alpha: 0.1, Batch_size: 128, Embedded_size: 64, Epoch_num: 30
Triplet_Model
number of classes: 775
Load traces with  (19375, 5000, 1)
Total size allocated on RAM :  96.875 MB
X_train Anchor:  (232500,)
X_train Positive:  (232500,)
Model: "model_2"
__________________________________________________________________________________________________
Layer (type)                    Output Shape         Param #     Connected to                     
==================================================================================================
anchor (InputLayer)             (None, 5000, 1)      0                                            
__________________________________________________________________________________________________
positive (InputLayer)           (None, 5000, 1)      0                                            
__________________________________________________________________________________________________
negative (InputLayer)           (None, 5000, 1)      0                                            
__________________________________________________________________________________________________
model_1 (Model)                 (None, 64)           1369344     anchor[0][0]                     
                                                                 positive[0][0]                   
                                                                 negative[0][0]                   
__________________________________________________________________________________________________
dot_1 (Dot)                     (None, 1)            0           model_1[1][0]                    
                                                                 model_1[2][0]                    
__________________________________________________________________________________________________
dot_2 (Dot)                     (None, 1)            0           model_1[1][0]                    
                                                                 model_1[3][0]                    
__________________________________________________________________________________________________
lambda_1 (Lambda)               (None, 1)            0           dot_1[0][0]                      
                                                                 dot_2[0][0]                      
==================================================================================================
Total params: 1,369,344
Trainable params: 1,369,344
Non-trainable params: 0
__________________________________________________________________________________________________
Epoch 1/1
1816/1816 [==============================] - 352s 194ms/step - loss: 0.0173
built new hard generator for epoch 1
Epoch 1/1
1816/1816 [==============================] - 358s 197ms/step - loss: 0.0428
built new hard generator for epoch 2
Epoch 1/1
1816/1816 [==============================] - 372s 205ms/step - loss: 0.0542
built new hard generator for epoch 3
Epoch 1/1
1816/1816 [==============================] - 362s 199ms/step - loss: 0.0606

...

1816/1816 [==============================] - 354s 195ms/step - loss: 0.0262
built new hard generator for epoch 29
Epoch 1/1
1816/1816 [==============================] - 354s 195ms/step - loss: 0.0247

```

The trained model will be saved in */trained_model/Triplet_Model.h5* and this model will be used in the subsequent evaluations.
Then, we use the trained model to evaluate the performance of WF attacks under the scenario that the distributions of training and testing datasets is similar but mutually exclusive datasets 

```
python src/exp1_CW_similar/cw_similar.py
```

The results will be shown by this format:

```
<SOP: N> 
<N-Shot: N>
<TOP1_Acc1> ... <TOP1_ACC10>
<TOP2_Acc1> ... <TOP2_ACC10>
<TOP5_Acc1> ... <TOP5_ACC10>
```
The interpretation of the format shown about is that
<SOP: N> is the size of problem -- Number of N websites to be inclded in the closed world (How large the close-world is)
<N-Shot: N> -- Number of N examples used in N-Shot e.g. 5-Shot
We evaluate 10 times of running for Top-1, Top-2, and Top-5 Accuracy
e.g. <Top1_Acc1> is the Top1 Accuracy#1

The example of output:

```
SOP:100
N_shot:5
0.92714,0.90986,0.92457,0.92829,0.91957,0.92014,0.92571,0.92271,0.91643,0.92957
0.959,0.94957,0.95686,0.96129,0.96,0.95814,0.959,0.95914,0.95271,0.96157
0.97614,0.97629,0.97757,0.97871,0.97729,0.97614,0.97929,0.97529,0.97557,0.97886
```
The result above shows the accuracies of 5-Shot Learning with 100 websites in the closed world.
The first line demonstrates 10 values of accuracies for 10-time runnings of the Top-1 Accuracies.
Thus, the average accuracy in this case is ~92% Accuracy

## WF attacks with different data distributions
Next, we evaluate the performance of the WF attack under the scenario in which the pre-training and classification datasets are collected at different times with different Tor Browser Bundles (TBB), leading to the data having different distributions.

We use the same triplet model from the first experiment trained with the AWF775 dataset for feature extraction. To N-train the k-NN classifier, however, we use the Wang100 dataset, which was collected three years prior to the collection of AWF775 using a much older TBB.

```
python src/exp2_CW_different/cw_different.py
```
The example of output:

```
SOP: 100
N_shot: 5
0.841, 0.84943, 0.841, 0.84171, 0.85057, 0.84643, 0.84086, 0.84743, 0.84557, 0.84943
0.89829, 0.89543, 0.89114, 0.89271, 0.90057, 0.895, 0.89643, 0.90457, 0.90214, 0.90214
0.95443, 0.94671, 0.946, 0.94257, 0.95671, 0.94471, 0.94757, 0.95486, 0.953, 0.95243
```
The interpretation of the result is also same as the previous experiment.
Thus, the average accuracy in this case is ~84% Accuracy for 5-Shot Learning

:information_source: The further guideline for other experiments will be posted soon.

# Questions and comments
Please, address any questions or comments to the authors of the paper. The main developers of this code are:

* Payap Sirinam ([payap_siri@rtaf.mi.th](mailto:payap_siri@rtaf.mi.th))
* Nate Mathews ([nate.mathews@mail.rit.edu](mailto:nate.mathews@mail.rit.edu)) 
* Mohammad Saidur Rahman ([saidur.rahman@mail.rit.edu](mailto:saidur.rahman@mail.rit.edu)) 
* Matthew Wright ([matthew.wright@rit.edu](mailto:matthew.wright@rit.edu)) 
