# DGL Implementation of JKNet

This DGL example implements the GNN model proposed in the paper [Representation Learning on Graphs with Jumping Knowledge Networks](https://arxiv.org/abs/1806.03536).

Contributor: [IPython](https://github.com/IPython1)

### Requirements
The codebase is implemented in Python 3.6. For version requirement of packages, see below.

```
dgl 0.6.0
scikit-learn 0.24.1
tqdm 4.56.0
torch 1.7.1
scipy 1.5.4
numpy 1.18.5
```

### The graph datasets used in this example

###### Node Classification

The DGL's built-in Cora, Citeseer datasets. Dataset summary:

| Dataset | #Nodes | #Edges | #Feats | #Classes | #Train Nodes | #Val Nodes | #Test Nodes |
| :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: |
| Cora | 2,708 | 10,556 | 1,433 | 7(single label) | 60% | 20% | 20% |
| Citeseer | 3,327 | 9,228 | 3,703 | 6(single label) | 60% | 20% | 20% |

### Usage

###### Dataset options
```
--dataset          str     The graph dataset name.             Default is 'Cora'.
```

###### GPU options
```
--gpu              int     GPU index.                          Default is -1, using CPU.
```

###### Model options
```
--run              int     Number of running times.                    Default is 10.
--epochs           int     Number of training epochs.                  Default is 500.
--lr               float   Adam optimizer learning rate.               Default is 0.01.
--lamb             float   L2 regularization coefficient.              Default is 0.0005.
--hid-dim          int     Hidden layer dimensionalities.              Default is 32.
--num-layers       int     Number of T.                                Default is 5.
--mode             str     Type of aggregation ['cat', 'max', 'lstm']. Default is 'cat'.
--dropout          float   Dropout applied at all layers.              Default is 0.5.
```

###### Examples

The following commands learn a neural network and predict on the test set.
Train a JKNet which follows the original hyperparameters on different datasets.
```bash
# Cora:
python main.py --gpu 0 --mode max --num-layers 6
python main.py --gpu 0 --mode cat --num-layers 6
python main.py --gpu 0 --mode lstm --num-layers 1

# Citeseer:
python main.py --gpu 0 --dataset Citeseer --mode max --num-layers 1
python main.py --gpu 0 --dataset Citeseer --mode cat --num-layers 1
python main.py --gpu 0 --dataset Citeseer --mode lstm --num-layers 2
```

### Performance

**As the author does not release the code, we don't have the access to the data splits they used.**

###### Node Classification

* Cora

|  | JK-Maxpool | JK-Concat | JK-LSTM |
| :-: | :-: | :-: | :-: |
| Metrics(Table 2) | 89.6±0.5 | 89.1±1.1 | 85.8±1.0 |
| Metrics(DGL) | 86.1±1.5 | 85.1±1.6 | 84.2±1.6 |

* Citeseer

|  | JK-Maxpool | JK-Concat | JK-LSTM |
| :-: | :-: | :-: | :-: |
| Metrics(Table 2) | 77.7±0.5 | 78.3±0.8 | 74.7±0.9 |
| Metrics(DGL) | 70.9±1.9 | 73.0±1.5 | 69.0±1.7 |

```2024.11.14 训练结果
C:\Users\IPython\.conda\envs\JKN\python.exe main.py --gpu 0 --mode max --num-layers 6 
Using backend: pytorch
Namespace(dataset='Cora', dropout=0.5, epochs=500, gpu=0, hid_dim=32, lamb=0.0005, lr=0.005, mode='max', num_layers=6, run=10)
Downloading C:\Users\IPython\.dgl\cora_v2.zip from https://data.dgl.ai/dataset/cora_v2.zip...
Extracting file to C:\Users\IPython\.dgl\cora_v2
Finished data loading and preprocessing.
  NumNodes: 2708
  NumEdges: 10556
  NumFeats: 1433
  NumClasses: 7
  NumTrainingSamples: 140
  NumValidationSamples: 500
  NumTestSamples: 1000
Done saving data into cached files.
Train Acc 0.8984 | Train Loss 0.3267 | Val Acc 0.8616 | Val loss 0.5594: 100%|██████████| 500/500 [00:25<00:00, 19.88it/s]
Test Acc 0.8598
  NumNodes: 2708
  NumEdges: 10556
  NumFeats: 1433
  NumClasses: 7
  NumTrainingSamples: 140
  NumValidationSamples: 500
  NumTestSamples: 1000
Done loading data from cached files.
Train Acc 0.8978 | Train Loss 0.3331 | Val Acc 0.8155 | Val loss 0.7158: 100%|██████████| 500/500 [00:24<00:00, 20.10it/s]
Test Acc 0.8727
  NumNodes: 2708
  NumEdges: 10556
  NumFeats: 1433
  NumClasses: 7
  NumTrainingSamples: 140
  NumValidationSamples: 500
  NumTestSamples: 1000
Done loading data from cached files.
Train Acc 0.9095 | Train Loss 0.2967 | Val Acc 0.8321 | Val loss 0.5934: 100%|██████████| 500/500 [00:24<00:00, 20.16it/s]
Test Acc 0.8542
  NumNodes: 2708
  NumEdges: 10556
  NumFeats: 1433
  NumClasses: 7
  NumTrainingSamples: 140
  NumValidationSamples: 500
  NumTestSamples: 1000
Done loading data from cached files.
Train Acc 0.8885 | Train Loss 0.3481 | Val Acc 0.8173 | Val loss 0.6513: 100%|██████████| 500/500 [00:24<00:00, 20.38it/s]
Test Acc 0.8672
  NumNodes: 2708
  NumEdges: 10556
  NumFeats: 1433
  NumClasses: 7
  NumTrainingSamples: 140
  NumValidationSamples: 500
  NumTestSamples: 1000
Done loading data from cached files.
Train Acc 0.9175 | Train Loss 0.2732 | Val Acc 0.8339 | Val loss 0.6589: 100%|██████████| 500/500 [00:24<00:00, 20.62it/s]
Test Acc 0.8672
  NumNodes: 2708
  NumEdges: 10556
  NumFeats: 1433
  NumClasses: 7
  NumTrainingSamples: 140
  NumValidationSamples: 500
  NumTestSamples: 1000
Done loading data from cached files.
Train Acc 0.8922 | Train Loss 0.3186 | Val Acc 0.8339 | Val loss 0.5987: 100%|██████████| 500/500 [00:24<00:00, 20.60it/s]
Test Acc 0.8413
  NumNodes: 2708
  NumEdges: 10556
  NumFeats: 1433
  NumClasses: 7
  NumTrainingSamples: 140
  NumValidationSamples: 500
  NumTestSamples: 1000
Done loading data from cached files.
Train Acc 0.9027 | Train Loss 0.3295 | Val Acc 0.8303 | Val loss 0.6615: 100%|██████████| 500/500 [00:24<00:00, 20.19it/s]
Test Acc 0.8487
  NumNodes: 2708
  NumEdges: 10556
  NumFeats: 1433
  NumClasses: 7
  NumTrainingSamples: 140
  NumValidationSamples: 500
  NumTestSamples: 1000
Done loading data from cached files.
Train Acc 0.8953 | Train Loss 0.3165 | Val Acc 0.8376 | Val loss 0.8137: 100%|██████████| 500/500 [00:24<00:00, 20.26it/s]
Test Acc 0.8745
  NumNodes: 2708
  NumEdges: 10556
  NumFeats: 1433
  NumClasses: 7
  NumTrainingSamples: 140
  NumValidationSamples: 500
  NumTestSamples: 1000
Done loading data from cached files.
Train Acc 0.9064 | Train Loss 0.3210 | Val Acc 0.8395 | Val loss 0.7174: 100%|██████████| 500/500 [00:24<00:00, 20.28it/s]
Test Acc 0.8469
  NumNodes: 2708
  NumEdges: 10556
  NumFeats: 1433
  NumClasses: 7
  NumTrainingSamples: 140
  NumValidationSamples: 500
  NumTestSamples: 1000
Done loading data from cached files.
Train Acc 0.8941 | Train Loss 0.3524 | Val Acc 0.8579 | Val loss 0.5681: 100%|██████████| 500/500 [00:25<00:00, 19.97it/s]
Test Acc 0.8598
total acc:  [0.8597785977859779, 0.8726937269372693, 0.8542435424354243, 0.8671586715867159, 0.8671586715867159, 0.8413284132841329, 0.8487084870848709, 0.8745387453874539, 0.8468634686346863, 0.8597785977859779]
mean 0.859
std 0.011

Process finished with exit code 0```
