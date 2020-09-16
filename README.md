# GraphNorm: A Principled Approach to Accelerating Graph Neural Network Training

[Tianle Cai*](https://tianle.website/), [Shengjie Luo*](https://github.com/lsj2408/), [Keyulu Xu](http://keyulux.com/), [Di He](https://www.microsoft.com/en-us/research/people/dihe/), [Tie-Yan Liu](https://www.microsoft.com/en-us/research/people/tyliu/), [Liwei Wang](http://www.liweiwang-pku.com/)

This repository is the official implementation of [GraphNorm: A Principled Approach to Accelerating Graph Neural Network Training](https://arxiv.org/abs/2009.03294), based on the *PyTorch* and *DGL* library.

**Contact**

Tianle Cai (caitianle1998@pku.edu.cn), Shengjie Luo (luosj@stu.pku.edu.cn)

Sincerely appreciate your suggestions on our work!

## Overview

GraphNorm is a principled normalization method that accelerates the GNNs training on graph classification tasks, where the key idea is to normalize all nodes for each individual graph with a learnable shift. Theoretically, we show that GraphNorm serves as a preconditioner that smooths the distribution of the graph aggregation's spectrum, and the learnable shift is used to improve the expressiveness of the networks. Empirically, we conduct experiments on several popular benckmark datasets, including the recently released [Open Graph Benchmark](https://ogb.stanford.edu/docs/leader_graphprop/). Results on datasets with different scale consistently show that GNNs with GraphNorm converge much faster and achieve better generalization performance.

## Installation

1. Clone this repository

```shell
git clone https://github.com/lsj2408/GraphNorm.git
```

2. Install the dependencies (Python 3.6.8)

```shell
pip install -r requirements.txt
```

## Examples

### Results for Bioinformatics and Social Network datasets

### Training

To reproduce Figure 4 in our paper, run as follows:

```shell
cd ./GraphNorm_ws/gnn_ws/gnn_example/scripts/example-train-comparison
```

- **Bioinformatics** datasets: MUTAG, PTC, PROTEINS, NCI1

  ```shell
  ./gin-train-bioinformatics.sh
  ./gcn-train-bioinformatics.sh
  ```

- **Social Network** datasets

  - REDDIT-BINARY

    ```shell
    ./gin-train-rdtb.sh
    ./gcn-train-rdtb.sh
    ```

  - IMDB-BINARY

    ```shell
    ./gin-train-imdbb.sh
    ./gcn-train-imdbb.sh
    ```

  - COLLAB

    ```shell
    ./gin-train-collab.sh
    ./gcn-train-collab.sh
    ```

Results are stored in `./gnn_ws/log/Example-train-performance/`, you can use the recorded metric numbers to plot the training curves as Figure 4 in our paper.

### Testing

Here we provide examples to reproduce the test results on Bioinformatics and Social Network datasets.

```shell
cd ./GraphNorm_ws/gnn_ws/gnn_example/scripts/example-test-comparison
```

- PROTEINS:

  ```shell
  ./gin-train-proteins.sh
  ```

- REDDIT-BINARY:

  ```shell
  ./gin-train-rdtb.sh
  ```

Results are stored in `./gnn_ws/log/Example-test-performance/`. For further results on other datasets, follow the configurations on Appendix C.

### Results for Ogbg-molhiv

### Training

Training process is performed with **10 different random seeds** 

- GCN

  ```shell
  cd ./GraphNorm_ws/ogbg_ws/scripts/Example-gcn-test-performance
  ./seed-1-5-gcn_run.sh
  ./seed-6-10-gcn_run.sh
  ```

- GIN

  ```shell
  cd ./GraphNorm_ws/ogbg_ws/scripts/Example-gin-test-performance
  ./seed-1-5-gin_run.sh
  ./seed-6-10-gin_run.sh
  ```

Results can be found in :

- `./GraphNorm_ws/ogbg_ws/log/`:  recorded metric values along training process,
- `./GraphNorm_ws/ogbg_ws/model`: model checkpoints, which has the maximum validation metric values along training process.

**Notes**:

The number of epoch as 20~30 is fine for GIN with GraphNorm.

### Evaluation

For evaluation, we use the dumped model checkpoints and report the mean and standard deviation metric values.

- Use your **own trained models**

  Set the `MODEL_PATH` in `./ogbg_ws/scripts/evaluate_gin.sh` / `./ogbg_ws/scripts/evaluate_gcn.sh` to the desired item.

- Use provided **pre-trained models**

  We provide **the TOP-1 model on OGBG-MOLHIV** datasets [here](https://1drv.ms/u/s!AgZyC7AzHtDBbZO5a6g6esOcuoQ?e=qGKkLg). The training command is provided below.

  ```shell
  #!/usr/bin/env bash
  
  set -e
  
  GPU=0
  NORM=gn
  BS=128
  DP=0.1
  EPOCH=50
  Layer=6
  LR=0.0001
  HIDDEN=300
  DS=ogbg-molhiv
  LOG_PATH=../../log/"$NORM"-BS-"$BS"-EPOCH-"$EPOCH"-L-"$Layer"-HIDDEN-"$HIDDEN"-LR-"$LR"-decay/
  MODEL_PATH=../../model/"$NORM"-BS-"$BS"-EPOCH-"$EPOCH"-L-"$Layer"-HIDDEN-"$HIDDEN"-LR-"$LR"-decay/
  DATA_PATH=../../data/dataset/
  
  
  for seed in {1..10}; do
      FILE_NAME=learn-"$DS"-gcn-seed-"$seed"
      python ../../src/train_dgl_ogb.py \
          --gpu $GPU \
          --epoch $EPOCH \
          --dropout $DP \
          --model GCN_dp \
          --batch_size $BS \
          --n_layers $Layer \
          --lr $LR \
          --n_hidden $HIDDEN \
          --seed $seed \
          --dataset $DS \
          --log_dir $LOG_PATH \
          --model_path $MODEL_PATH \
          --data_dir $DATA_PATH \
          --exp $FILE_NAME \
          --norm_type $NORM \
          --log_norm
  done
  
  ```

## Results

- **Training Performance**

![Fig-Training Performance](https://github.com/lsj2408/GraphNorm/blob/master/README.assets/Fig-Training%20Performance.png)

- **Test Performance**

**GCN with GraphNorm outperforms several sophisticated GNNs on OGBG-MOLHIV datasets.**

| Rank  | Method            | Test ROC-AUC          |
| ----- | ----------------- | --------------------- |
| **1** | **GCN+GraphNorm** | **$0.7883\pm0.0100$** |
| 2     | HIMP              | $0.7880\pm0.0082$     |
| 3     | DeeperGCN         | $0.7858\pm0.0117$     |
| 4     | WEGL              | $0.7757\pm0.0111$     |
| 5     | GIN+virtual node  | $0.7707\pm0.0149$     |
| 6     | GCN               | $0.7606\pm0.0097$     |
| 7     | GCN+virtual node  | $0.7599\pm0.0119$     |
| 8     | GIN               | $0.7558\pm0.0140$     |

![Fig-Test Performance](https://github.com/lsj2408/GraphNorm/blob/master/README.assets/Fig-Test%20Performance.png)

- **Ablation Study**

![Fig-Ablation Study](https://github.com/lsj2408/GraphNorm/blob/master/README.assets/Fig-Ablation%20Study.png)

- **Visualizations of Singular value distribution**

![Fig-Singular Value Distribution](https://github.com/lsj2408/GraphNorm/blob/master/README.assets/Fig-Singular%20Value%20Distribution.png)

- **Visualizations of Noisy Batch-level Statistics**

![Fig-Batch Level Statistics](https://github.com/lsj2408/GraphNorm/blob/master/README.assets/Fig-Batch%20Level%20Statistics.png)

## Citation

```latex
@misc{cai2020graphnorm,
    title={GraphNorm: A Principled Approach to Accelerating Graph Neural Network Training},
    author={Tianle Cai and Shengjie Luo and Keyulu Xu and Di He and Tie-yan Liu and Liwei Wang},
    year={2020},
    eprint={2009.03294},
    archivePrefix={arXiv},
    primaryClass={cs.LG}
}
```

