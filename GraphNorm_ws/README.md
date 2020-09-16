# Codes for GraphNorm

## Prerequisites

- **ogbg_ws**: see `./ogbg_ws/requirements.txt`
- **gnn_ws**: see `./gnn_ws/requirements.txt`

## Results for Bioinformatics and Social Network datasets

### Training

To reproduce Figure 4 in our paper, run as follows:

```shell
cd ./gnn_ws/gnn_example/scripts/example-train-comparison
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
cd ./gnn_ws/gnn_example/scripts/example-test-comparison
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

## Results for Ogbg-molhiv

### Training

Training process is performed with **10 different random seeds** 

- GCN

  ```shell
  cd ./ogbg_ws/scripts/Example-gcn-test-performance
  ./seed-1-5-gcn_run.sh
  ./seed-6-10-gcn_run.sh
  ```

- GIN

  ```shell
  cd ./ogbg_ws/scripts/Example-gin-test-performance
  ./seed-1-5-gin_run.sh
  ./seed-6-10-gin_run.sh
  ```
  

Results can be found in :

- `./ogbg_ws/log/`:  recorded metric values along training process,
- `./ogbg_ws/model`: model checkpoints, which has the maximum validation metric values along training process.

**Notes**:

The number of epoch as 20~30 is fine for GIN with GraphNorm.

### Evaluation

For evaluation, we use the dumped model checkpoints and report the mean and standard deviation metric values.

- Use your **own trained models**

  Set the `MODEL_PATH` in `./ogbg_ws/scripts/evaluate_gin.sh` / `./ogbg_ws/scripts/evaluate_gcn.sh` to the desired item.

- Use provided **pre-trained models**

  we will release our pre-trained model soon.

## Notes

To further experiment on the provided codes, you can read the source codes in `./gnn_ws/gnn_examples/` or `./ogbg_ws/src/` and tune the hyper-parameters freely.