# MAMRL: A Multimodal Adaptive Molecular Representation Learning Framework for Molecular Property Prediction

We present a multimodal adaptive molecular representation framework (**MAMRL**) including MPP dataset construction, model evaluation, a multimodal module, and a case study. We construct a **large-scale benchmark** named **MPPDB**, comprising **ADMET**, **physicochemical properties**, **activity cliffs**, and **chirality-sensitive** datasets. MAMRL systematically benchmarks baseline and select high-performing candidates as **feature generators**. MAMRL integrates Random **Forest-based** importance scores and **bidirectional attention** to adaptively preserves and analyzes task-relevant **multimodal features**, reducing feature redundancy and computational cost. Across ADMET/physicochemical and activity-cliff datasets, MAMRL improves average R² by **21.4%** and **29.4%** over baselines, respectively. On a dataset of **90,364** optical rotatory strengths computes via **TD-DFT**, MAMRL demonstrates robust **stereochemical discrimination**. SHAP analyses of thrombin ligands show that MAMRL consistently identifies **six-membered aromatic rings** as key activity motifs, demonstrating its strong interpretability and structural sensitivity, which can further guide the design of **bioactive compounds**. MAMRL also correctly predicted **rofecoxib cardiotoxicity** and **obeticholic-acid hepatotoxicity**, supporting its utility for activity and safety assessment in real-world drug development.

![MAMRL Framework](./images/MAMRL_framework.png)

## Setups

```bash
git clone https://github.com/linshaolong11/MAMRL.git
cd MAMRL
mamba env create -f environment.yaml
mamba activate MAMRL
```

## Datasets
Due to the large size of the original training dataset **MPPDB**, it is not suitable for direct upload to GitHub. Therefore, we have hosted it on [Zenodo](https://zenodo.org/records/17310984/files/MPPDB.zip).

Please download the dataset from the link above and extract it into the `./data` directory before running the training or evaluation scripts.

```bash
cd data 
wget https://zenodo.org/records/17310984/files/MPPDB.zip
unzip MPPDB.zip
```


## Train

We provide an example of training using the bbb_logbb dataset. You need to specify the input file path, task type (classification or regression), and the output file path for saving the trained model.

```bash
python train.py --input data/bbb_logbb.pkl --task_type classification --output result/out_bbb_logbb.pkl
```

This will train a MAMRL model on the given dataset and save the trained model to the specified location.

## Evaluation

To evaluate a trained MAMRL model, specify the path to the saved model file, the input dataset, and the task type (classification or regression):

```bash
python evaluate.py --model result/out_bbb_logbb.pkl --input data/bbb_logbb.pkl --task_type classification
```
This script will load the model and evaluate its performance on the test set contained in the specified dataset. 

## Others

For questions, issues, or dataset requests, please contact us directly at `shaolonglin2023@163.com`.
