# Geometric Sensitivity Decomposition 
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) 

![Diagram of Contribution](https://github.com/GT-RIPL/Geometric-Sensitivity-Decomposition/blob/main/images/summary_diagram.png)

1. This repo is the official implementation of [A Geometric Perspective towards Neural Calibration via Sensitivity Decomposition]() (*tian21gsd*). The pape is accpted at [NeurIPS 2021](https://nips.cc/Conferences/2021/ScheduleMultitrack?event=26546). as a spotlight paper.
2. We **reimplememented** [Exploring Covariate and Concept Shift for Out-of-Distribution Detection]() (*tian21explore*) and include it in the code base as well. The paper is accepted at [NeurIPS 2021 workshop on Distribution Shift](https://sites.google.com/view/distshift2021). 
3. For a brief introduction to these two papers, please visit the [project page](https://sites.google.com/view/geometric-decomposition/home).


## Create conda environment
```
conda env create -f requirements.yaml
conda activate gsd
```

## Training
1. Dataset will be automatically downloaded in the `./datasets` directory the first time. 
2. We provide support for *CIFAR10* and *CIFAR100*. Please change `name` in the configuration file accordingly (default: *CIFAR10*).

```yaml
data: 
    name: cifar10 
```
3. Three sample training configuration files are provided. 
    - To train a vanilla model.
        ```
        python train.py --config ./configs/train/resnet_vanilla.yaml   
        ```

    - To train the GSD model proposed in [tian21gsd]().
        ```
        python train.py --config ./configs/train/resnet_gsd.yaml   
        ```
    - To train the Geometric ODIN model proposed in [tian21exploring]().
        ```
        python train.py --config ./configs/train/resnet_geo_odin.yaml   
        ```

## Evaluation 
1, We provide support for evaluation on *CIFAR10*, *CIFAR100*, *CIFAR10C*, *CIFAR100C* and *SVHN*. We consider both **out-of-distribution (OOD) detection** and **confidence calibration**. Models trained on different datasets will use different evaluation datasets. 

|       |   OOD detection ||||  Calibration  || 
| :---------: | :------------: | :-----------: | :---------: |:---------: |:---------: |:---------: |
|Training | *Near OOD* ||*Far OOD*|*Special*|*ID*|*OOD*|
|CIFAR10|	CIFAR10C|	CIFAR100|	SVHN|	CIFAR100 Splits|	CIFAR10|	CIFAR10C|
|CIFAR100|	CIFAR100C|	CIFAR10|	SVHN|		               |CIFAR100|	CIFAR100C|

2. The `eval.py` file *optionally* calibrates a model. It **1)** evaluates calibration performance and **2)** saves several scores for OOD detection evaluation *later*.
    - Run the following commend to evaluate on a test set. 
        ```
        python eval.py --config ./configs/eval/resnet_{model}.yaml 
        ```

    - To specify a calibration method, select the `calibration` attribute out of supported ones (use `'none'` to avoid calibration). Note that a vanilla model can be calibrated using three supported methods, [temperature scaling](https://arxiv.org/abs/1706.04599), [matrix scaling](https://arxiv.org/abs/1706.04599) and [dirichlet scaling](https://arxiv.org/abs/1910.12656). GSD and Geometric ODIN use the alpha-beta scaling. 

        ```yaml
            testing: 
                calibration: temperature # ['temperature','dirichlet','matrix','alpha-beta','none'] 
        ```
    - To select a testing dataset, modify the `dataset` attribute. Note that the calibration dataset (specified under `data: name`) can be *different* than the testing dataset. 
        ```yaml
            testing: 
                dataset: cifar10 # cifar10, cifar100, cifar100c, cifar10c, svhn testing dataset
        ```

3. Calibration benchmark
    - Results will be saved under `./runs/test/{data_name}/{arch}/{calibration}/{test_dataset}_calibration.txt`.
    - We use Expected Calibration Error (ECE), Negative Log Likelihood and Brier score for calibration evaluation. 
    - We recommend using a 5-fold evalution for in-distribution (ID) calibration benchmark because `CIFAR10/100` does not have a val/test split. Note that  `evalx.py` does *not* save OOD scores. 
        ```
        python evalx.py --config ./configs/train/resnet_{model}.yaml 
        ```
    - (Optional) To use the proposed exponential mapping ([tian21gsd]()) for calibration, set the attribute `exponential_map` to 0.1.
    
4. Out-of-Distribution (OOD) benchmark
    - OOD evaluation needs to run  `eval.py` two times to extract OOD scores from both the ID and OOD datasets.
    - Results will be saved under `./runs/test/{data_name}/{arch}/{calibration}/{test_dataset}_scores.csv`. For example, to evaluate OOD detection performance of a vanilla model (ID:*CIFAR10* vs. OOD:*CIFAR10C*), you need to run `eval.py` twice on *CIFAR10* and *CIFAR10C* as the testing dataset. Upon completion, you will see two files, `cifar10_scores.csv` and  `cifar10c_scores.csv` in the same folder.
    - After the evaluation results are saved, to calculate OOD detection performance, run  `calculate_ood.py` and specify the conditions of the model: training set, testing set, model name and calibration method. The flags will help the function locate csv files saved in the previous step.
        ```
        python utils/calculate_ood.py --train cifar10 --test cifar10c --model resnet_vanilla --calibration none
        ``` 
    - We use AUROC and TNR@TPR95 as evaluation metrics.



## Performance
1. confidence calibration Performance of models trained on *CIFAR10*

|| accuracy|| ECE || Nll||
| :---------: | :------------: | :-----------: | :---------: |:---------: |:---------: |:---------: |
||CIFAR10|CIFAR10C|CIFAR10|CIFAR10C|CIFAR10|CIFAR10C|
|Vanilla| 96.25| 69.43| 0.0151|	0.1433|	0.1529 | 1.0885|
|[Temperature Scaling](https://arxiv.org/abs/1706.04599)|96.02| 71.54 |	*0.0028*|	0.0995|	0.1352|	0.8699|
|[Dirichlet Scaling](https://arxiv.org/abs/1910.12656) |95.93|	71.15|	0.0049|	0.1135|	**0.1305**|	0.9527|
|GSD (tian21gsd)|96.23|	71.7|	0.0057|	**0.0439**|	0.1431|	**0.7921**|
|Geometric ODIN (tian21explore)| 95.92|70.18|**0.0016**|*0.0454*|**0.1309**|*0.8138*|

2. Out-of-Distribution Detection Performance (AUROC) of models trained on *CIFAR10*

|AUROC|score function| CIFAR100| CIFAR10C | SVHN|
| :---------: | :------------: | :-----------: | :---------: |:---------: |
|Vanilla|[MSP](https://arxiv.org/abs/1610.02136)| 88.33	| 71.49 | 91.88|
||[Energy](https://arxiv.org/abs/2010.03759)| 88.11 | 71.94| 92.88|
|GSD (tian21gsd)|U| **92.68** |	77.68 |	**99.29**|
|Geometric ODIN (tian21explore)|U|**92.53**	| **78.77**	|**99.60**|




## Additional Resources
1. Pretrained models
    - [vanilla wide ResNet](https://drive.google.com/file/d/1sY5BoB3czDqWWwVlJV-fR5FG2p8qiwxK/view?usp=sharing)
    - [GSD wide ResNet](https://drive.google.com/file/d/1nS84Db7BMouUqTzR4M2U3WcwWu0v5w2T/view?usp=sharing)
    - [Geometric ODIN wide ResNet](https://drive.google.com/file/d/1ftOYTMuuaPnTjHfSEniOU612VUyD926m/view?usp=sharing)
