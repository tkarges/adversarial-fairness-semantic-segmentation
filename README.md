# Reinforcement Learning for Adversarial Fairness in Semantic Segmentation
This is the code repository for the master's thesis "Reinforcement Learning for Adversarial Fairness in Semantic Segmentation". It contains all necessary code to reproduce model checkpoints, evaluation results, and plots from the thesis.

## Abstract
This thesis investigates the adversarial fairness problem in semantic segmentation, whichis concerned with class-wise disparities in model performance caused by adversarial examples. Existing class-wise differences are amplified by adversarial attacks, which leads
to problems in safety-critical applications of semantic segmentation like autonomous driving or medical imaging, where a model should be able to reliably predict each object class. While this issue has been subject to research in the context of image classification tasks, there is little direct literature on identifying and mitigating this problem in
segmentation tasks. In this thesis, common semantic segmentation datasets are analyzed with respect to adversarial fairness. The results indicate that robust disparities are present and not only caused directly by dataset imbalance, but rather by additional prediction biases and model behavior, which are amplified by adversarial attacks. The transfer of an existing adversarial fairness framework for image classification, Distance-Aware Fair Adversarial Training (DAFA), does not show the desired improvement, with
the method performing worse than an adversarially trained baseline in terms of both overall robustness and class-wise performance. For this reason, a policy gradient-based reinforcement learning framework for adaptive weight computation for class-wise losses is proposed. Multiple variants of this framework are presented, all of which show improve-
ment of class-wise IoU scores of the most vulnerable classes. The method shows that adaptive loss weighting strategies can mitigate class-wise robust disparities in segmentation. However, increased scores for vulnerable classes are achieved with substantial decreases of aggregate robustness and clean performance. Challenges like aggregate model degradation, enhanced experimental scope, and comprehensive ablation studies represent opportunities for future research.

## Important Setup Notes

### Models

The model classes used here are adapted from the source paper on adversarial robustness in segmentation. These models use a specialized pretrained ResNet backbone that cannot be downloaded from the official PyTorch source. In the release provided alongside this repository, there is a file named **resnet50_v2.pth**. This checkpoint file has to be copied to **checkpoints/resnet/**. Without this step it is not possible to initialize models for training! Using pretrained models should still be possible without this, as the ResNet parameters are saved in the final checkpoint files.

### Data

This repository contains training code for Cityscapes and PASCAL VOC 2012. To actually run experiments, the corresponding datasets need to be located on your disk somewhere. Some additional steps are necessary to use these local datasets with the code provided here.<br>All data loading is performed by using predefined root directories for both datasets. These are located in **datasets/metadata.py**. In order for the dataset loading code to find the data files, the variables **CITYSCAPES_ROOT** and **VOC_2012_ROOT** need to be changed accordingly.<br>
The recommended way is to use a symlink to your dataset directories to keep the original repo structure unchanged. This is done as follows:
```
ln -s /path/to/your/cityscapes/dataset /datasets/cityscapes
ln -s /path/to/your/voc2012/dataset /datasets/voc2012
```

The directories for the corresponding dataset in **/datasets/** should not exist in advance, as that might lead to problems. A valid cityscapes instance should contain subdirectories **gtFine** and **leftImg8bit**. A valid VOC 2012 instance should contain a folder named **VOCdevkit**.

### GPU Training
All training scripts support training and validation on multiple GPUs, a single GPU, and CPU. Evaluating checkpoints with **evaluate_checkpoints.py** only uses a single GPU.<br>
In case the training script starts but appears to be stuck before the first epoch without any command line outputs while using distributed training, putting the following arguments in front of the corresponding *torchrun* command can potentially fix this problem:
```
NCCL_IB_DISABLE=1 NCCL_P2P_DISABLE=1 torchrun --standalone etc.
```
## Running Experiments
All training invocations are centralized in the file **main.py**. There, all arguments are specified. These will be described here. Argument type *store_true* means that the argument is set true if it is included and false otherwise. No explicit value is needed.
### General Arguments
- **--training-type** (str) in ["rl", "normal", "dafa"]: controls which training script is called. "normal" refers to methods discussed in the analysis, whereas "rl" and "dafa" refer to training the policy gradient controller and DAFA, respectively
- **--model** (str) in ["deeplabv3", "pspnet"]: model that is used for training
- **--dataset** (str) in ["cityscapes", "voc2012"]: dataset used for training. **Note:** training types "rl" and "dafa" have not been reliably tested with "voc2012"
- **--model-savedir** (str): where the resulting checkpoints and logs are saved. This should be a valid path.
- **--load-checkpoint** (store_true): indicates whether a pretrained checkpoint should be loaded for continued training
- **--checkpoint-path** (str): path to the model checkpoint used for further training
- **--checkpoint-finetune** (store_true): specific argument used by the RL training strategy for the finetuning-based approach
- **--epochs** (int): number of training epochs
- **--learning-rate** (float): learning rate (the default of 0.01 should be kept for reproducibility; if a checkpoint is loaded, the learning rate is automatically adjusted according to the polynomially decaying schedule)
- **--batch-size** (int): batch size (default is 16)
- **--eval-batch-size** (int): batch size for model evaluation. **Important: VOC 2012 can only be evaluated with eval-batch-size of 1 due to size differences between images**
- **--evaluate-robust** (store_true): determines whether robust evaluation is performed during training. Clean evaluation is always performed.
- **--eval-interval** (int): determines how often the model runs evaluation during training (set this to a very large number to effectively disable evaluation)

### Adversarial Training
These arguments are used by all approaches for the specific adversarial training setup.
- **--attacker** (str) in ["segpgd", "trades-pgd"]: attacker that is used for adversarial training. "trades-pgd" is only availabe for DAFA
- **--attack-iterations** (int): number of PGD iterations
- **--attack-epsilon** (float): attack margins
- **--attack-alpha** (float): attack gradient scale

### Experiments from the Analysis (SegPGD-AT and SegPGD-AT-100)
These parameters are used by **training_type=normal**
- **--adversarial-training** (store_true): activates adversarial training; if this is not included, standard training is performed
- **--adversarial-loss-weight** (float): controls how much the adversarial loss term influences the objective; should be left at 1.0 for reproducibility
- **--adversarial-training-model** (str) in ["at_50_50", "at_100"]: at_50_50 replicates SegPGD-AT, at_100 corresponds to SegPGD-AT-100

### DAFA
The arguments control DAFA training for **training_type=dafa**
- **--dafa-mode** (str) in ["segpgd", "trades-pgd"]: segpgd corresponds to DAFA-CE, trades-pgd trains DAFA-TRADES
- **--dafa-warmup-epochs** (int): number of DAFA warmup epochs; a checkpoint is saved after warmup
- **--dafa-lambda** (float): DAFA weight update scaling factor
- **--dafa-beta** (float): DAFA-TRADES KL-divergence regularization strength
- **--dafa-scale-margins** (store_true): determines whether attack margins are scaled (available for both attack implementations)

### Reinforcement Learning Controller
These arguments control RL training for **training_type=rl**
- **--rl-action-epochs** (int): how frequently the policy steps (new action + update)
- **--rl-hidden-dim** (int): hidden layer dimension of the policy network (should be kept at 32)
- **--rl-learning-rate** (float): policy learning rate (should be kept at 1e-4)
- **--rl-entropy-coefficient** (float): strength of entropy regularization in the REINFORCE loss (should be kept at 1e-3)
- **--rl-reward-scale** (float): scales the reward (default is 10)
- **--rl-uniform-mix** (float): controls how much influnce uniform weights get on the final class weights
- **--rl-min-weight** (float): lower bound for clamping class weights
- **--rl-max-weight** (float): upper bound for clamping class weights
- **--rl-checkpoint-path** (str): path to a pretrained policy, if one should be used
- **--rl-apply-clean-weight** (store_true): applies weights to the clean loss term if included
- **--rl-apply-adv-weight** (store_true): applies weights to the adversarial loss term if included
- **--rl-clean-weight-mode** (str) in ["pixel", "macro"]: determines the class-wise aggregation used by the clean loss function
- **--rl-adv-weight-mode** (str) in ["pixel", "macro"]: determines the class-wise aggregation used by the adversarial loss function

## Evaluating Checkpoints
Checkpoints can be evaluated using the script **evaluate_checkpoints.py**. Results are saved in .pt format and can be inspected afterwards, e.g., using a short notebook and looking at the dictionary keys that are of interest.

- **--model** (str) in ["deeplabv3", "pspnet"]: model class of the checkpoint that is evaluated; specifying the wrong model here leads to obvious problems
- **--dataset** (str) in ["cityscapes", "voc2012"]: dataset on which the checkpoint should be evaluated
- **--checkpoint-path** (str): path to the model checkpoint
- **--result-path** (str): where results are saved
- **--result-name** (str): name of the evaluation result file; prefixes for clean and robust results are added automatically
- **--eval-batch-size** (int): batch size for evaluation (VOC 2012 only works with 1)
- **--evaluate-clean** (store_true): evaluates the checkpoint on clean data
- **--evaluate-robust** (store_true): evaluates the checkpoint on perturbed data