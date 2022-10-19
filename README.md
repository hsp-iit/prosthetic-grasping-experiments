<div align="center">    
 
# Grasp Pre-shape Selection by Synthetic Training:<br> Eye-in-hand Shared Control on the Hannes Prosthesis
</div>

We propose a pipeline to select the pre-shape of the Hannes prosthesis using visual input. We collected a real dataset and used it to train and test our models. The test sets are organized into 5 different sets of increasing complexity. Each test set represents a different condition that doesn't appear in the real training set. <br>Our main contribution is a [synthetic data generation pipeline](https://github.com/hsp-iit/prosthetic-grasping-simulation) designed for vision-based prosthetic grasping. We compare a model trained on real data with the same model trained on the proposed synthetic data. As shown in the table below, the synthetically-trained model achieves comparable average value and better standard deviation, proving our method robustness.<br>
_Our work is accepted to IROS 2022_

<div align="center">    

![test_sets_reduced](https://user-images.githubusercontent.com/50639319/192563622-ab8fd7f5-715a-4c12-bae6-cd1bf09117da.gif)


| Test set |      Real training<br>Video acc. (%)      |  Synthetic training<br>Video acc. (%) |
|:----------:|:-------------:|:------:|
| Same person |  **98.9 &pm; 0.8** | 80.2 &pm; 0.9 |
||||
| Different velocity |  81.7 &pm; 0.9 | 79.7 &pm; 0.8 |
| From ground |  76.2 &pm; 1.0 | 76.0 &pm; 0.9 |
| Seated |  63.9 &pm; 1.0 | **68.1 &pm; 1.0** |
||||
| Different background |  56.2 &pm; 1.7 | **76.4 &pm; 2.0** |
||||
| Average over test sets | 75.4 &pm; 14.8 | **76.1 &pm; 4.3** |
</div>

## Description   
This repository contains the PyTorch code to reproduce the results presented in [our work](https://arxiv.org/abs/2203.09812).

## Install
The code is developed with <b>Python 3.8 - PyTorch 1.10.1 - CUDA 10.2.</b>

Clone project and install dependencies:
```bash
# clone project   
git clone https://github.com/hsp-iit/prosthetic-grasping-experiments
# create virtual environment and install dependencies
cd prosthetic-grasping-experiments
python3 -m venv pge-venv
source pge-venv/bin/activate
pip install -r requirements.txt

# in the install above, torch installation may fail. Try with the command below:
pip install torch==1.10.1+cu102 torchvision==0.11.2+cu102 torchaudio==0.10.1 -f https://download.pytorch.org/whl/cu102/torch_stable.html
# ..or go through the website: https://pytorch.org/get-started/previous-versions/
 ```   


## Dataset preparation
Download the [realTODO]() and [syntheticTODO]() dataset
``` bash
wget xxx
unzip xxx
```   
The `datasets` folder contains all the datasets: both the real (i.e. `iHannesDataset`) and the synthetic (i.e. `ycb_synthetic_dataset`) dataset. For each dataset, both the frames and the pre-extracted features using _mobilenet_v2_ (pre-trained on ImageNet) are available. <br>The `datasets` folder has the following macro-structure (i.e., path to the specific dataset folder):

```
datasets/
├── real/
|   ├── frames/
|   |   ├── iHannesDataset
|   |   
│   ├── features/
|       ├── mobilenet_v2/
|           ├── iHannesDataset
|
├── synthetic/
    ├── frames/
    |   ├── ycb_synthetic_dataset
    |   
    ├── features/
        ├── mobilenet_v2/
            ├── ycb_synthetic_dataset
```
Each dataset (i.e., iHannesDataset, ycb_synthetic_dataset) has the following path to the frames/features:
``` bash
DATASET_BASE_FOLDER/CATEGORY_NAME/OBJECT_NAME/PRESHAPE_NAME/Wrist_d435/rgb*
```
If you want to use our _dataloaders_, make sure that the above arrangement (both the macro-structure and the path to frames/features) is maintained.


Create softlinks:
```bash
cd prosthetic-grasping-experiments/data
ln -s /YOUR_PATH_TO_DOWNLOADED_DATASET/datasets/real 
ln -s /YOUR_PATH_TO_DOWNLOADED_DATASET/datasets/synthetic 
```

and the resulting structure is:

```
prosthetic-grasping-experiments/
├── data/
    ├── real/
    |   ├── ...
    |   
    ├── synthetic/
        ├── ...
```
## Extract features [optional]
Pre-extracted features are already provided by downloading the datasets above. However, to extract features on your own, you can use:
```bash
cd prosthetic-grasping-experiments
python3 src/tools/cnn/extract_features.py \
--batch_size 1 --source Wrist_d435 \
--input rgb --model cnn --dataset_type SingleSourceImage \
--feature_extractor mobilenet_v2 --pretrain imagenet \
--dataset_name iHannesDataset 
```
For each video, a `features.npy` file is generated. The file has shape `(num_frames_in_video, feature_vector_dim)` and will be located according to the path defined above.

## Training
All runnable files are located under the `src/tools` folder. At the beginning of each file you can find some run command examples, with different arguments.

When the training starts, a folder is created at the `prosthetic-grasping-experiments/runs` path (you can specify the folder name with `--log_dir` argument). This folder is used to store the measures and the best model checkpoint. 


**Example 1**: train the fully-connected classifier of _mobilenet_v2_ on the real dataset, starting from pre-extracted features:
```bash
cd prosthetic-grasping-experiments
python3 src/tools/cnn/train.py --epochs 5 \
--batch_size 32 --source Wrist_d435 --dataset_type SingleSourceImage \
--split random --input rgb --output preshape --model cnn \
--feature_extractor mobilenet_v2 --pretrain imagenet --freeze_all_conv_layers \
--from_features --dataset_name iHannesDataset \
--log_dir train_from_features
```

**Example 2**: same as above, but training on synthetic data (remember to add the `--synthetic` argument, otherwise a wrong path to the dataset is constructed):
```bash
cd prosthetic-grasping-experiments
python3 src/tools/cnn/train.py --epochs 5 \
--batch_size 64 --source Wrist_d435 --dataset_type SingleSourceImage \
--split random --input rgb --output preshape --model cnn \
--feature_extractor mobilenet_v2 --pretrain imagenet --freeze_all_conv_layers \
--from_features --dataset_name ycb_synthetic_dataset --synthetic
```

**Example 3**: train the LSTM on the real dataset, starting from pre-extracted features:
```bash
cd prosthetic-grasping-experiments
python3 src/tools/cnn_rnn/train.py --epochs 10 \
--batch_size 32 --source Wrist_d435 --dataset_type SingleSourceVideo \
--split random --input rgb --output preshape --model cnn_rnn --rnn_type lstm \
--rnn_hidden_size 256 --feature_extractor mobilenet_v2 --pretrain imagenet \
--freeze_all_conv_layers --from_features --dataset_name iHannesDataset
```


**Example 4**: fine-tune the whole network (i.e., use RGB frames instead of pre-extracted features) starting from the ImageNet weights:
```bash
cd prosthetic-grasping-experiments
python3 src/tools/cnn/train.py --epochs 10 \
--batch_size 64 --source Wrist_d435 --dataset_type SingleSourceImage \
--split random --input rgb --output preshape --model cnn \
--feature_extractor mobilenet_v2 --pretrain imagenet \
--lr 0.0001 --dataset_name ycb_synthetic_dataset --synthetic
```



## Test
To test a model, copy and paste its running command used for training and substitute the `train.py` script with `eval.py`. Moreover, you have to specify the path to the model checkpoint with `--checkpoint` argument and the test set with `--test_type` argument.

**Example 1**: test the model on the _Same person_ test set:
```bash
cd prosthetic-grasping-experiments
python3 src/tools/cnn/eval.py --epochs 5 \
--batch_size 32 --source Wrist_d435 --dataset_type SingleSourceImage \
--split random --input rgb --output preshape --model cnn \
--feature_extractor mobilenet_v2 --pretrain imagenet --freeze_all_conv_layers \
--from_features --dataset_name iHannesDataset \
--log_dir train_from_features \
--checkpoint runs/train_from_features/best_model.pth --test_type test_same_person
```
Some confusion matrices will be displayed on screen, you can simply close them and visualize later on _tensorboard_. Many different metrics, both at per-frame and video granularity, are printed on the shell. In our work, the results are presented as video accuracy (obtained from per-frame predictions through majority voting, excluding the background class). This value is printed on the shell as follows:
```
.
.
.

=== VIDEO METRICS ===

ACCURACY W BACKGR: xx.xx%

ACCURACY W/O BACKGR: xx.xx%       <==

.
.
.
```

You can visualize both the training and evaluation metrics on _tensorboard_ with:
```bash
cd prosthetic-grasping-experiments
tensorboard --logdir runs/train_from_features
```

## Citation   
```
@inproceedings{vasile2022,
    author    = {F. Vasile and E. Maiettini and G. Pasquale and A. Florio and N. Boccardo and L. Natale},
    booktitle = {2022 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)},
    title     = {Grasp Pre-shape Selection by Synthetic Training: Eye-in-hand Shared Control on the Hannes Prosthesis},
    year      = {2022},
    month     = {Oct},
}
```

## Mantainer
This repository is mantained by:
| | |
|:---:|:---:|
| [<img src="https://github.com/FedericoVasile1.png" width="40">](https://github.com/FedericoVasile1) | [@FedericoVasile1](https://github.com/FedericoVasile1) |

## Related links:
- For further details about our synthetic data generation pipeline, please refer to our [paper](https://arxiv.org/abs/2203.09812) (specifically SEC. IV) and feel free to contact me: federico.vasile@iit.it
- A demonstration video of our model trained on the synthetic data and tested on the Hannes prosthesis is available [hereTODO]()
- A presentation video summarizing our work is available [hereTODO]()
