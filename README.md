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
| Average over test sets |  <span style="color:red"> 75.4 &pm; 14.8 </span> | <span style="color:green"> **76.1 &pm; 4.3** </span> |
</div>

## Description   
This repository contains the Pytorch code to reproduce the results presented in [our work](https://arxiv.org/abs/2203.09812).

## Install
The code is developed with TODO <b>Python XX - Pytorch XX - CUDA XX.</b>

Clone project and install dependencies:
```bash
# clone project   
git clone https://github.com/hsp-iit/prosthetic-grasping-experiments
# create virtual environment and install dependencies
cd prosthetic-grasping-experiments
python3 -m venv pge-venv
source pge-venv/bin/activate
pip install -r requirements.txt
 ```   

## Dataset preparation
Download the [realTODO]() and [syntheticTODO]() dataset
``` bash
wget xxx
unzip xxx
```   
The `datasets` folder contains all the datasets: both the real (i.e. `iHannesDataset`) and the synthetic (i.e. `ycb_synthetic_dataset`) dataset. For each dataset, both the frames and the pre-extracted features using _mobilenet_v2_ are available. <br>The `datasets` folder has the following macro-structure (i.e. path to the specific dataset folder):

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
If you want to use our Dataset classes, make sure that the above arrangement (both the macro-structure and the path to frames/features) is maintained.


Create softlinks:
```bash
ln -s datasets/real prosthetic-grasping-experiments/data
ln -s datasets/synthetic prosthetic-grasping-experiments/data
```


## Extract features [optional]
Pre-extracted features are already provided by downloading the datasets above. However, to extract features on your own, you can use:
```bash
TODO
```
For each video, a `features.npy` file is generated. The file has shape `(num_frames_in_video, feature_vector_dim)` and is located at the path defined above.

## Training
All runnable files are located under the `src/tools` folder. At the beginning of each file you can find some run command examples, with different arguments.


Train the classifier of `mobilenet_v2` CNN, starting from pre-extracted features:
```bash
TODO
```

## Test
TODO

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

## Manteiner
This repository is manteined by:
| | |
|:---:|:---:|
| [<img src="https://github.com/FedericoVasile1.png" width="40">](https://github.com/FedericoVasile1) | [@FedericoVasile1](https://github.com/FedericoVasile1) |

## Related links:
- For further details about our synthetic data generation pipeline, please refer to our [paper](https://arxiv.org/abs/2203.09812) (specifically SEC. IV) and feel free to contact me: federico.vasile@iit.it
- A demonstration video of our model trained on the synthetic data and tested on the Hannes prosthesis is available [hereTODO]()
- A presentation video summarizing our work is available [hereTODO]()
