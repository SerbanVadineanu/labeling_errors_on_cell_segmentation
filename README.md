# Labeling Errors on Cell Segmentation
This repository contains:
- A framework for applying perturbations (omission, inclusion, bias) to cell segmentation data sets as described in the paper An Analysis of the Impact of Annotation Errors on the
Accuracy of Deep Learning for Cell Segmentation;
- The experimental setup by which paper experiments were performed.

## Package Requirements
The requirements differ by the intended usage of the repository.

### Perturbation Framework
- python >= 3.7
- numpy >= 1.20
- opencv-python >= 4.5

### Experimental Setup
- python >= 3.7
- numpy >= 1.20
- opencv-python >= 4.5
- pytorch >= 1.9
- torchvision >= 0.2.2
- cudatoolkit >= 10
- tqdm >= 4.59
- tifffile >= 2021.8.30
- pandas >= 1.2.5
- msd_pytorch = 0.10.1 ([repository](https://github.com/ahendriksen/msd_pytorch))

## Data Requirements
The requirements differ by the intended usage of the repository.

### Perturbation Framework
- input image/volume and mask image/volume are numpy arrays;
- the cells belonging to the main class are individually labeled from 1 to *total_number_of_main_cells* (0 is reserved for background);
- the cells belonging to the additional class(es) are individually labeled from (*total_number_of_main_cells* + 1) to *total_number_of_cells*;
- for merging images/volumes to artificially perform inclusion, the images/volumes of both the main data set and the additional data set **must** have the same shape

### Experimental Setup
The requirements differ since, in this case, for the synthetic data we perform the perturbation online, i.e., when reading each slice. For the manually-annotated data perform the perturbations offline and save the perturbed labels in folders named accordingly.
All images must be saved as tiff.
#### Synthetic Data
- for each cell type we have two folders containing all slices and masks for all volumes
- one folder contains all available slices and must be named **hl60_tiff_all** and **granulocytes_tiff_all**, respectively (these will be used to ensure we have matching slices to perform inclusion);
- the other folder contains only the slices whose masks are not empty and must be named **hl60_all** and **granulocytes_all**, respectively (these will be used for all other purposes);
- the naming scheme for a slice is: image-final_{volume number in 4 digits}_{slice number}. For instance, for volume 0 and slice 10 we would have image-final_0000_10;
- the naming scheme for a mask slice is: image-label_{volume number in 4 digits}_{slice number}. For instance, for volume 0 and slice 10 we would have image-label_0000_10;


#### Manually-annotated Data
- all images and masks must have the same shape;
- the name of the folder with the images can be arbitrarily chosen;
- for each perturbation, the name of the folder with the perturbed labels can be arbitrarily chosen. However, the ending of the directory name must contain a numerical identifier separated by '_'. For instance, if the folder core name is **labels_perturbed** and it is the 3rd time we do the same perturbation, we denote it by naming the folder **labels_perturbed_2**;
- the naming scheme for images is: img_{numerical identifier}_{patch number}. For instance, if img_14 has been split in 10 patches of the same dimension, we refer to the 3rd patch of this image as img_14_2;
- the naming scheme for masks is: lab_{numerical identifier}_{cell type code}_{patch number}. The cell type code can be (the other cell types labels are ignored):
  - E - epithelial
  - L - lymphocytes

# Experimental Setup Manual
There are two python files that the user can run to repeat our experiments: `train_net.py` which is used to train a network on a specific type of perturbation, and `run_experiment.py` which is used to generate the test results of trained network for specific setups.
For both files there is a configuration file associated to them:
- `train_config.json` for `train_net.py` with the following fields:
  - **ds_name**: the name of the data set (e.g., granulocytes);
  - **path_to_data**: the path to the folder with images/volumes (hl60_all or granulocytes_all folders for synthetic data);
  - **path_to_labels**: the path to the folder with perturbed labels. It is to be used only for manually-annotated data where the perturbations are offline (must be set to null for the synthetic data);
  - **path_to_models**: the path to where the trained models should be saved;
  - **omission_rate**: the omission rate (as a float <= 1);
  - **inclusion_rate**: the inclusion rate (as a float <= 1);
  - **bias**: the bias (as an integer);
  - **lr**: the learning rate;
  - **epochs**: the number of epochs;
  - **batch_size**: the batch size;
  - **reps**: a list with the random seeds with which the different instances of the models are trained;
  - **model_name**: the name of the model. It can be either: **msd**, **unet** or **segnet**;
  - **criterion**: the loss function. It can be either **dice** or **crossentropy**;
  - **cuda_no**: the number of the cuda-enabled device on which the training will be performed.

- `experiment_config.json` for `run_experiment.py` with the following fields:
  - **path_to_data**: the path to the testing data (hl60_all or granulocytes_all folders for synthetic data; the testing volumes are already set in the script);
  - **path_to_models**: the path with the trained models;
  - **networks**: a list of the networks to be tested. The network can be either **msd**, **unet** or **segnet**;
  - **setup_list**: a list with the perturbation setups for which the experiment is run. Each element of the list is a tuple defined as **(omission_rate, inclusion_rate, bias)**;
  - **path_for_csv**: path for saving the results;
  - **datasets**: a list of the data sets for which the experiments will be performed. It can contain: hl60, granulocytes and/or epithelial.
  - **reps**: a list with the random seeds with which the different instances of the models were trained;
  - **criterion**: the loss function with which the models were trained. It can be either **dice** or **crossentropy**;
  - **cuda_no**: the number of the cuda-enabled device on which the models will be loaded;
  - **experiment_name**: the name for the csv with the results.


# Example of the Perturbation Framework
An example of the usage of the perturbation framework is provided in `perturbation_examples.ipynb`.
