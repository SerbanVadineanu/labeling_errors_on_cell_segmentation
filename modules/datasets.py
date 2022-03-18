import torch
import numpy as np
import tifffile as tiff
import cv2
from glob import glob
import re


class SyntheticDataset(torch.utils.data.Dataset):
    def __init__(self, data_path, train=None, train_test_split=0.7, seed=42, file_range=(0, 25),
                 binary_mask=True, drop_p=0,
                 incl_p=0, max_iter=0, test=False, scaling=True):
        assert file_range[0] >= 0 and file_range[1] <= 30
        assert 0 <= drop_p < 1

        super().__init__()
        self.train = train
        self.test = test    # If true, then the labels are kept perfect
        self.binary_mask = binary_mask
        self.drop_p = drop_p
        self.incl_p = incl_p
        self.morph_op = max_iter > 0
        self.scaling = scaling

        tiff_list = ['00' + f"{x:02d}" for x in range(*file_range)]
         
        def get_filenames(root):
            img_list = []
            lab_list = []
            for tiff_num in tiff_list:
                crt_img_files = glob(root + f'image-final_{tiff_num}_*.tiff')
                crt_label_files = glob(root + f'image-labels_{tiff_num}_*.tiff')

                crt_img_files = sorted(crt_img_files, key=lambda x: int(x.split('_')[-2].split('.')[0]))
                crt_label_files = sorted(crt_label_files, key=lambda x: int(x.split('_')[-2].split('.')[0]))

                img_list.extend(crt_img_files)
                lab_list.extend(crt_label_files)
            
            return sorted(img_list), sorted(lab_list)

        self.img_filenames, self.lab_filenames = get_filenames(data_path)

        # Check if we drop cell labels
        if 'hl60' in str.lower(data_path):
            n_labels = 20
        elif 'granulocytes' in str.lower(data_path):
            n_labels = 15
        
        if drop_p != 0:
            np.random.seed(seed)
            # Requires knowledge about the number of cells from the mask
            self.dropped_labels = np.random.choice(range(1, n_labels+1), int(drop_p * n_labels), replace=False)
        
        # Check if we do morphological operations to labels
        if max_iter > 0:
            np.random.seed(seed)
            # Choose operations and iterations for all 30 volumes
            self.operation_per_volume = np.random.choice(['d', 'e'], 30)
            self.iterations_per_volume = np.random.choice(range(1, max_iter+1), 30)
        
        # Check if we use inclusion
        if incl_p != 0:
            np.random.seed(seed)
            
            if 'hl60' in str.lower(data_path):
                extra_ds_folder = 'granulocytes_tiff_all'
                n_labels_extra_ds = 15
            elif 'granulocytes' in str.lower(data_path):
                extra_ds_folder = 'hl60_tiff_all'
                n_labels_extra_ds = 20
            
            self.included_labels = np.random.choice(range(1, n_labels_extra_ds + 1), int(incl_p * n_labels_extra_ds),
                                                    replace=False)
            self.img_filenames_extra_ds, self.lab_filenames_extra_ds = [], []
        
            # Make sure that the HL60 images have corresponding granulocytes images or vice versa
            for i_f, l_f in zip(self.img_filenames, self.lab_filenames):
                split_i_f, split_l_f = i_f.split('/'), l_f.split('/')
                split_i_f[-2], split_l_f[-2] = extra_ds_folder, extra_ds_folder
                self.img_filenames_extra_ds.append('/'.join(split_i_f))
                self.lab_filenames_extra_ds.append('/'.join(split_l_f))

        dataset_size = len(self.img_filenames)

        if train is not None:
            train_size = int(train_test_split * dataset_size)

            np.random.seed(seed)
            self.train_indices = np.random.choice(dataset_size, size=train_size,
                                                  replace=False)
            self.test_indices = np.setdiff1d(np.arange(dataset_size), self.train_indices)

    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()

        if self.train is None:
            pass
        elif self.train:
            index = self.train_indices[index]
        else:
            index = self.test_indices[index]

        img_filename = self.img_filenames[index]
        lab_filename = self.lab_filenames[index]

        img = tiff.imread(img_filename)
        if self.scaling:
            img = img / np.max(img)    # Scale the image
        label = tiff.imread(lab_filename).astype(np.uint8)
        
        volume_no = int(img_filename.split('/')[-1].split('_')[1])

        # Optional drop of a few labels
        if self.drop_p != 0:
            if not self.test:
                for l in self.dropped_labels:
                    label[np.where(label == l)] = 0

        # Inclusion part
        if self.incl_p != 0:
            # First, read the image from extra ds
            img_filename_extra_ds = self.img_filenames_extra_ds[index]
            lab_filename_extra_ds = self.lab_filenames_extra_ds[index]

            img_extra_ds = tiff.imread(img_filename_extra_ds)
            label_extra_ds = tiff.imread(lab_filename_extra_ds).astype(np.uint8)
            
            # Let's scale the additional images
            img_extra_ds = img_extra_ds / np.max(img_extra_ds)
            
            # Put our image on top (hardcoded intensity threshold)
            matching_indices = np.where(((label_extra_ds > 0) | (img_extra_ds > 0.8)) & (label == 0))
            img[matching_indices] = img_extra_ds[matching_indices]
            # Now, for the label
            if not self.test:
                for l in self.included_labels:
                    label[np.where(label_extra_ds == l)] = 1

        # Ensure the mask is binary
        if self.binary_mask:
            label[np.where(label != 0)] = 1

        # Bias part
        if self.morph_op:
            if not self.test:
                kernel = np.ones((3, 3), np.uint8)    # Hardcoded kernel
                operation = cv2.erode if self.operation_per_volume[volume_no] == 'e' else cv2.dilate
                label = operation(label, kernel, iterations=self.iterations_per_volume[volume_no])

        return (torch.from_numpy(img.reshape(1, *img.shape).astype(float)).float(),
                torch.from_numpy(label).long())

    def __len__(self):
        if self.train is None:
            return len(self.img_filenames)
        elif self.train:
            return len(self.train_indices)
        else:
            return len(self.test_indices)
      

class RealDataset(torch.utils.data.Dataset):
    def __init__(self, data_path, labels_path=None, ds='L', train_test_split=None,
                 train_data=True, transforms=None, binary_mask=True, seed=42):
        """
        Initializes the dataset object

        Params:
            root : str
                the path to the directory containing the data
            root_labels : str
                path to the folder with the labels for our images. We use this when we have multiple folders generated for
                different setups that concern only the labels
            ds : str
                the type of data that we want
                can be L, E or LE/EL (when using inclusion)
        """
        if data_path[-1] != '/':
            data_path += '/'
        
        if labels_path is None:
            labels_path = data_path
        elif labels_path[-1] != '/':
            labels_path += '/'
            
        self.img_filenames = []
        self.lab_filenames = glob(labels_path + 'lab_*.tif')
        self.train_test_split = train_test_split
        self.train_data = train_data
        self.binary_mask = binary_mask
        self.train_data = train_data
        self.transforms = transforms
        
        # Filter the labels to only include the intended ds
        regex = re.compile(f'.+lab_(\d+)-[{ds}]_(\d+).*.tif')
        matches = list(map(regex.match, self.lab_filenames))
        self.lab_filenames = []
        for m in matches:
            if m:
                self.lab_filenames.append(m.group(0))
                self.img_filenames.append(data_path + f'img_{m.group(1)}_{m.group(2)}.tif')
        
        if train_test_split:
            dataset_size = len(self.img_filenames)
            train_size = int(train_test_split * dataset_size)

            np.random.seed(seed)
            self.train_indices = np.random.choice(dataset_size, size=train_size,
                                                  replace=False)
            self.test_indices = np.setdiff1d(np.arange(dataset_size), self.train_indices)

    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()

        if self.train_test_split:
            if self.train_data:
                index = self.train_indices[index]
            else:
                index = self.test_indices[index]

        img_filename = self.img_filenames[index]
        lab_filename = self.lab_filenames[index]

        img = tiff.imread(img_filename)[:, :, :3].astype(np.uint8)
        label = tiff.imread(lab_filename).astype(np.uint8)
        # Ensure the mask is binary
        if self.binary_mask:
            label[np.where(label != 0)] = 1

        if self.transforms:
            img, label = self.transforms(img, label)

        else:
            img = img / 255    # Scale the image

        if self.transforms is None:
            return (torch.from_numpy(np.transpose(img.astype(float), (2, 0, 1))).float(),
                    torch.from_numpy(label).long())
        else:
            return img, torch.from_numpy(label).long()

    def __len__(self):
        if self.train_test_split:
            if self.train_data:
                return len(self.train_indices)
            else:
                return len(self.test_indices)
        else:
            return len(self.img_filenames)
