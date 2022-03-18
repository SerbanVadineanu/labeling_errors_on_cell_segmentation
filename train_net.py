import json
from modules.datasets import SyntheticDataset, RealDataset
from torch.utils.data import DataLoader
from modules.utils import get_model, train, get_device
import torch
import numpy as np
import torchvision.transforms.functional as tf
import random


def transform(image, mask):
    # To PIL image
    image = tf.to_pil_image(image)
    mask = tf.to_pil_image(mask)

    # Random horizontal flipping
    if random.random() > 0.5:
        image = tf.hflip(image)
        mask = tf.hflip(mask)

    # Random vertical flipping
    if random.random() > 0.5:
        image = tf.vflip(image)
        mask = tf.vflip(mask)

    # Transform to tensor
    image = np.array(image) / 255

    return torch.from_numpy(np.transpose(image.astype(float), (2, 0, 1))).float(), np.array(mask)


with open('train_config.json') as json_data:
    cfg = json.load(json_data)

test = False

extension = cfg['criterion'] + '_'

extension += f"omission_{cfg['omission_rate']}_"
extension += f"inclusion_{cfg['inclusion_rate']}_"
extension += f"bias_{cfg['bias']}_"

extension += f"ds_{cfg['ds_name']}_"

for i in cfg['reps']:
    crt_extension = extension + f'{i}'

    if str.lower(cfg['ds_name']) in ['hl60', 'granulocytes']:
        file_range = (0, 25)    # The range of volumes for training. Default from 0 to 25 (exclusive) out of 30
        train_data = SyntheticDataset(data_path=cfg['path_to_data'], train=True,
                                      drop_p=cfg['omission_rate'], incl_p=cfg['inclusion_rate'], max_iter=cfg['bias'],
                                      file_range=file_range, seed=i, test=test, train_test_split=0.7)

        val_data = SyntheticDataset(data_path=cfg['path_to_data'], train=False,
                                    drop_p=cfg['omission_rate'], incl_p=cfg['inclusion_rate'], max_iter=cfg['bias'],
                                    file_range=file_range, seed=i, test=test, train_test_split=0.7)
        n_channels = 1
        n_classes = 2
    else:
        crt_root_labels = cfg['path_to_labels'] + f'_{i}'
        train_data = RealDataset(data_path=cfg['path_to_data'], labels_path=crt_root_labels, ds='E',
                                 train_test_split=0.7, seed=i, train_data=True, transforms=transform)
        val_data = RealDataset(data_path=cfg['path_to_data'], labels_path=crt_root_labels,
                               ds='E', train_test_split=0.7, seed=i, train_data=False, transforms=None)
        n_channels = 3
        n_classes = 2

    train_loader = DataLoader(dataset=train_data, batch_size=cfg['batch_size'],
                              num_workers=2, pin_memory=True, drop_last=True)
    val_loader = DataLoader(dataset=val_data, batch_size=cfg['batch_size'], num_workers=2,
                            pin_memory=True, drop_last=True)

    model = get_model(n_channels=n_channels, n_classes=n_classes, model_name=cfg['model_name'])
    model = model.to(get_device(cfg['cuda_no']))

    _, _ = train(model=model, train_loader=train_loader, val_loader=val_loader,
                 criterion=cfg['criterion'], lr=cfg['lr'], epochs=cfg['epochs'],
                 cuda_no=cfg['cuda_no'], seed=i,
                 save_path=cfg['path_to_models'], save_extension=crt_extension)

    print(f'Done with iteration {i}', flush=True)
