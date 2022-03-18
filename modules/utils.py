import torch
from torch import nn
import numpy as np
import random
from modules.networks.segnet import SegNet
from modules.networks.unet import UNet
from modules.losses import DiceLoss
import math
import time
import msd_pytorch as mp
from tqdm import tqdm
import re


def get_device(cuda_no=0):
    return torch.device(f'cuda:{cuda_no}' if torch.cuda.is_available() else 'cpu')


def train(model, train_loader, val_loader, epochs=50, lr=0.01,
          optimizer='adam', criterion='crossentropy', save_extension='',
          cuda_no=0, seed=42, save_path=''):
    
    # Set the seeds
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    device = get_device(cuda_no)
    print(device, flush=True)

    train_history_loss = []
    val_history_loss = []

    if isinstance(model, SegNet):
        model_name = 'segnet'
    elif isinstance(model, UNet):
        model_name = 'unet'
    else:
        model_name = 'msd'
        model.set_normalization(train_loader)

    print(f'Training {model_name}\n', flush=True)

    if optimizer == 'adam':
        optimizer = torch.optim.Adam(params=model.parameters(), lr=lr)
    if criterion == 'crossentropy':
        criterion = nn.CrossEntropyLoss()
    elif criterion == 'dice':
        criterion = DiceLoss()

    min_val_loss = math.inf
    
    for i in range(epochs):
        start = time.time()
        train_crt_loss = 0.0
        val_crt_loss = 0.0
        train_dice_loss = 0.0

        for step in ['train', 'val']:
            if step == 'train':
                loader = train_loader
            else:
                if val_loader is None:
                    break
                else:
                    loader = val_loader
                    
            for batch, labels in loader:
                batch = batch.to(device)
                labels = labels.to(device)
                
                # Extra step for MSD
                if model_name == 'msd':
                    model.set_input(batch)
                    model.set_target(labels)

                output = model(batch)
                loss = criterion(output, labels)

                if step == 'train':
                    # Should speed things up a bit
                    for param in model.parameters():    # Speedup
                        param.grad = None    # Speedup

                    loss.backward()
                    optimizer.step()

                    train_crt_loss += loss.item()
                    # train_dice_loss += DiceLoss()(output, labels)
                    train_dice_loss += 0
                else:
                    val_crt_loss += loss.item()
                
                # We get out if the loss is nan
                if math.isnan(loss.item()):
                    break
                
            if step == 'train':
                train_history_loss.append(train_crt_loss / len(train_loader))
            else:
                val_history_loss.append(val_crt_loss / len(val_loader))

        # Break if the loss reaches untrainable value (1.000 or nan)        
        if train_history_loss[i] == 1 or math.isnan(train_history_loss[i]):
            return None

        # We do not save the model after the first epoch
        if i > 0 and val_history_loss[i] < min_val_loss:
            min_val_loss = val_history_loss[i]
            torch.save(model.state_dict(), f'{save_path}{model_name}_{save_extension}_best.pt')

        print('Epoch {:} took {:.4f} seconds'.format(i, time.time() - start), flush=True)

        t_l = train_crt_loss / len(train_loader)
        v_l = val_crt_loss / len(val_loader)
            
        print('Train loss: {:.4f}'.format(t_l), flush=True)
        print('Valid loss: {:.4f}'.format(v_l), flush=True)

    history = (train_history_loss, val_history_loss)

    return model, history


def get_model(n_channels, n_classes, model_name='segnet', parallel=False):
    if model_name == 'segnet':
        model = SegNet(n_channels=n_channels, n_classes=n_classes)
    elif model_name == 'unet':
        model = UNet(n_channels=n_channels, n_classes=n_classes)
    elif model_name == 'msd':
        depth = 100
        width = 1
        dilations = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

        model = mp.MSDSegmentationModel(n_channels, n_classes, depth, width,
                                        dilations=dilations, parallel=parallel)

    return model


def net_prediction(model, img, cuda_no=0):
    with torch.no_grad():
        device = get_device(cuda_no)
            
        img = img.view(1, *img.shape)
        model = model.to(device)
        pred = model(img.to(device))

    return pred


def get_acc_metrics(predict, target):
    """
    We get here TP, FP, FN. We also assume binary classification.
    """
    if predict.shape[0] > 1:
        predict = torch.argmax(predict, dim=0)
    
    target = target.to(predict.get_device()).view(*predict.shape)

    tp = torch.sum(predict * target)
    fp = torch.sum(predict) - tp
    fn = torch.sum(target) - tp
                
    return tp, fp, fn


def get_performance(model, data, cuda_no=0):
    device = get_device(cuda_no)

    tp = 0
    fp = 0
    fn = 0
    
    for datum in tqdm(data):
        img, label = datum
        pred = net_prediction(model, img, cuda_no=cuda_no)

        tp_, fp_, fn_ = get_acc_metrics(pred.detach(), label.to(device))

        tp += tp_.item()
        fp += fp_.item()
        fn += fn_.item()

    return tp, fp, fn


def get_inclusion_images(lab_filenames):
    """
    From the real dataset extract the images that contain both E- and L-type cells

    Params:
        lab_filenames : list of str
            list with the paths pointing to the label files
    Returns:
        img_filenames : list of str
        lab_filenames : list of tuples (lab_filename_E, lab_filename_L)
    """

    root = '/'.join(lab_filenames[0].split('/')[:-1]) + '/'

    regex = re.compile(f'.+lab_(\d+)-(\D).tif')
    matches = list(map(regex.match, lab_filenames))

    useful_img_filenames = []
    useful_lab_filenames = []
    lab_dict = {}

    # Getting the paths of labels where both types of cells appear
    for cnt, m in enumerate(matches):
        if m:
            lab_no = m.group(1)

            if lab_no in lab_dict:
                lab_dict[lab_no] += 1
            else:
                lab_dict[lab_no] = 1

        if lab_dict[lab_no] == 2:
            useful_img_filenames.append(root + f'img_{lab_no}.tif')

            lab_filename_root = root + f'lab_{lab_no}'
            lab_pair = []
            for c in ['E', 'L']:
                lab_pair.append(lab_filename_root + f'-{c}.tif')
            useful_lab_filenames.append(lab_pair)

    return useful_img_filenames, useful_lab_filenames