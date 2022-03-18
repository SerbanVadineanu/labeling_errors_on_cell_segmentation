import json
import pandas as pd
from modules.datasets import SyntheticDataset, RealDataset
from modules.utils import get_model, get_performance, get_device
import torch

with open('experiment_config.json') as json_data:
    cfg_exp = json.load(json_data)

columns = ['model_name', 'omission_rate', 'inclusion_rate', 'bias', 'data_set', 'model_no', 'dice']
rows = []

for (omission_rate, inclusion_rate, bias) in cfg_exp['setup_list']:
    for data_set in cfg_exp['datasets']:
        if data_set in ['hl60', 'granulocytes']:
            test_data = SyntheticDataset(data_path=cfg_exp['path_to_data'], test=True,
                                         drop_p=omission_rate, incl_p=inclusion_rate, max_iter=bias,
                                         file_range=(25, 30))
            # Grayscale images
            n_channels = 1
            n_classes = 2
        else:
            test_data = RealDataset(data_path=cfg_exp['path_to_data'], train_test_split=None, ds='E')
            # RGB images
            n_channels = 3
            n_classes = 2
        for model_name in cfg_exp['networks']:
            for rep in cfg_exp['reps']:
                model = get_model(model_name=model_name, n_channels=n_channels, n_classes=n_classes)
                model_path = cfg_exp['path_to_models'] + \
                             f"{model_name}_{cfg_exp['criterion']}_" \
                             f"omission_{omission_rate}_inclusion_{inclusion_rate}_bias_{bias}_" \
                             f"ds_{data_set}_{rep}_best.pt"
                model.load_state_dict(torch.load(model_path, map_location=get_device(cfg_exp['cuda_no'])))
                tp, fp, fn = get_performance(model, test_data, cuda_no=cfg_exp['cuda_no'])
                dice_score = 2 * tp / (2 * tp + fp + fn)

                row = [model_name, omission_rate, inclusion_rate, bias, data_set, rep, dice_score]
                rows.append(row)

df = pd.DataFrame(rows, columns=columns)
df.to_csv(cfg_exp['path_for_csv'] + cfg_exp['experiment_name'] + '.csv')
