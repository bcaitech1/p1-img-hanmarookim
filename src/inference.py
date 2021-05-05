import argparse
import os
from attrdict import AttrDict
import json
from importlib import import_module

import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from dataset import TestDataset, MaskBaseDataset
from model import *

def load_model(saved_model, num_classes, device):
    model_cls = getattr(import_module("model"), args.model)
    model = model_cls(
        num_classes=num_classes
    )

    # tarpath = os.path.join(saved_model, 'best.tar.gz')
    # tar = tarfile.open(tarpath, 'r:gz')
    # tar.extractall(path=saved_model)

    model_path = saved_model
    model.load_state_dict(torch.load(model_path, map_location=device))

    return model

def custom_load_model(model_name):
    model_cls = getattr(import_module("model"), model_name)
    model = model_cls(
        num_classes = 18
    )
    return model

@torch.no_grad()
def inference(data_dir, model_dir, output_dir, args):
    """
    """
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    s = model_dir.split('/')

    with open('/'.join(s[:-1]) + '/config.json') as f:
        train_config = AttrDict(json.load(f))
    num_classes = MaskBaseDataset.num_classes  # 18


    model = load_model(model_dir, num_classes, device).to(device)
    model.eval()
    img_root = os.path.join(data_dir, 'images')

    info_path = os.path.join(data_dir, 'info.csv')
    info = pd.read_csv(info_path)

    img_paths = [os.path.join(img_root, img_id) for img_id in info.ImageID]
    dataset = TestDataset(img_paths, args.resize)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=2,
        shuffle=False,
        pin_memory=use_cuda,
        drop_last=False,
    )

    print("Calculating inference results..")
    preds = []
    with torch.no_grad():
        for idx, images in enumerate(loader):
            images = images.to(device)
            pred = model(images)
            if args.multi == 'on':
                mask_pred = torch.argmax(pred[:, 0:3], dim=-1)
                gender_pred = torch.argmax(pred[:, 3:5], dim=-1)
                age_pred = torch.argmax(pred[:, 5:8], dim=-1)
                pred = mask_pred * 6 + gender_pred * 3 + age_pred
            else:
                pred = pred.argmax(dim=-1)
            preds.extend(pred.cpu().numpy())

    info['ans'] = preds
    s = model_dir.split('/')
    info.to_csv(os.path.join(output_dir, f'{s[-2]}_output.csv'), index=False)
    print(f'Inference Done!')

def custom_inference(data_dir, output_dir, args):
    """
    for ensemble in image classification 0408
    """
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    model_multipred1 = custom_load_model('Testload').to(device)
    model_multipred2 = custom_load_model('MultiPredict4Model').to(device)
    model_single1 = custom_load_model('ViTModel').to(device)
    model_single2 = custom_load_model('EfficientNet4Model').to(device)

    model_multipred1.load_state_dict(torch.load('/opt/ml/pycharm/src/model/conf_31_35_vit16_customloss_adamp1e-5_noval_multipred/4_22.24%.pth', map_location=device))
    model_single1.load_state_dict(torch.load('/opt/ml/pycharm/src/model/conf_24_vit16_ce2/best_f1.pth', map_location=device))
    model_multipred2.load_state_dict(
        torch.load('/opt/ml/pycharm/src/model/conf_30_efb4_customloss_adamp1e-5_noval_multipred5/4_20.78%.pth',
                   map_location=device))
    model_single2.load_state_dict(
        torch.load('/opt/ml/pycharm/src/model/conf_17_focal_b4_1e-2/best_accuracy_49.pth', map_location=device))
    model_multipred1.eval()
    model_single1.eval()
    model_multipred2.eval()
    model_single2.eval()

    img_root = os.path.join(data_dir, 'images')

    info_path = os.path.join(data_dir, 'info.csv')
    info = pd.read_csv(info_path)

    img_paths = [os.path.join(img_root, img_id) for img_id in info.ImageID]
    dataset = TestDataset(img_paths, args.resize)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=2,
        shuffle=False,
        pin_memory=use_cuda,
        drop_last=False,
    )

    print("Calculating inference results..")
    soft_preds = []
    hard_preds = []
    with torch.no_grad():
        for idx, images in enumerate(loader):
            images = images.to(device)
            pred_multi1 = model_multipred1(images)
            pred_multi2 = model_multipred2(images)

            mask_pred1 = F.softmax(pred_multi1[:, 0:3], dim=1)
            gender_pred1 = F.softmax(pred_multi1[:, 3:5], dim=1)
            age_pred1 = F.softmax(pred_multi1[:, 5:8], dim=1)
            mask_pred2 = F.softmax(pred_multi2[:, 0:3], dim=1)
            gender_pred2 = F.softmax(pred_multi2[:, 3:5], dim=1)
            age_pred2 = F.softmax(pred_multi2[:, 5:8], dim=1)


            """
            soft_m1 = []
            soft_m2 = []
            for m in range(3):
                for g in range(2):
                    for a in range(3):
                        soft_m1.append(mask_pred1[:, m] * gender_pred1[:, g] * age_pred1[:, a])
                        soft_m2.append(mask_pred2[:, m] * gender_pred2[:, g] * age_pred2[:, a])
            soft_m1 = torch.stack(soft_m1, dim=1)
            soft_m2 = torch.stack(soft_m2, dim=1)
            """
            pred_single1 = model_single1(images)
            pred_single2 = model_single2(images)
            """
            soft_s1 = F.softmax(pred_single1, dim=1)
            soft_s2 = F.softmax(pred_single2, dim=1)

            soft_pred = (soft_m1 + soft_s1 + soft_m2 + soft_s2) / 4
            soft_pred = pred.argmax(dim=-1)
            soft_preds.extend(pred.cpu().numpy())
            """
            m = torch.argmax(pred_multi1[:, 0:3], dim=-1)
            g = torch.argmax(pred_multi1[:, 3:5], dim=-1)
            a = torch.argmax(pred_multi1[:, 5:8], dim=-1)
            hard_m1 = m * 6 + g * 3 + a
            m = torch.argmax(pred_multi2[:, 0:3], dim=-1)
            g = torch.argmax(pred_multi2[:, 3:5], dim=-1)
            a = torch.argmax(pred_multi2[:, 5:8], dim=-1)
            hard_m2 = m * 6 + g * 3 + a
            hard_s1 = pred_single1.argmax(dim=-1)
            hard_s2 = pred_single2.argmax(dim=-1)
            hard_pred = []
            hard_pred.append(hard_m1)
            hard_pred.append(hard_m2)
            hard_pred.append(hard_s1)
            hard_pred.append(hard_s2)
            hard_pred2 = []
            for i in torch.stack(hard_pred, dim=1):
                l, c = torch.unique_consecutive(i, return_counts=True)
                hard_pred2.append(i[c.argmax()])
            hard_pred = torch.tensor(hard_pred2)
            hard_preds.extend(hard_pred.cpu().numpy())
    info['ans'] = hard_preds
    info.to_csv(os.path.join(output_dir, f'ensemble_m12_s12_hard_output.csv'), index=False)
    print(f'Inference Done!')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Data and model checkpoints directories
    parser.add_argument('--batch_size', type=int, default=32, help='input batch size for validing (default: 1000)')
    parser.add_argument('--resize', type=tuple, default=(384, 384), help='resize size for image when you trained (default: (96, 128))')
    parser.add_argument('--model', type=str, default='BaseModel', help='model type (default: BaseModel)')
    parser.add_argument('--arcface', type=str, default='off')
    parser.add_argument('--metric_dir', type=str, default='off')
    parser.add_argument('--multi', type=str, default='off')
    parser.add_argument('--last', type=str, default='off')

    # Container environment
    parser.add_argument('--data_dir', type=str, default=os.environ.get('SM_CHANNEL_EVAL', '/opt/ml/input/data/eval'))
    parser.add_argument('--model_dir', type=str, default=os.environ.get('SM_CHANNEL_MODEL', './model'))
    parser.add_argument('--output_dir', type=str, default=os.environ.get('SM_OUTPUT_DATA_DIR', './output'))

    args = parser.parse_args()

    data_dir = args.data_dir
    model_dir = args.model_dir
    output_dir = args.output_dir

    os.makedirs(output_dir, exist_ok=True)

    if args.last == 'on':
        custom_inference(data_dir, output_dir, args)
    else:
        inference(data_dir, model_dir, output_dir, args)

