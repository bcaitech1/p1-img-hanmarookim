import argparse
import glob
import json
import os
import random
import re
import time
from importlib import import_module
from pathlib import Path
from attrdict import AttrDict

import numpy as np
import matplotlib.pyplot as plt
import torch
import pandas as pd
from adamp import AdamP
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import Subset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.utils.class_weight import compute_class_weight

from dataset import *
from loss import *
from scheduler import *
from model import *


def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def grid_image(np_images, gts, preds, n=16, shuffle=False):
    batch_size = np_images.shape[0]
    assert n <= batch_size

    choices = random.choices(range(batch_size), k=n) if shuffle else list(range(n))
    figure = plt.figure(figsize=(12, 18 + 2))  # cautions: hardcoded, 이미지 크기에 따라 figsize 를 조정해야 할 수 있습니다. T.T
    plt.subplots_adjust(top=0.8)               # cautions: hardcoded, 이미지 크기에 따라 top 를 조정해야 할 수 있습니다. T.T
    n_grid = np.ceil(n ** 0.5)
    tasks = ["mask", "gender", "age"]
    for idx, choice in enumerate(choices):
        gt = gts[choice].item()
        pred = preds[choice].item()
        image = np_images[choice]
        # title = f"gt: {gt}, pred: {pred}"
        gt_decoded_labels = MaskBaseDataset.decode_multi_class(gt)
        pred_decoded_labels = MaskBaseDataset.decode_multi_class(pred)
        title = "\n".join([
            f"{task} - gt: {gt_label}, pred: {pred_label}"
            for gt_label, pred_label, task
            in zip(gt_decoded_labels, pred_decoded_labels, tasks)
        ])

        plt.subplot(n_grid, n_grid, idx + 1, title=title)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(image, cmap=plt.cm.binary)

    return figure


def increment_path(path, exist_ok=False):
    """ Automatically increment path, i.e. runs/exp --> runs/exp0, runs/exp1 etc.

    Args:
        path (str or pathlib.Path): f"{model_dir}/{args.name}".
        exist_ok (bool): whether increment path (increment if False).
    """
    path = Path(path)
    if (path.exists() and exist_ok) or (not path.exists()):
        return str(path)
    else:
        dirs = glob.glob(f"{path}*")
        matches = [re.search(rf"%s(\d+)" % path.stem, d) for d in dirs]
        i = [int(m.groups()[0]) for m in matches if m]
        n = max(i) + 1 if i else 2
        return f"{path}{n}"


def train(data_dir, model_dir, args):
    seed_everything(args.seed)

    save_dir = increment_path(os.path.join(model_dir, args.name))

    # -- settings
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    # -- dataset
    dataset_module = getattr(import_module("dataset"), args.dataset)  # default: BaseAugmentation
    dataset = dataset_module(
        data_dir=data_dir,
    )
    num_classes = dataset.num_classes  # 18
    dataset_mask = dataset.mask_labels
    dataset_gender = dataset.gender_labels
    dataset_age = dataset.age_labels
    dataset_labels = [mask_label * 6 + gender_label * 3 + age_label for mask_label, gender_label, age_label in zip(dataset.mask_labels, dataset.gender_labels, dataset.age_labels)]

    # -- augmentation
    transform_module = getattr(import_module("dataset"), args.augmentation)  # default: BaseAugmentation
    transform = transform_module(
        resize=args.resize,
        mean=dataset.mean,
        std=dataset.std,
    )
    dataset.set_transform(transform)

    # -- data_loader
    if args.no_validate == 'on':
        dataset.set_val_ratio(0)
    train_set, val_set = dataset.split_dataset()

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        num_workers=2,
        shuffle=True,
        pin_memory=use_cuda,
        drop_last=True,
    )
    if args.no_validate != 'on':
        val_loader = DataLoader(
            val_set,
            batch_size=args.valid_batch_size,
            num_workers=2,
            shuffle=False,
            pin_memory=use_cuda,
            drop_last=True,
        )

    # -- model
    model_module = getattr(import_module("model"), args.model)  # default: BaseModel
    if args['arcface'] == 'on':
        model = model_module(
            num_classes=512
        ).to(device)
        model = torch.nn.DataParallel(model)
        metric_fc = ArcMarginModel(num_classes=num_classes, margin_s=45.0, margin_m=args['arcface_margin_m'])
        metric_fc = torch.nn.DataParallel(metric_fc)
    else:
        model = model_module(
            num_classes=num_classes
        ).to(device)
    # -- loss & metric
    train_info = pd.read_csv('/opt/ml/input/data/train/train.csv')
    if args['class_weights']=='on':
        if args['multi'] == 'on':
            class_weights_mask = compute_class_weight(class_weight="balanced", classes=np.unique(dataset_mask), y=dataset_mask)
            class_weights_gender = compute_class_weight(class_weight="balanced", classes=np.unique(dataset_gender), y=dataset_gender)
            class_weights_age = compute_class_weight(class_weight="balanced", classes=np.unique(dataset_age), y=dataset_age)
            class_weights_mask = torch.tensor(class_weights_mask, dtype=torch.float32)
            class_weights_gender = torch.tensor(class_weights_gender, dtype=torch.float32)
            class_weights_age = torch.tensor(class_weights_age, dtype=torch.float32)
            class_weights = None
        else:
            class_weights = compute_class_weight(class_weight="balanced", classes=np.unique(dataset_labels), y=dataset_labels)
            class_weights = torch.tensor(class_weights, dtype=torch.float32)
    else:
        class_weights = None
    if args['class_weights'] == 'on':
        criterion = create_criterion(args.criterion, weight=class_weights)  # default: cross_entropy
    else:
        criterion = create_criterion(args.criterion)
    if args.optimizer == 'AdamP':
        optimizer = AdamP(model.parameters(), lr=args.lr, weight_decay=5e-4)
    else:
        opt_module = getattr(import_module("torch.optim"), args.optimizer)  # default: SGD
        optimizer = opt_module(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=args.lr,
            weight_decay=5e-4
        )
    scheduler = StepLR(optimizer, args.lr_decay_step, gamma=0.5)

    # -- logging
    logger = SummaryWriter(log_dir=save_dir)
    with open(os.path.join(save_dir, 'config.json'), 'w', encoding='utf-8') as f:
        json.dump(args, f, ensure_ascii=False, indent=4)

    best_val_acc = 0
    best_val_f1 = 0
    best_val_loss = np.inf
    since = time.time()
    for epoch in range(args.epochs):
        # train loop
        model.train()
        if args['arcface'] == 'on':
            metric_fc.train()
        loss_value = 0
        matches = 0
        gt = []
        pred = []
        for idx, train_batch in enumerate(train_loader):
            if args['multi'] == 'on':
                inputs, mask_target, gender_target, age_target = train_batch
                mask_target = mask_target.to(device)
                gender_target = gender_target.to(device)
                age_target = age_target.to(device)
                labels = mask_target * 6 + age_target * 3 + age_target
            else:
                inputs, labels = train_batch

            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outs = model(inputs)

            if args['arcface'] == 'on':
                outs = metric_fc(outs, labels)
                #print(outs)
            if args['multi'] == 'on':
                mask_pred = torch.argmax(outs[:, 0:3], dim=-1)
                gender_pred = torch.argmax(outs[:, 3:5], dim=-1)
                age_pred = torch.argmax(outs[:, 5:8], dim=-1)
                preds = mask_pred * 6 + gender_pred * 3 + age_pred

            else:
                preds = torch.argmax(outs, dim=-1)
            #print(preds)
            gt += list(map(int, labels))
            pred += list(map(int, preds))

            if args['multi'] == 'on':
                if args['class_weights'] == 'on':
                    criterion = create_criterion(args.criterion, num_classes=3, weight=class_weights_mask)
                else:
                    criterion = create_criterion(args.criterion, num_classes=3)
                mask_loss = criterion(outs[:, 0:3], mask_target)
                if args['class_weights'] == 'on':
                    criterion = create_criterion(args.criterion, num_classes=2, weight=class_weights_mask)
                else:
                    criterion = create_criterion(args.criterion, num_classes=2)
                gender_loss = criterion(outs[:, 3:5], gender_target)
                if args['class_weights'] == 'on':
                    criterion = create_criterion(args.criterion, num_classes=3, weight=class_weights_mask)
                else:
                    criterion = create_criterion(args.criterion, num_classes=3)
                age_loss = criterion(outs[:, 5:8], age_target)
                loss = mask_loss + gender_loss + age_loss
            else:
                loss = criterion(outs, labels)

            loss.backward()
            optimizer.step()

            loss_value += loss.item()
            matches += (preds == labels).sum().item()
            if (idx + 1) % args.log_interval == 0:
                train_loss = loss_value / args.log_interval
                train_acc = matches / args.batch_size / args.log_interval
                train_f1 = f1_score(gt, pred, average='macro')
                current_lr = get_lr(optimizer)
                time_elapse = time.time() - since
                print(
                    f"Epoch[{epoch}/{args.epochs}]({idx + 1}/{len(train_loader)}) ({time_elapse//60}m {time_elapse%60}s) || "
                    f"training loss {train_loss:4.4} || training accuracy {train_acc:4.2%} || training f1 {train_f1:4.2%} || lr {current_lr}"
                )
                logger.add_scalar("Train/loss", train_loss, epoch * len(train_loader) + idx)
                logger.add_scalar("Train/accuracy", train_acc, epoch * len(train_loader) + idx)
                logger.add_scalar("Train/f1", train_f1, epoch * len(train_loader) + idx)
                loss_value = 0
                matches = 0

        scheduler.step()
        if args['no_validate'] == 'on':
            print(f"saving the best model..")
            try:
                torch.save(model.module.state_dict(), f"{save_dir}/{epoch}_{train_f1:4.2%}.pth")
            except:
                torch.save(model.state_dict(), f"{save_dir}/{epoch}_{train_f1:4.2%}.pth")
            if args['arcface'] == 'on':
                torch.save(metric_fc.module.state_dict(), f"{save_dir}/metric_epoch_{epoch}.pth")

        else:
            # val loop
            with torch.no_grad():
                print("Calculating validation results...")
                model.eval()
                if args['arcface'] == 'on':
                    metric_fc.eval()
                val_loss_items = []
                val_acc_items = []
                figure = None
                gt = []
                pred = []
                for val_batch in val_loader:
                    if args['multi'] == 'on':
                        inputs, mask_target, gender_target, age_target = val_batch
                        mask_target = mask_target.to(device)
                        gender_target = gender_target.to(device)
                        age_target = age_target.to(device)
                        labels = mask_target * 6 + gender_target * 3 + age_target
                    else:
                        inputs, labels = val_batch
                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    outs = model(inputs)
                    if args['arcface'] == 'on':
                        outs = metric_fc(outs, labels)
                    if args['multi'] == 'on':
                        mask_pred = torch.argmax(outs[:, 0:3], dim=-1)
                        gender_pred = torch.argmax(outs[:, 3:5], dim=-1)
                        age_pred = torch.argmax(outs[:, 5:8], dim=-1)
                        preds = mask_pred * 6 + gender_pred * 3 + age_pred
                    else:
                        preds = torch.argmax(outs, dim=-1)

                    gt += list(map(int, labels))
                    pred += list(map(int, preds))
                    if args['multi'] == 'on':
                        if args['class_weights'] == 'on':
                            criterion = create_criterion(args.criterion, weight=class_weights_mask)
                        mask_loss = criterion(outs[:, 0:3], mask_target)
                        if args['class_weights'] == 'on':
                            criterion = create_criterion(args.criterion, weight=class_weights_gender)
                        gender_loss = criterion(outs[:, 3:5], gender_target)
                        if args['class_weights'] == 'on':
                            criterion = create_criterion(args.criterion, weight=class_weights_age)
                        age_loss = criterion(outs[:, 5:8], age_target)
                        loss_item = mask_loss.item() + gender_loss.item() + age_loss.item()
                    else:
                        loss_item = criterion(outs, labels).item()
                    acc_item = (labels == preds).sum().item()
                    val_loss_items.append(loss_item)
                    val_acc_items.append(acc_item)
                    if figure is None:
                        inputs_np = torch.clone(inputs).detach().cpu().permute(0, 2, 3, 1).numpy()
                        inputs_np = dataset_module.denormalize_image(inputs_np, dataset.mean, dataset.std)
                        figure = grid_image(inputs_np, labels, preds, args.dataset != "MaskSplitByProfileDataset")

                val_loss = np.sum(val_loss_items) / len(val_loader)
                val_acc = np.sum(val_acc_items) / len(val_set)
                val_f1 = f1_score(gt, pred, average="macro")
                best_val_loss = min(best_val_loss, val_loss)
                if val_acc > best_val_acc:
                    print(f"New best model for val accuracy : {val_acc:4.2%}! saving the best model..")
                    try:
                        torch.save(model.module.state_dict(), f"{save_dir}/best_accuracy.pth")
                    except:
                        torch.save(model.state_dict(), f"{save_dir}/best_accuracy.pth")
                    if args['arcface'] == 'on':
                        torch.save(metric_fc.module.state_dict(), f"{save_dir}/best_accuracy_metric.pth")
                    best_val_acc = val_acc
                if val_f1 > best_val_f1:
                    print(f"New best model for val f1 : {val_f1:4.2%}! saving the best model..")
                    try:
                        torch.save(model.module.state_dict(), f"{save_dir}/best_f1.pth")
                    except:
                        torch.save(model.state_dict(), f"{save_dir}/best_f1.pth")
                    if args['arcface'] == 'on':
                        torch.save(metric_fc.module.state_dict(), f"{save_dir}/best_f1_metric.pth")
                    best_val_f1 = val_f1
                try:
                    torch.save(model.module.state_dict(), f"{save_dir}/last.pth")
                except:
                    torch.save(model.state_dict(), f"{save_dir}/last.pth")
                if args['arcface'] == 'on':
                    torch.save(metric_fc.module.state_dict(), f"{save_dir}/last_metric.pth")
                time_elapse = time.time() - since
                print(
                    f"({time_elapse//60}m {time_elapse%60}s) [Val] acc : {val_acc:4.2%}, loss: {val_loss:4.2} || "
                    f"best acc : {best_val_acc:4.2%}, best f1 : {best_val_f1:4.2%}, best loss: {best_val_loss:4.2}"
                )
                logger.add_scalar("Val/loss", val_loss, epoch)
                logger.add_scalar("Val/accuracy", val_acc, epoch)
                logger.add_scalar("Val/f1", val_f1, epoch)
                logger.add_figure("results", figure, epoch)
                print()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    """
    # Data and model checkpoints directories
    parser.add_argument('--seed', type=int, default=42, help='random seed (default: 42)')
    parser.add_argument('--epochs', type=int, default=1, help='number of epochs to train (default: 1)')
    parser.add_argument('--dataset', type=str, default='MaskBaseDataset', help='dataset augmentation type (default: MaskBaseDataset)')
    parser.add_argument('--augmentation', type=str, default='BaseAugmentation', help='data augmentation type (default: BaseAugmentation)')
    parser.add_argument("--resize", type=int, default=224, help='resize size for image when training')
    parser.add_argument('--batch_size', type=int, default=64, help='input batch size for training (default: 64)')
    parser.add_argument('--valid_batch_size', type=int, default=64, help='input batch size for validing (default: 1000)')
    parser.add_argument('--model', type=str, default='BaseModel', help='model type (default: BaseModel)')
    parser.add_argument('--optimizer', type=str, default='SGD', help='optimizer type (default: SGD)')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate (default: 1e-3)')
    parser.add_argument('--val_ratio', type=float, default=0.2, help='ratio for validaton (default: 0.2)')
    parser.add_argument('--criterion', type=str, default='cross_entropy', help='criterion type (default: cross_entropy)')
    parser.add_argument('--lr_decay_step', type=int, default=20, help='learning rate scheduler deacy step (default: 20)')
    parser.add_argument('--log_interval', type=int, default=20, help='how many batches to wait before logging training status')
    parser.add_argument('--name', default='exp', help='model save at {SM_MODEL_DIR}/{name}')

    # Container environment
    parser.add_argument('--data_dir', type=str, default=os.environ.get('SM_CHANNEL_TRAIN', '/opt/ml/input/data/train/images'))
    parser.add_argument('--model_dir', type=str, default=os.environ.get('SM_MODEL_DIR', './model'))
    """
    parser.add_argument('--config_file', type=str, required=True)
    args = parser.parse_args()
    with open(args.config_file) as f:
        args = AttrDict(json.load(f))

    print(args)

    data_dir = os.environ.get('SM_CHANNEL_TRAIN', '/opt/ml/input/data/train/images')
    model_dir = os.environ.get('SM_MODEL_DIR', './model')

    train(data_dir, model_dir, args)