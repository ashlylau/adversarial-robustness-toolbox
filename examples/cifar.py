import sys
import os
import numpy as np
import argparse
import time
import json
from pathlib import Path

import argparse
import os
import shutil
import sys

import numpy as np
import torch
import random 
random.seed(42)
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
from torch.utils.data import DataLoader, TensorDataset
import torch.utils.data.distributed
from sklearn.metrics import roc_auc_score, accuracy_score
import torch.utils.tensorboard as tensorboard
import torchvision.transforms as transforms
from opacus import PrivacyEngine
# from opacus.layers import DifferentiallyPrivateDistributedDataParallel as DPDDP
from opacus.utils import stats
from opacus.utils.uniform_sampler import UniformWithReplacementSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torchvision.datasets import CIFAR10
from tqdm import tqdm
from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import train_test_split

sys.path.append(os.path.abspath('../'))
from art.estimators.classification import PyTorchClassifier
from art.utils import load_mnist
from art.attacks.inference.membership_inference import MembershipInferenceBlackBoxRuleBased, MembershipInferenceBlackBox
from art.utils import load_dataset

def convnet(num_classes):
    return nn.Sequential(
        nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
        nn.ReLU(),
        nn.AvgPool2d(kernel_size=2, stride=2),
        nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
        nn.ReLU(),
        nn.AvgPool2d(kernel_size=2, stride=2),
        nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
        nn.ReLU(),
        nn.AvgPool2d(kernel_size=2, stride=2),
        nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
        nn.ReLU(),
        nn.AdaptiveAvgPool2d((1, 1)),
        nn.Flatten(start_dim=1, end_dim=-1),
        nn.Linear(128, num_classes, bias=True),
    )
    
def accuracy(preds, labels):
    return (preds == labels).mean()

def save_checkpoint(state, is_best, filename="checkpoint.tar"):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, "model_best.pth.tar")

def train(model, train_loader, optimizer, epoch, device):
    model.train()
    criterion = nn.CrossEntropyLoss()

    losses = []
    top1_acc = []

    for i, (images, target) in enumerate(train_loader):
        images = images.to(device)
        target = target.to(device)

        # compute output
        output = model(images)
        loss = criterion(output, target)
        preds = np.argmax(output.detach().cpu().numpy(), axis=1)
        labels = target.detach().cpu().numpy()
        # measure accuracy and record loss
        acc1 = accuracy(preds, labels)

        losses.append(loss.item())
        top1_acc.append(acc1)
        stats.update(stats.StatType.TRAIN, acc1=acc1)

        # compute gradient and do SGD step
        loss.backward()

        # make sure we take a step after processing the last mini-batch in the
        # epoch to ensure we start the next epoch with a clean state
        if ((i + 1) % n_accumulation_steps == 0) or ((i + 1) == len(train_loader)):
            optimizer.step()
            optimizer.zero_grad()
        else:
            optimizer.virtual_step()

        if i % print_freq == 0:
            if not args.disable_dp:
                epsilon, best_alpha = optimizer.privacy_engine.get_privacy_spent(
                    _delta
                )
                print(
                    f"\tTrain Epoch: {epoch} \t"
                    f"Loss: {np.mean(losses):.6f} "
                    f"Acc@1: {np.mean(top1_acc):.6f} "
                    f"(ε = {epsilon:.2f}, δ = {_delta}) for α = {best_alpha}"
                )
            else:
                print(
                    f"\tTrain Epoch: {epoch} \t"
                    f"Loss: {np.mean(losses):.6f} "
                    f"Acc@1: {np.mean(top1_acc):.6f} "
                )


def test(model, test_loader, device):
    model.eval()
    criterion = nn.CrossEntropyLoss()
    losses = []
    top1_acc = []

    with torch.no_grad():
        for images, target in test_loader:
            images = images.to(device)
            target = target.to(device)

            output = model(images)
            loss = criterion(output, target)
            preds = np.argmax(output.detach().cpu().numpy(), axis=1)
            labels = target.detach().cpu().numpy()
            acc1 = accuracy(preds, labels)

            losses.append(loss.item())
            top1_acc.append(acc1)

    top1_avg = np.mean(top1_acc)
    stats.update(stats.StatType.TEST, acc1=top1_avg)

    print(f"\tTest set:" f"Loss: {np.mean(losses):.6f} " f"Acc@1: {top1_avg :.6f} ")
    return np.mean(top1_acc)


def calc_precision_recall(predicted, actual, positive_value=1):
    score = 0  # both predicted and actual are positive
    num_positive_predicted = 0  # predicted positive
    num_positive_actual = 0  # actual positive
    for i in range(len(predicted)):
        if predicted[i] == positive_value:
            num_positive_predicted += 1
        if actual[i] == positive_value:
            num_positive_actual += 1
        if predicted[i] == actual[i]:
            if predicted[i] == positive_value:
                score += 1
    
    if num_positive_predicted == 0:
        precision = 1
    else:
        precision = score / num_positive_predicted  # the fraction of predicted “Yes” responses that are correct
    if num_positive_actual == 0:
        recall = 1
    else:
        recall = score / num_positive_actual  # the fraction of “Yes” responses that are predicted correctly

    return precision, recall




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Membership Inference Attack on Resampling')
    parser.add_argument('--dataset', default='cifar', help='dataset to test')
    parser.add_argument('--batch_size', type=int, default=128, help='batch size')
    parser.add_argument('--epochs', type=int, default=60, help='num epochs')
    parser.add_argument('--noise_multiplier', type=float, default=1.3, help='noise multiplier')
    parser.add_argument('--max_grad_norm', type=float, default=10.0, help='max grad norm')
    parser.add_argument('--lr', type=float, default=1, help='learning rate')
    parser.add_argument('--delta', type=float, default=0.00002, help='delta')
    parser.add_argument('--disable_dp', action='store_true', default=False, help='train non-private model')
    parser.add_argument('--load_model', action='store_true', default=False, help='use pre trained model')
    parser.add_argument('--perform_aug', action='store_true', default=False, help='perform data augmentation')
    parser.add_argument('--sampling_type', default='none', help='over, under or smote sampling')
    parser.add_argument('--attack_model', default='rf', help='attack model type -- rf, nn')
    parser.add_argument('--sampling_ratio', type=float, default=0.5, help='sampling ratio')
    args = parser.parse_args()
    print(vars(args))

    device = torch.device('cpu')
    start = time.time()
    epsilon = -1

    # hparams
    _sample_rate=0.04
    batch_size_test=256
    workers=2
    wd=0
    _weight_decay=0
    _momentum=0.9
    na=1
    n_accumulation_steps=1
    local_rank=-1
    lr_schedule='cos'
    _optim='SGD'
    log_dir=""
    data_root='../cifar10'
    checkpoint_file='checkpoint'
    _delta=1e-5
    _secure_rng=False
    resume=""
    print_freq=10
    
    # Load data
    generator=None
    augmentations = [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
    ]
    normalize = [
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ]
    
    if args.perform_aug:
        train_transform = transforms.Compose(
        augmentations + normalize
        # augmentations + normalize if disable_dp else normalize
    )
    else:
        train_transform = transforms.Compose(normalize)

    test_transform = transforms.Compose(normalize)

    train_dataset = CIFAR10(
        root=data_root, train=True, download=True, transform=train_transform
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        num_workers=workers,
        generator=generator,
        batch_sampler=UniformWithReplacementSampler(
            num_samples=len(train_dataset),
            sample_rate=_sample_rate,
            generator=generator,
        ),
    )

    test_dataset = CIFAR10(
        root=data_root, train=False, download=True, transform=train_transform
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size_test,
        shuffle=False,
        num_workers=workers,
    )

    X = np.empty(shape=(0,3,32,32))
    y = np.empty(shape=(0))
    for images, target in train_loader:
        X = np.append(X, images, axis=0)
        y = np.append(y, target, axis=0)
    for images, target in test_loader:
        X = np.append(X, images, axis=0)
        y = np.append(y, target, axis=0)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)
    print(X_train.shape)
    print(X_test.shape)


    # Perform resampling
    if args.sampling_type == 'over':
        print("Performing oversampling at {}".format(args.sampling_ratio))
        X_train, y_train = RandomOverSampler(sampling_strategy=args.sampling_ratio).fit_resample(X_train, y_train)
    elif args.sampling_type == 'under':
        print("Performing undersampling at {}".format(args.sampling_ratio))
        X_train, y_train = RandomUnderSampler(sampling_strategy=args.sampling_ratio).fit_resample(X_train, y_train)
    elif args.sampling_type == 'smote':
        print("Performing SMOTE oversampling at {}".format(args.sampling_ratio))
        X_train, y_train = SMOTE(sampling_strategy=args.sampling_ratio).fit_resample(X_train, y_train)


    print(X_train.shape)
    print(y_train.shape)


    clipping = {"clip_per_layer": False, "enable_stat": True}
    best_acc1 = 0
    device = torch.device("cuda")

    model = convnet(num_classes=10)
    model = model.to(device)

    if _optim == "SGD":
        optimizer = optim.SGD(
            model.parameters(),
            lr=args.lr,
            momentum=_momentum,
            weight_decay=_weight_decay,
        )
    elif _optim == "RMSprop":
        optimizer = optim.RMSprop(model.parameters(), lr=args.lr)
    elif _optim == "Adam":
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
    else:
        raise NotImplementedError("Optimizer not recognized. Please check spelling")

    if not args.disable_dp:
        privacy_engine = PrivacyEngine(
            model,
            sample_rate=_sample_rate * n_accumulation_steps,
            alphas=[1 + x / 10.0 for x in range(1, 100)] + list(range(12, 64)),
            noise_multiplier=args.noise_multiplier,
            max_grad_norm=args.max_grad_norm,
            secure_rng=_secure_rng,
            **clipping,
        )
        privacy_engine.attach(optimizer)

    if not args.load_model:
        for epoch in range(1, args.epochs + 1):
            if lr_schedule == "cos":
                lr = args.lr * 0.5 * (1 + np.cos(np.pi * epoch / (args.epochs + 1)))
                for param_group in optimizer.param_groups:
                    param_group["lr"] = lr

            train(model, train_loader, optimizer, epoch, device)
            top1_acc = test(model, test_loader, device)

            # remember best acc@1 and save checkpoint
            is_best = top1_acc > best_acc1
            best_acc1 = max(top1_acc, best_acc1)

            save_checkpoint(
                {
                    "epoch": epoch + 1,
                    "arch": "Convnet",
                    "state_dict": model.state_dict(),
                    "best_acc1": best_acc1,
                    "optimizer": optimizer.state_dict(),
                },
                is_best,
                filename=checkpoint_file + ".tar",
            )
    else:
        checkpoint = torch.load('model_best.pth.tar')
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])

    art_classifier = PyTorchClassifier(model, loss=nn.CrossEntropyLoss(),
                                        optimizer=optimizer,
                                        input_shape=(3,32,32),
                                        nb_classes=10,)
    
    pred = np.array([np.argmax(arr) for arr in art_classifier.predict(torch.from_numpy(X_test).type(torch.FloatTensor))])
    acc = accuracy_score(y_test, pred)
    print('Base private model accuracy: ', acc)


    # Black box attack
    attack_train_ratio = 0.5
    attack_train_size = int(len(X_train) * attack_train_ratio)
    attack_test_size = int(len(X_test) * attack_train_ratio)

    mlp_attack_bb = MembershipInferenceBlackBox(art_classifier, attack_model_type=args.attack_model)

    # train attack model
    mlp_attack_bb.fit(torch.from_numpy(X_train[:attack_train_size]).type(torch.FloatTensor), torch.from_numpy(y_train[:attack_train_size]).type(torch.FloatTensor),
                torch.from_numpy(X_test[:attack_test_size]).type(torch.FloatTensor), torch.from_numpy(y_test[:attack_test_size]).type(torch.FloatTensor))

    # infer 
    mlp_inferred_train_bb = mlp_attack_bb.infer(torch.from_numpy(X_train).type(torch.FloatTensor), torch.from_numpy(y_train).type(torch.FloatTensor))
    mlp_inferred_test_bb = mlp_attack_bb.infer(torch.from_numpy(X_test).type(torch.FloatTensor), torch.from_numpy(y_test).type(torch.FloatTensor))

    # check accuracy
    print("Random forest model attack results: ")
    mlp_train_acc_bb = np.sum(mlp_inferred_train_bb) / len(mlp_inferred_train_bb)
    mlp_test_acc_bb = 1 - (np.sum(mlp_inferred_test_bb) / len(mlp_inferred_test_bb))
    mlp_acc_bb = (mlp_train_acc_bb * len(mlp_inferred_train_bb) + mlp_test_acc_bb * len(mlp_inferred_test_bb)) / (len(mlp_inferred_train_bb) + len(mlp_inferred_test_bb))
    print('train acc: {}'.format(mlp_train_acc_bb))
    print('test acc: {}'.format(mlp_test_acc_bb))
    print('total acc: {}'.format(mlp_acc_bb))
    mlp_prec_recall_bb = calc_precision_recall(np.concatenate((mlp_inferred_train_bb, mlp_inferred_test_bb)), 
                                np.concatenate((np.ones(len(mlp_inferred_train_bb)), np.zeros(len(mlp_inferred_test_bb)))))
    print('precision, recall: {}'.format(mlp_prec_recall_bb))
    rf_results = {
        'train acc': mlp_train_acc_bb,
        'test acc': mlp_test_acc_bb,
        'total acc': mlp_acc_bb,
        'prec, recall': mlp_prec_recall_bb
    }

    results = [epsilon, acc, mlp_test_acc_bb, mlp_acc_bb]

    results_json = {
        'experiment_args': vars(args),
        'experiment_time': time.time() - start,
        'epsilon': epsilon,
        'accuracy': acc,
        'rf_acc': rf_results,
    }

    print(results)


    # Create experiment directory.
    experiment_path = f'/homes/al5217/adversarial-robustness-toolbox/examples/{args.dataset}/{args.sampling_type}/'
    experiment_number = len(os.listdir(experiment_path))
    print("experiment_number: {}".format(experiment_number))

    # Dump the results to file
    json_file = Path.cwd() / f'{args.dataset}/{args.sampling_type}/test_results-{experiment_number}.json'
    with json_file.open('w') as f:
        json.dump(results_json, f, indent="  ")

    print("dumped results to {}".format(str(json_file)))