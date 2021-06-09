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
from sklearn.metrics import roc_auc_score, accuracy_score, classification_report
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
from art.attacks.inference.membership_inference import MembershipInferenceBlackBox

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
                    args.delta
                )
                print(
                    f"\tTrain Epoch: {epoch} \t"
                    f"Loss: {np.mean(losses):.6f} "
                    f"Acc@1: {np.mean(top1_acc):.6f} "
                    f"(ε = {epsilon:.2f}, δ = {args.delta}) for α = {best_alpha}"
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
    parser.add_argument('--epochs', type=int, default=25, help='num epochs')
    parser.add_argument('--noise_multiplier', type=float, default=1.3, help='noise multiplier')
    parser.add_argument('--target_epsilon', type=float, default=10.0, help='target epsilon')
    parser.add_argument('--max_grad_norm', type=float, default=10.0, help='max grad norm')
    parser.add_argument('--lr', type=float, default=0.1, help='learning rate')
    parser.add_argument('--delta', type=float, default=1e-5, help='delta')
    parser.add_argument('--disable_dp', action='store_true', default=False, help='train non-private model')
    parser.add_argument('--load_model', action='store_true', default=False, help='use pre trained model')
    parser.add_argument('--perform_aug', action='store_true', default=False, help='perform data augmentation')
    parser.add_argument('--sampling_type', default='none', help='over, under or smote sampling')
    parser.add_argument('--attack_model', default='rf', help='attack model type -- rf, nn')
    parser.add_argument('--sampling_ratio', type=float, default=0.5, help='sampling ratio')
    args = parser.parse_args()
    print(vars(args))

    if torch.cuda.is_available():
        print('cuda')
        device = torch.device("cuda")
    else:
        print('cpu')
        device = torch.device('cpu')
        
    start = time.time()

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
        transform = transforms.Compose(
        augmentations + normalize
        # augmentations + normalize if disable_dp else normalize
    )
    else:
        transform = transforms.Compose(normalize)

    train_dataset = CIFAR10(
        root=data_root, train=True, download=True, transform=transform
    )

    test_dataset = CIFAR10(
        root=data_root, train=False, download=True, transform=transform
    )
    
    # create numpy arrays from dataloaders
    X = np.append(train_dataset.data, test_dataset.data, axis=0)
    y = np.append(train_dataset.targets, test_dataset.targets, axis=0)
    print(X.shape)
    print('original counts: ', torch.bincount(torch.IntTensor(y).squeeze()))
    
    # remove 1/6 of the minority class
    min_classes = [0,1,2,3,4]
    maj_classes = [5,6,7,8,9]
    cat_indices = []
    for i in range(len(X)):
        if y[i] in min_classes and (i % 6 != 0):
            cat_indices.append(i)
    print('removed indices', len(cat_indices))

    X = np.delete(X, cat_indices, axis=0)
    y = np.delete(y, cat_indices, axis=0)

    print(X.shape) 
    print('imbalanced counts: ', torch.bincount(torch.IntTensor(y).squeeze()))

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)
    X_train_orig, X_test_orig, y_train_orig, y_test_orig = X_train, X_test, y_train, y_test

    orig_train_counts = torch.bincount(torch.IntTensor(y_train).squeeze())
    orig_test_counts = torch.bincount(torch.IntTensor(y_test).squeeze())
    print('original train counts: ', orig_train_counts)
    print('original test counts: ', orig_test_counts)
    max_count = max(max(orig_test_counts).item(), max(orig_train_counts).item())
    min_count = min(min(orig_test_counts).item(), min(orig_train_counts).item())
   
    # Perform resampling
    sampling_dict = {}
    if args.sampling_type == 'over':
        print('performing oversampling at ', args.sampling_ratio)
        for i in min_classes:
            sampling_dict[float(i)] = int(args.sampling_ratio * max_count)
        for i in maj_classes:
            sampling_dict[float(i)] = max_count
        print(sampling_dict)
        _X_train, y_train = RandomOverSampler(sampling_strategy=sampling_dict).fit_resample(X_train.reshape(len(X_train), 3*32*32), y_train)
        _X_test, y_test = RandomOverSampler(sampling_strategy=sampling_dict).fit_resample(X_test.reshape(len(X_test), 3*32*32), y_test)
        X_train = _X_train.reshape(len(_X_train), 32, 32, 3)
        X_test = _X_test.reshape(len(_X_test), 32, 32, 3)
    elif args.sampling_type == 'under':
        print('performing undersampling at ', args.sampling_ratio)
        for i in min_classes:
            sampling_dict[float(i)] = min_count
        for i in maj_classes:
            sampling_dict[float(i)] = int((1/args.sampling_ratio) * min_count)
        print(sampling_dict)
        _X_train, y_train = RandomUnderSampler(sampling_strategy=sampling_dict).fit_resample(X_train.reshape(len(X_train), 3*32*32), y_train)
        _X_test, y_test = RandomUnderSampler(sampling_strategy=sampling_dict).fit_resample(X_test.reshape(len(X_test), 3*32*32), y_test)
        X_train = _X_train.reshape(len(_X_train), 32, 32, 3)
        X_test = _X_test.reshape(len(_X_test), 32, 32, 3)
    elif args.sampling_type == 'smote':
        print('performing smote resampling at ', args.sampling_ratio)
        for i in min_classes:
            sampling_dict[float(i)] = int(args.sampling_ratio * max_count)
        for i in maj_classes:
            sampling_dict[float(i)] = max_count
        print(sampling_dict)
        _X_train, y_train = SMOTE(sampling_strategy=sampling_dict).fit_resample(X_train.reshape(len(X_train), 3*32*32), y_train)
        _X_test, y_test = SMOTE(sampling_strategy=sampling_dict).fit_resample(X_test.reshape(len(X_test), 3*32*32), y_test)
        X_train = _X_train.reshape(len(_X_train), 32, 32, 3)
        X_test = _X_test.reshape(len(_X_test), 32, 32, 3)


    X_train, X_test = X_train.transpose(0,3,1,2), X_test.transpose(0,3,1,2)
    X_train_orig, X_test_orig = X_train_orig.transpose(0,3,1,2), X_test_orig.transpose(0,3,1,2)

    train_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(torch.Tensor(X_train), torch.LongTensor(y_train)),
        batch_sampler=UniformWithReplacementSampler(
            num_samples=len(X_train),
            sample_rate=_sample_rate,
            generator=generator),
        )
    test_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(torch.Tensor(X_test), torch.LongTensor(y_test)),
        batch_size=batch_size_test,
        shuffle=True
        )
    train_loader_orig = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(torch.Tensor(X_train_orig), torch.LongTensor(y_train_orig)),
        batch_sampler=UniformWithReplacementSampler(
            num_samples=len(X_train_orig),
            sample_rate=_sample_rate,
            generator=generator),
        )
    test_loader_orig = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(torch.Tensor(X_test_orig), torch.LongTensor(y_test_orig)),
        batch_size=batch_size_test,
        shuffle=True
        )

    # create numpy arrays from dataloaders to train MI models
    X_train = np.empty(shape=(0,3,32,32))
    y_train = np.empty(shape=(0))
    for images, target in train_loader:
        X_train = np.append(X_train, images, axis=0)
        y_train = np.append(y_train, target, axis=0)
    
    X_test = np.empty(shape=(0,3,32,32))
    y_test = np.empty(shape=(0))
    for images, target in test_loader:
        X_test = np.append(X_test, images, axis=0)
        y_test = np.append(y_test, target, axis=0)

    # create numpy arrays from dataloaders to train MI models
    X_train_orig = np.empty(shape=(0,3,32,32))
    y_train_orig = np.empty(shape=(0))
    for images, target in train_loader_orig:
        X_train_orig = np.append(X_train_orig, images, axis=0)
        y_train_orig = np.append(y_train_orig, target, axis=0)
    
    X_test_orig = np.empty(shape=(0,3,32,32))
    y_test_orig = np.empty(shape=(0))
    for images, target in test_loader_orig:
        X_test_orig = np.append(X_test_orig, images, axis=0)
        y_test_orig = np.append(y_test_orig, target, axis=0)

    print('resampled train counts: ', torch.bincount(torch.IntTensor(y_train).squeeze()))
    print('resampled test counts: ', torch.bincount(torch.IntTensor(y_test).squeeze()))
    print('training set size: ', X_train.shape)
    print('test set size: ', X_test.shape)

    clipping = {"clip_per_layer": False, "enable_stat": True}
    best_acc1 = 0
        
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
            # noise_multiplier=args.noise_multiplier,
            max_grad_norm=args.max_grad_norm,
            target_epsilon=args.target_epsilon,
            target_delta=args.delta,
            epochs=args.epochs,
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
            top1_acc = test(model, test_loader_orig, device)

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

    epsilon = -1
    if not args.disable_dp:
        epsilon, best_alpha = optimizer.privacy_engine.get_privacy_spent(args.delta)

    art_classifier = PyTorchClassifier(model, loss=nn.CrossEntropyLoss(),
                                        optimizer=optimizer,
                                        input_shape=(3,32,32),
                                        nb_classes=10,)
    
    pred = np.array([np.argmax(arr) for arr in art_classifier.predict(torch.from_numpy(X_test_orig).type(torch.FloatTensor))])
    acc = accuracy_score(y_test_orig, pred)
    class_report = classification_report(y_test_orig, pred)
    print('Base private model test accuracy: ', acc)
    print(class_report)

    pred = np.array([np.argmax(arr) for arr in art_classifier.predict(torch.from_numpy(X_train).type(torch.FloatTensor))])
    train_acc = accuracy_score(y_train, pred)
    train_class_report = classification_report(y_train, pred)
    print('Base private model train accuracy: ', train_acc)
    print(train_class_report)


    # Black box attack
    attack_train_ratio = 0.5
    attack_train_size = int(len(X_train) * attack_train_ratio)
    attack_test_size = int(len(X_test) * attack_train_ratio)

    mlp_attack_bb = MembershipInferenceBlackBox(art_classifier, attack_model_type=args.attack_model)

    # get min and maj indices from testing set
    min_train_indices = np.where(np.isin(y_train[:attack_test_size], [0,1,2,3,4]))[0]
    maj_train_indices = np.where(np.isin(y_train[:attack_test_size], [5,6,7,8,9]))[0]
    min_test_indices = np.where(np.isin(y_test[:attack_test_size], [0,1,2,3,4]))[0]
    maj_test_indices = np.where(np.isin(y_test[:attack_test_size], [5,6,7,8,9]))[0]
    
    # get one of each class from min adn majority groups
    min_train_subset = []
    min_test_subset = []
    maj_train_subset = []
    maj_test_subset = []
    for i in [0,1,2,3,4]:
        min_train_subset.append(np.where(y_train == i)[0][0])
        min_test_subset.append(np.where(y_test == i)[0][0])
    for i in [5,6,7,8,9]:
        maj_train_subset.append(np.where(y_train == i)[0][0])
        maj_test_subset.append(np.where(y_test == i)[0][0])

    print(set(y_train[min_train_indices]))
    print(set(y_train[maj_train_indices]))
    print(set(y_test[min_test_indices]))
    print(set(y_test[maj_test_indices]))

    # add some of each class so that we have all class
    min_train_indices = np.append(min_train_indices, maj_train_subset, axis=0)
    maj_train_indices = np.append(maj_train_indices, min_train_subset, axis=0)
    min_test_indices = np.append(min_test_indices, maj_test_subset, axis=0)
    maj_test_indices = np.append(maj_test_indices, min_test_subset, axis=0)

    print(set(y_train[min_train_indices]))
    print(set(y_train[maj_train_indices]))
    print(set(y_test[min_test_indices]))
    print(set(y_test[maj_test_indices]))

    # train attack model
    mlp_attack_bb.fit(torch.from_numpy(X_train[attack_train_size:]).type(torch.FloatTensor), torch.from_numpy(y_train[attack_train_size:]).type(torch.FloatTensor),
                torch.from_numpy(X_test[attack_test_size:]).type(torch.FloatTensor), torch.from_numpy(y_test[attack_test_size:]).type(torch.FloatTensor))

    # infer 
    mlp_inferred_train_bb = mlp_attack_bb.infer(torch.from_numpy(X_train[:attack_train_size]).type(torch.FloatTensor), torch.from_numpy(y_train[:attack_train_size]).type(torch.FloatTensor))
    mlp_inferred_test_bb = mlp_attack_bb.infer(torch.from_numpy(X_test[:attack_test_size]).type(torch.FloatTensor), torch.from_numpy(y_test[:attack_test_size]).type(torch.FloatTensor))
    min_train_inferred = mlp_attack_bb.infer(torch.from_numpy(X_train[min_train_indices]).type(torch.FloatTensor), torch.from_numpy(y_train[min_train_indices]).type(torch.FloatTensor))
    maj_train_inferred = mlp_attack_bb.infer(torch.from_numpy(X_train[maj_train_indices]).type(torch.FloatTensor), torch.from_numpy(y_train[maj_train_indices]).type(torch.FloatTensor))
    min_test_inferred = mlp_attack_bb.infer(torch.from_numpy(X_test[min_test_indices]).type(torch.FloatTensor), torch.from_numpy(y_test[min_test_indices]).type(torch.FloatTensor))
    maj_test_inferred = mlp_attack_bb.infer(torch.from_numpy(X_test[maj_test_indices]).type(torch.FloatTensor), torch.from_numpy(y_test[maj_test_indices]).type(torch.FloatTensor))

    # check accuracy
    print("NN model attack results: ")
    mlp_train_acc_bb = np.sum(mlp_inferred_train_bb) / len(mlp_inferred_train_bb)
    mlp_test_acc_bb = 1 - (np.sum(mlp_inferred_test_bb) / len(mlp_inferred_test_bb))
    min_train_acc = np.sum(min_train_inferred) / len(min_train_inferred)
    maj_train_acc = np.sum(maj_train_inferred) / len(maj_train_inferred)
    min_test_acc = 1 - (np.sum(min_test_inferred) / len(min_test_inferred))
    maj_test_acc = 1 - (np.sum(maj_test_inferred) / len(maj_test_inferred))
    mlp_acc_bb = (mlp_train_acc_bb * len(mlp_inferred_train_bb) + mlp_test_acc_bb * len(mlp_inferred_test_bb)) / (len(mlp_inferred_train_bb) + len(mlp_inferred_test_bb))
    print('train acc: {}'.format(mlp_train_acc_bb))
    print('test acc: {}'.format(mlp_test_acc_bb))
    print('min train acc: {}'.format(min_train_acc))
    print('maj train acc: {}'.format(maj_train_acc))
    print('min test acc: {}'.format(min_test_acc))
    print('maj test acc: {}'.format(maj_test_acc))
    total_min_acc = (min_train_acc * len(min_train_inferred) + min_test_acc * len(min_test_inferred)) / (len(min_train_inferred) + len(min_test_inferred))
    total_maj_acc = (maj_train_acc * len(maj_train_inferred) + maj_test_acc * len(maj_test_inferred)) / (len(maj_train_inferred) + len(maj_test_inferred))
    print('total acc: {}'.format(mlp_acc_bb))
    print('total min acc: {}'.format(total_min_acc))
    print('total maj acc: {}'.format(total_maj_acc))
    mlp_prec_recall_bb = calc_precision_recall(np.concatenate((mlp_inferred_train_bb, mlp_inferred_test_bb)), 
                                np.concatenate((np.ones(len(mlp_inferred_train_bb)), np.zeros(len(mlp_inferred_test_bb)))))
    print('precision, recall: {}'.format(mlp_prec_recall_bb))
    rf_results = {
        'train acc': mlp_train_acc_bb,
        'test acc': mlp_test_acc_bb,
        'total acc': mlp_acc_bb,
        'prec, recall': mlp_prec_recall_bb
    }

    print('sampling type, sampling ratio, epsilon, acc, train acc, mia, min mia, maj mia, report, train report')
    results = [args.sampling_type, args.sampling_ratio, epsilon, acc, train_acc, mlp_acc_bb, total_min_acc, total_maj_acc, class_report, train_class_report]

    results_json = {
        'experiment_args': vars(args),
        'experiment_time': time.time() - start,
        'epsilon': epsilon,
        'accuracy': acc,
        'train acc': train_acc,
        'rf_acc': rf_results,
        'class report': class_report,
        'train class report': train_class_report,
        'results': results,
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