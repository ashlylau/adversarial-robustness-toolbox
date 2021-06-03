import sys
import os
import numpy as np
import argparse
import time
import json
from pathlib import Path

import random
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import roc_auc_score, accuracy_score
from opacus import PrivacyEngine
import torch.nn.functional as F
from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.datasets import make_imbalance

sys.path.append(os.path.abspath('../'))
from art.estimators.classification import PyTorchClassifier
from art.attacks.inference.membership_inference import MembershipInferenceBlackBoxRuleBased, MembershipInferenceBlackBox


# Define model
class Net(nn.Module):
    def __init__(self, num_features, num_classes):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(num_features, 32).to(torch.float64)
        self.fc2 = nn.Linear(32, num_classes).to(torch.float64)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def train(model, train_loader, optimizer, epoch):
    model.train()
    
    for inputs, target in train_loader:
        inputs, target = inputs.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(inputs)
        loss = loss_fn(output, target)
        # Backprop
        loss.backward()
        optimizer.step()
        ###

def test(model, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    test_size = 0
    preds = []
    targets = []
    
    with torch.no_grad():
        for inputs, target in test_loader:
            inputs, target = inputs.to(device), target.to(device)
            
            output = model(inputs)
            test_size += len(inputs)
            test_loss += test_loss_fn(output, target).item() 
            pred = output.max(1, keepdim=True)[1]
            pred_list = pred.tolist()
            target_list = target.tolist()
            preds.extend(pred_list)
            targets.extend(target_list)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= test_size
    accuracy = correct / test_size
    roc_auc = roc_auc_score(targets, preds)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.4f}%, ROC AUC: {:.4f}%)\n'.format(
        test_loss, correct, test_size,
        100. * accuracy,
        100. * roc_auc))
    
    return test_loss, accuracy

# Perform over and under sampling to maintain dataset size
def resample_equal(X, y, sampling_ratio):
    minority_indices = np.where(y == 1)[0]
    minority_X, minority_y = np.take(X, minority_indices, axis=0), np.take(y, minority_indices, axis=0)
    majority_X, majority_y = np.delete(X, minority_indices, axis=0), np.delete(y, minority_indices, axis=0)
    
    # calculate amount of each class to swap
    amt_to_swap = int((len(majority_X) * sampling_ratio - len(minority_X))/(1 + sampling_ratio))

    # undersample majority class
    undersample_indices = random.choices(np.arange(len(majority_X)), k=amt_to_swap)
    under_X, under_y = np.delete(majority_X, undersample_indices, axis=0), np.delete(majority_y, undersample_indices, axis=0)
    
    # replace with oversampled minority class
    oversample_indices = random.choices(np.arange(len(minority_X)), k=amt_to_swap)
    over_X = np.concatenate((under_X, minority_X, np.take(minority_X, oversample_indices, axis=0)), axis=0)
    over_y = np.concatenate((under_y, minority_y, np.take(minority_y, oversample_indices, axis=0)), axis=0)

    return over_X, over_y

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
    parser.add_argument('--dataset', default='adult', help='dataset to test')
    parser.add_argument('--batch_size', type=int, default=128, help='batch size')
    parser.add_argument('--epochs', type=int, default=20, help='num epochs')
    parser.add_argument('--noise_multiplier', type=float, default=1.3, help='noise multiplier')
    parser.add_argument('--max_grad_norm', type=float, default=10.0, help='max grad norm')
    parser.add_argument('--lr', type=float, default=0.00025, help='learning rate')
    parser.add_argument('--delta', type=float, default=0.00002, help='delta')
    parser.add_argument('--disable_dp', action='store_true', default=False, help='train non-private model')
    parser.add_argument('--sampling_type', default='none', help='over, under or smote sampling')
    parser.add_argument('--attack_model', default='rf', help='attack model type -- rf, nn')
    parser.add_argument('--sampling_ratio', type=float, default=0.5, help='sampling ratio')
    args = parser.parse_args()
    print(vars(args))

    device = torch.device('cuda')
    start = time.time()
    
    # Load data
    X_train = np.loadtxt("https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data",
                            usecols=(0, 4, 10, 11, 12), delimiter=", ")
    y_train = np.loadtxt("https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data",
                            usecols=14, dtype=str, delimiter=", ")
    X_test = np.loadtxt("https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test",
                            usecols=(0, 4, 10, 11, 12), delimiter=", ", skiprows=1)
    y_test = np.loadtxt("https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test",
                            usecols=14, dtype=str, delimiter=", ", skiprows=1)
    # Must trim trailing period "." from label
    y_test = np.array([a[:-1] for a in y_test])
    # Convert to ints for tensor loading
    y_train, y_test = np.where(y_train == '>50K', 1, 0), np.where(y_test == '>50K', 1, 0)
    min_train_indices = np.where(y_train == 1)[0]
    maj_train_indices = np.where(y_train == 0)[0]
    min_test_indices = np.where(y_test == 1)[0]
    maj_test_indices = np.where(y_test == 0)[0]
    
    X_test_attack, y_test_attack = X_test, y_test

    # Perform resampling
    print("X_train: {}, y_train: {}, X_test: {}, y_test: {}".format(X_train.shape, y_train.shape, X_test.shape, y_test.shape))
    if args.sampling_type == 'over':
        print("Performing oversampling at {}".format(args.sampling_ratio))
        X_train, y_train = RandomOverSampler(sampling_strategy=args.sampling_ratio).fit_resample(X_train, y_train)
        X_test_attack, y_test_attack = RandomOverSampler(sampling_strategy=args.sampling_ratio).fit_resample(X_test_attack, y_test_attack)
    elif args.sampling_type == 'under':
        print("Performing undersampling at {}".format(args.sampling_ratio))
        X_train, y_train = RandomUnderSampler(sampling_strategy=args.sampling_ratio).fit_resample(X_train, y_train)
        X_test_attack, y_test_attack = RandomUnderSampler(sampling_strategy=args.sampling_ratio).fit_resample(X_test_attack, y_test_attack)
    elif args.sampling_type == 'smote':
        print("Performing SMOTE oversampling at {}".format(args.sampling_ratio))
        X_train, y_train = SMOTE(sampling_strategy=args.sampling_ratio).fit_resample(X_train, y_train)
        X_test_attack, y_test_attack = SMOTE(sampling_strategy=args.sampling_ratio).fit_resample(X_test_attack, y_test_attack)
    elif args.sampling_type == 'equal':
        print("Performing over and under sampling at {}".format(args.sampling_ratio))
        X_train, y_train = resample_equal(X_train, y_train, args.sampling_ratio)
        X_test_attack, y_test_attack = resample_equal(X_test_attack, y_test_attack, args.sampling_ratio)
    print("X_train: {}, y_train: {}, X_test: {}, y_test: {}".format(X_train.shape, y_train.shape, X_test.shape, y_test.shape))

    # Convert to tensors.
    train_inputs = torch.from_numpy(X_train).to(torch.float64)
    train_targets = torch.from_numpy(y_train)
    test_inputs = torch.from_numpy(X_test).to(torch.float64)
    test_targets = torch.from_numpy(y_test)
    test_attack_inputs = torch.from_numpy(X_test_attack).to(torch.float64)
    test_attack_targets = torch.from_numpy(y_test_attack)

    train_ds = TensorDataset(train_inputs, train_targets)
    val_ds = TensorDataset(test_inputs, test_targets)

    input_size = X_train.shape[1]
    num_classes = 2

    train_loader = DataLoader(train_ds, args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, args.batch_size, pin_memory=True)

    # Train model.
    model = Net(input_size, num_classes).to(device)
    test_accuracy = []
    train_loss = []
    weight_decay = 0
    epsilon = -1
    best_alpha = -1
    accuracy = -1
    roc_auc = -1

    # Surrogate loss used for training
    loss_fn = nn.CrossEntropyLoss()
    test_loss_fn = nn.CrossEntropyLoss(reduction='sum')

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=weight_decay)
    if not args.disable_dp:
        privacy_engine = PrivacyEngine(
            model,
            batch_size=args.batch_size,
            sample_size=len(X_train),
            alphas=[1 + x / 10.0 for x in range(1, 100)] + list(range(12, 64)),
            noise_multiplier=args.noise_multiplier,
            max_grad_norm=args.max_grad_norm
        )
        privacy_engine.attach(optimizer)

    print('Training beginning...')
    for epoch in range(1, args.epochs + 1):
        print('Epoch ', epoch, ':')
        train(model, train_loader, optimizer, epoch)
        loss, acc = test(model, val_loader)
        
        # save results every epoch
        test_accuracy.append(acc)
        train_loss.append(loss)

        if not args.disable_dp:
            epsilon, best_alpha = optimizer.privacy_engine.get_privacy_spent(args.delta)
            print(
                f"\tTrain Epoch: {epoch} \t"
                f"Loss: {np.mean(train_loss):.6f} "
                f"Acc@1: {np.mean(test_accuracy):.6f} "
                f"(ε = {epsilon:.2f}, δ = {args.delta}) for α = {best_alpha}"
            )

    mlp_art_model = PyTorchClassifier(model=model, loss=loss_fn, optimizer=optimizer, input_shape=(5,), nb_classes=2)
    pred = np.array([np.argmax(arr) for arr in mlp_art_model.predict(X_test)])
    accuracy = accuracy_score(y_test, pred)
    roc_auc = roc_auc_score(y_test, pred)
    print('Epsilon = {}, model accuracy: {}, roc auc: {}'.format(epsilon, accuracy, roc_auc))


    # Separately test minority and majority data points
    test_majority_indices = np.append(maj_test_indices, 7)
    test_minority_indices = np.append(min_test_indices, 1)
    majority_X_test, majority_y_test = np.take(X_test, test_majority_indices, axis=0), np.take(y_test, test_majority_indices, axis=0)
    minority_X_test, minority_y_test = np.take(X_test, test_minority_indices, axis=0), np.take(y_test, test_minority_indices, axis=0)
    
    majority_test_inputs = torch.from_numpy(minority_X_test).to(torch.float64)
    majority_test_targets = torch.from_numpy(minority_y_test)
    minority_test_inputs = torch.from_numpy(minority_X_test).to(torch.float64)
    minority_test_targets = torch.from_numpy(minority_y_test)

    # Black box attack
    attack_train_ratio = 0.5
    attack_train_size = int(len(X_train) * attack_train_ratio)
    attack_train_indices = random.choices(np.arange(len(X_train)), k=attack_train_size)
    attack_test_size = int(len(X_test_attack) * attack_train_ratio)
    attack_test_indices = random.choices(np.arange(len(X_test_attack)), k=attack_test_size)

    mlp_attack = MembershipInferenceBlackBox(mlp_art_model, attack_model_type=args.attack_model)

    # train attack model
    mlp_attack.fit(train_inputs[attack_train_indices], train_targets[attack_train_indices],
                test_attack_inputs[attack_test_indices], test_attack_targets[attack_test_indices])

    # infer 
    inferred_test_majority = mlp_attack.infer(majority_test_inputs, majority_test_targets)
    inferred_test_minority = mlp_attack.infer(minority_test_inputs, minority_test_targets)
    inferred_train = mlp_attack.infer(train_inputs, train_targets)
    inferred_test = mlp_attack.infer(test_inputs, test_targets)

    # check accuracy
    print("Random forest model attack results: ")
    train_acc = np.sum(inferred_train) / len(inferred_train)
    test_acc = 1 - (np.sum(inferred_test) / len(inferred_test))
    test_acc_majority = np.sum(inferred_test_majority) / len(inferred_test_majority)
    test_acc_minority = 1 - (np.sum(inferred_test_minority) / len(inferred_test_minority))
    acc = (train_acc * len(inferred_train) + test_acc * len(inferred_test)) / (len(inferred_train) + len(inferred_test))
    print('train acc: {}'.format(train_acc))
    print('test acc: {}'.format(test_acc))
    print('test acc majority: {}'.format(test_acc_majority))
    print('test acc minority: {}'.format(test_acc_minority))
    print('total acc: {}'.format(acc))
    rf_results = {
        'train acc': train_acc,
        'test acc': test_acc,
        'majority test acc': test_acc_majority,
        'minority test acc': test_acc_minority,
        'total acc': acc,
    }

    results = [args.sampling_type, args.sampling_ratio, epsilon, accuracy, roc_auc, test_acc_minority, test_acc_majority, test_acc]

    # Compile results
    results_json = {
        'experiment_args': vars(args),
        'experiment_time': time.time() - start,
        'accuracy': accuracy,
        'roc_auc': roc_auc,
        'epsilon': epsilon,
        'best_alpha': best_alpha,
        'rf_acc': rf_results,
        'results': results
    }

    # Create experiment directory.
    experiment_path = f'/homes/al5217/adversarial-robustness-toolbox/examples/{args.dataset}/{args.sampling_type}/'
    experiment_number = len(os.listdir(experiment_path))
    print("experiment_number: {}".format(experiment_number))

    # Dump the results to file
    json_file = Path.cwd() / f'{args.dataset}/{args.sampling_type}/test_results-{experiment_number}.json'
    with json_file.open('w') as f:
        json.dump(results_json, f, indent="  ")

    print(results)
    print("dumped results to {}".format(str(json_file)))