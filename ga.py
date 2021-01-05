from __future__ import print_function
import matplotlib.pyplot as plt
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
from torch.utils.tensorboard import SummaryWriter
import torch.backends.cudnn as cudnn
import yaml
import numpy as np
import os
import glob
import logging
from pathlib import Path
import random
import time
import matplotlib

logger = logging.getLogger(__name__)


def hist2d(x, y, n=100):
    # 2d histogram used in labels.png and evolve.png
    xedges, yedges = np.linspace(
        x.min(), x.max(), n), np.linspace(y.min(), y.max(), n)
    hist, xedges, yedges = np.histogram2d(x, y, (xedges, yedges))
    xidx = np.clip(np.digitize(x, xedges) - 1, 0, hist.shape[0] - 1)
    yidx = np.clip(np.digitize(y, yedges) - 1, 0, hist.shape[1] - 1)
    return np.log(hist[xidx, yidx])


def plot_evolution(yaml_file='data/hyp.finetune.yaml'):
    # Plot hyperparameter evolution results in evolve.txt
    with open(yaml_file) as f:
        hyp = yaml.load(f, Loader=yaml.FullLoader)
    x = np.loadtxt('evolve.txt', ndmin=2)
    f = x[:, 0]
    # weights = (f - f.min()) ** 2  # for weighted results
    plt.figure(figsize=(10, 12), tight_layout=True)
    matplotlib.rc('font', **{'size': 8})
    for i, (k, v) in enumerate(hyp.items()):
        y = x[:, i + 1]
        # mu = (y * weights).sum() / weights.sum()  # best weighted result
        mu = y[f.argmax()]  # best single result
        plt.subplot(6, 5, i + 1)
        plt.scatter(y, f, c=hist2d(y, f, 20), cmap='viridis',
                    alpha=.8, edgecolors='none')
        plt.plot(mu, f.max(), 'k+', markersize=15)
        plt.title('%s = %.3g' % (k, mu), fontdict={
                  'size': 9})  # limit to 40 characters
        if i % 5 != 0:
            plt.yticks([])
        print('%15s: %.3g' % (k, mu))
    plt.savefig('evolve.png', dpi=200)
    print('\nPlot saved as evolve.png')


def init_torch_seeds(seed=0):
    torch.manual_seed(seed)

    # Speed-reproducibility tradeoff https://pytorch.org/docs/stable/notes/randomness.html
    if seed == 0:  # slower, more reproducible
        cudnn.deterministic = True
        cudnn.benchmark = False
    else:  # faster, less reproducible
        cudnn.deterministic = False
        cudnn.benchmark = True


def init_seeds(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    init_torch_seeds(seed)


def check_file(file):
    # Search for file if not found
    if os.path.isfile(file) or file == '':
        return file
    else:
        files = glob.glob('./**/' + file, recursive=True)  # find file
        assert len(files), 'File Not Found: %s' % file  # assert file was found
        assert len(files) == 1, "Multiple files match '%s', specify exact path: %s" % (
            file, files)  # assert unique
        return files[0]  # return file


def print_mutation(hyp, result, yaml_file='hyp_evolved.yaml'):
    # Print mutation results to evolve.txt (for use with train.py --evolve)
    a = '%10s' * len(hyp) % tuple(hyp.keys())  # hyperparam keys
    b = '%10.3g' * len(hyp) % tuple(hyp.values())  # hyperparam values
    # result accuracy (%)
    c = result
    print('\n%s\n%s\nEvolved fitness: %s\n' % (a, b, c))

    with open('./evolve.txt', 'a') as f:  # append result
        f.write(str(c) + b + '\n')
    x = np.unique(np.loadtxt('./evolve.txt', ndmin=2),
                  axis=0)  # load unique rows

    # sort in descending order on fitness (first column value)
    x = x[x[:, 0].argsort()[::-1]]
    np.savetxt('./evolve.txt', x, '%10.3g')  # save sort by fitness

    # Save yaml
    for i, k in enumerate(hyp.keys()):
        hyp[k] = float(x[0, i + 1])
    with open(yaml_file, 'w') as f:
        results = tuple(x[0, :1])
        # results (results, val_loss)
        c = '%10.4g' * len(results) % results
        f.write('# Hyperparameter Evolution Results\n# Generations: %g\n# Metrics: ' % len(
            x) + c + '\n\n')
        yaml.dump(hyp, f, sort_keys=False)


def get_config(config):
    with open(config, "r") as stream:
        return yaml.load(stream, Loader=yaml.FullLoader)


def write_results_to_tb(tb_writer, test_perf, train_perf, epoch):
    if tb_writer:
        x = [test_perf, train_perf]
        titles = ['Test MAE', 'Train MAE']
        for xi, title in zip(x, titles):
            tb_writer.add_scalar(title, xi, epoch)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output


def train(hyp, args, device, train_loader, test_loader, tb_writer=None):

    init_seeds()

    model = Net().to(device)
    optimizer = optim.Adam(
        model.parameters(),
        lr=hyp['lr'],
        betas=(hyp['momentum'], 0.999))
    scheduler = StepLR(optimizer, step_size=1, gamma=hyp['gamma'])

    log_dir = Path(tb_writer.log_dir) if tb_writer else Path(
        args.logdir) / 'evolve'  # logging directory
    results_file = str(log_dir / 'results.txt')
    wdir = log_dir / 'weights'  # weights directory
    os.makedirs(wdir, exist_ok=True)
    last = wdir / 'last.pt'
    best = wdir / 'best.pt'

    # Save run settings
    with open(log_dir / 'hyp.yaml', 'w+') as f:
        yaml.dump(hyp, f, sort_keys=False)
    with open(log_dir / 'opt.yaml', 'w+') as f:
        yaml.dump(vars(args), f, sort_keys=False)

    model.train()

    best_fitness = 0.0
    for epoch in range(1, args.epochs + 1):

        final_epoch = epoch + 1 == args.epochs

        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = F.nll_loss(output, target)
            loss.backward()
            optimizer.step()
            if batch_idx % args.log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tLR: {}'.format(
                    epoch,
                    batch_idx * len(data),
                    len(train_loader.dataset),
                    100. * batch_idx / len(train_loader),
                    loss.item(),
                    scheduler.get_last_lr()))
                if args.dry_run:
                    break
        fitness = test(model, device, test_loader)
        if fitness > best_fitness:
            best_fitness = fitness
        scheduler.step()

        if tb_writer:
            tags = ['train/loss', 'test/accuracy(%)']  # params
            for x, tag in zip([loss.item(), fitness], tags):
                tb_writer.add_scalar(tag, x, epoch)

        if args.save_model:
            with open(results_file, 'r') as f:  # create checkpoint
                ckpt = {'epoch': epoch,
                        'best_fitness': best_fitness,
                        'training_results': f.read(),
                        'model': model.state_dict(),
                        'optimizer': None if final_epoch else optimizer.state_dict()}
            torch.save(model.state_dict(), "mnist_cnn.pt")
            # Save last, best and delete
            torch.save(ckpt, last)
            if best_fitness == fitness:
                torch.save(ckpt, best)

    return fitness


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            # sum up batch loss
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            # get the index of the max log-probability
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

    accuracy = 100. * correct / len(test_loader.dataset)

    return accuracy


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=5, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=1.0, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--dry-run', action='store_true', default=False,
                        help='quickly check a single pass')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    parser.add_argument("-e", "--evolve", action='store_true', default=False,
                        help='evolve hyperparameters')
    parser.add_argument('--logdir', type=str,
                        default='runs/', help='logging directory')

    # hyperparameters file load and start training with
    parser.add_argument(
        '--hyp', type=str, default='hyp.scratch.yaml', help='hyperparameters path')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    args.hyp = check_file(args.hyp)

    train_kwargs = {'batch_size': args.batch_size}
    test_kwargs = {'batch_size': args.test_batch_size}

    if use_cuda:
        cuda_kwargs = {'num_workers': 6,
                       'pin_memory': True,
                       'shuffle': True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    dataset1 = datasets.MNIST('./data', train=True, download=True,
                              transform=transform)
    dataset2 = datasets.MNIST('./data', train=False,
                              transform=transform)

    train_loader = torch.utils.data.DataLoader(dataset1, **train_kwargs)
    test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)

    # For TensorBoard run:
    #   tensorboard --logdir=runs
    #   open: http://localhost:6006/
    tb_writer = SummaryWriter()

    with open(args.hyp) as f:
        hyp = yaml.load(f, Loader=yaml.FullLoader)  # load hyps

    # Standard train
    if not args.evolve:
        result = train(hyp, args, device, train_loader,
                       test_loader)

    # Evolve hyperparameters (optional)
    else:
        # Hyperparameter evolution metadata (mutation scale 0-1, lower_limit, upper_limit)
        meta = {'weight_decay': (1, 1e-5, 1e-1),  # Weight decay
                'beta1': (1, 0, 1),  # Adam parameter
                'beta2': (1, 0, 1),  # Adam parameter
                'lr': (1, 1e-5, 1e-1),  # Initial learning rate
                'gamma': (1, 0, 1),  # How much to decay learning rate
                'momentum': (0.3, 0.6, 0.98)}  # RMSprop parameter

        yaml_file = Path(args.logdir) / 'evolve' / \
            'hyp_evolved.yaml'  # save best result here

        for _ in range(20):  # generations to evolve
            # if evolve.txt exists: select best hyps and mutate
            if os.path.exists('evolve.txt'):
                # Select parent(s)
                parent = 'single'  # parent selection method: 'single' or 'weighted'
                x = np.loadtxt('evolve.txt', ndmin=2)
                n = min(5, len(x))  # number of previous results to consider
                # sort lines in hyperparam file based on first value (accuracy) then pick the
                # top n mutations
                x = x[x[:, 0].argsort()[::-1]][:n]
                print(x)
                w = x[:, 0] - x[:, 0].min()  # weights
                if parent == 'single' or len(x) == 1:
                    # x = x[random.randint(0, n - 1)]  # random selection
                    x = x[random.choices(range(n), weights=w)[
                        0]]  # weighted selection
                elif parent == 'weighted':
                    x = (x * w.reshape(n, 1)).sum(0) / \
                        w.sum()  # weighted combination

                # Mutate
                mp, s = 0.8, 0.2  # mutation probability, sigma
                npr = np.random
                npr.seed(int(time.time()))
                g = np.array([x[0] for x in meta.values()])  # gains 0-1
                ng = len(meta)
                v = np.ones(ng)
                while all(v == 1):  # mutate until a change occurs (prevent duplicates)
                    v = (g * (npr.random(ng) < mp) * npr.randn(ng) *
                         npr.random() * s + 1).clip(0.3, 3.0)
                for i, k in enumerate(hyp.keys()):  # plt.hist(v.ravel(), 300)
                    # mutate
                    hyp[k] = float(x[i + 1] * v[i])

            for k, v in meta.items():
                hyp[k] = max(hyp[k], v[1])  # lower limit
                hyp[k] = min(hyp[k], v[2])  # upper limit
                hyp[k] = round(hyp[k], 5)  # significant digits

            # Train mutation
            result = train(hyp, args, device, train_loader,
                           test_loader, tb_writer)

            # Write mutation results
            print_mutation(hyp.copy(), result, yaml_file)

        # Plot results
        plot_evolution(yaml_file)
        print(f'Hyperparameter evolution complete. Best results saved as: {yaml_file}\n'
              f'Command to train a new model with these hyperparameters: $ python train.py --hyp {yaml_file}')


if __name__ == '__main__':
    main()

