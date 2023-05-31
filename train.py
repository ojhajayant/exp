import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import argparse
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torchsummary import summary
from tqdm import tqdm
import numpy as np
import traceback

dropout_value = 0.029


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # Input Block
        self.convblock1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=10, kernel_size=(3, 3),
                      padding=0, bias=False),
            nn.BatchNorm2d(10),
            nn.Dropout(dropout_value),
            nn.ReLU()
        )  # input_size = 28 output_size = 26 receptive_field = 3

        # CONVOLUTION BLOCK 1
        self.convblock2 = nn.Sequential(
            nn.Conv2d(in_channels=10, out_channels=10, kernel_size=(3, 3),
                      padding=0, bias=False),
            nn.BatchNorm2d(10),
            nn.Dropout(dropout_value),
            nn.ReLU()
        )  # input_size = 26 output_size = 24 receptive_field = 5
        self.convblock3 = nn.Sequential(
            nn.Conv2d(in_channels=10, out_channels=15, kernel_size=(3, 3),
                      padding=0, bias=False),
            nn.BatchNorm2d(15),
            nn.Dropout(dropout_value),
            nn.ReLU()
        )  # input_size = 24 output_size = 22 receptive_field = 7

        # TRANSITION BLOCK 1
        self.pool1 = nn.MaxPool2d(2,
                                  2)  # input_size = 22 output_size = 11
        # receptive_field = 8
        self.convblock4 = nn.Sequential(
            nn.Conv2d(in_channels=15, out_channels=10, kernel_size=(1, 1),
                      padding=0, bias=False),
            nn.BatchNorm2d(10),
            nn.Dropout(dropout_value),
            nn.ReLU()
        )  # input_size = 11 output_size = 10 receptive_field = 8

        # CONVOLUTION BLOCK 2
        self.convblock5 = nn.Sequential(
            nn.Conv2d(in_channels=10, out_channels=10, kernel_size=(3, 3),
                      padding=0, bias=False),
            nn.BatchNorm2d(10),
            nn.Dropout(dropout_value),
            nn.ReLU()
        )  # input_size = 11 output_size = 9 receptive_field = 12
        self.convblock6 = nn.Sequential(
            nn.Conv2d(in_channels=10, out_channels=10, kernel_size=(3, 3),
                      padding=0, bias=False),
            nn.BatchNorm2d(10),
            nn.Dropout(dropout_value),
            nn.ReLU()
        )  # input_size = 9 output_size = 7 receptive_field = 16
        self.convblock7 = nn.Sequential(
            nn.Conv2d(in_channels=10, out_channels=32, kernel_size=(3, 3),
                      padding=0, bias=False),
            nn.BatchNorm2d(32),
            nn.Dropout(dropout_value),
            nn.ReLU()
        )  # input_size = 7 output_size = 5 receptive_field = 20
        # OUTPUT BLOCK
        self.gap = nn.Sequential(
            nn.AvgPool2d(kernel_size=5)
        )  # input_size = 5 output_size = 1 receptive_field = 28

        self.convblock8 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=10, kernel_size=(1, 1),
                      padding=0, bias=False),
            # No BatchNorm/DropOut/ReLU
        )  # input_size = 1 output_size = 1 receptive_field = 28

    def forward(self, x):
        # INPUT BLOCK LAYER
        x = self.convblock1(x)

        # CONVOLUTION BLOCK 1
        x = self.convblock2(x)
        x = self.convblock3(x)

        # TRANSITION BLOCK 1
        x = self.pool1(x)
        x = self.convblock4(x)

        # CONVOLUTION BLOCK 2
        x = self.convblock5(x)
        x = self.convblock6(x)
        x = self.convblock7(x)

        # OUTPUT BLOCK
        x = self.gap(x)
        x = self.convblock8(x)

        x = x.view(-1, 10)
        return F.log_softmax(x, dim=-1)


def get_args():
    parser = argparse.ArgumentParser(
        description='MNIST Training Script')
    parser.add_argument("--resume", action="store_true", default=False,
                        help="Resume training from checkpoint")
    parser.add_argument('--cmd', default='train',
                        choices=['train', 'test', 'lr_find'])
    parser.add_argument('--SEED', '-S', default=1, type=int, help='Random Seed')
    parser.add_argument('--dataset', '-D', default='MNIST', type=str,
                        help='Dataset--MNIST, or...')
    parser.add_argument('--img_size', '-I', default=(28, 28), type=tuple,
                        help='Image Size')
    parser.add_argument('--batch_size', '-b', default=512, type=int,
                        help='batch size')
    parser.add_argument('--epochs', '-e', default=15, type=int,
                        help='training epochs')
    parser.add_argument('--criterion', default=nn.NLLLoss(),
                        type=nn.modules.loss._Loss,
                        help='The loss function to be used during training')
    parser.add_argument('--optimizer', default=optim.SGD, type=type(optim.SGD),
                        help='The optimizer to be used during training')
    parser.add_argument('--cuda', default=torch.cuda.is_available(), type=bool,
                        help='use gpu or not')
    parser.add_argument('--init_lr', default=1e-4, type=float,
                        help='lr lower range value used for the LR-range-test')
    parser.add_argument('--end_lr', default=1, type=float,
                        help='lr upper range value used for the LR-range-test')
    parser.add_argument('--max_lr_epochs', '-M', default=5, type=int,
                        help='at what epoch Max LR should reach?')
    parser.add_argument('--lr_range_test_epochs', '-E', default=10, type=int,
                        help='epoch value used for the LR-range-test')
    parser.add_argument('--best_lr', default=0.03, type=float,
                        help='best_lr obtained from the LR-range-test')
    parser.add_argument('--cycle_momentum', default=True, type=bool,
                        help='Make cyclic changes to momentum value during OCP?')
    parser.add_argument('--div_factor', '-f', default=10, type=int,
                        help='OCP div factor')
    parser.add_argument('--l1_weight', default=0.000025, type=float,
                        help='L1-penalty value')
    parser.add_argument('--l2_weight_decay', default=0.0002125, type=float,
                        help='L2-penalty/weight_decay value')
    parser.add_argument('--L1', default=True, type=bool,
                        help='L1-penalty to be used or not?')
    parser.add_argument('--L2', default=False, type=bool,
                        help='L2-penalty/weight_decay to be used or not?')
    parser.add_argument('--data', '-s', default='./data/',
                        help='path to save train/test data')
    parser.add_argument('--best_model_path', default='./saved_models/',
                        help='best model saved path')
    parser.add_argument('--prefix', '-p', default='data', type=str,
                        help='folder prefix')
    parser.add_argument('--best_model', '-m',
                        default=' ',
                        type=str, help='name of best-accuracy model saved')
    args = parser.parse_args()
    return args


def get_dataset_mean_std():
    """
    Get the MNIST dataset mean and std to be used as tuples
    @ transforms.Normalize
    """
    # load the training data
    dataset_train = datasets.MNIST('./data', train=True, download=True)
    # use np.concatenate to stick all the images together to form a 1600000 X
    # 32 X 3 array
    x = np.concatenate([np.asarray(dataset_train[i][0]) for i in
                        range(len(dataset_train))])
    # print(x)
    # print(x.shape)
    # calculate the mean and std
    train_mean = np.mean(x, axis=(0, 1)) / 255
    train_std = np.std(x, axis=(0, 1)) / 255
    # the mean and std
    print(
        "\nThe mean & std-dev tuples for the {} dataset:".format(args.dataset))
    print(train_mean, train_std)
    class_names = dataset_train.classes
    return train_mean, train_std, class_names


def preprocess_data(mean_tuple, std_tuple):
    """
    Used for pre-processing the data
    """
    # Train Phase transformations
    global args
    file_path = args.data
    train_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.RandomRotation(degrees=7),
        transforms.Normalize(mean_tuple, std_tuple),
    ])

    # Test Phase transformations
    test_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean_tuple, std_tuple)
    ])
    train_dataset = datasets.MNIST(file_path, train=True, download=True,
                                   transform=train_transforms)
    test_dataset = datasets.MNIST(file_path, train=False, download=True,
                                  transform=test_transforms)

    print("CUDA Available?", args.cuda)

    # For reproducibility
    torch.manual_seed(args.SEED)

    if args.cuda:
        torch.cuda.manual_seed(args.SEED)
    args.batch_size = 64
    # dataloader arguments - something you'll fetch these from cmdprmt
    dataloader_args = dict(shuffle=True, batch_size=args.batch_size,
                           num_workers=4,
                           pin_memory=True) if args.cuda else \
        dict(shuffle=True, batch_size=args.batch_size)
    # train dataloader
    train_loader = torch.utils.data.DataLoader(train_dataset, **dataloader_args)
    # test dataloader
    test_loader = torch.utils.data.DataLoader(test_dataset, **dataloader_args)
    return train_dataset, test_dataset, train_loader, test_loader


def l1_penalty(x):
    """
    L1 regularization adds an L1 penalty equal
    to the absolute value of the magnitude of coefficients
    """
    global args

    return torch.abs(x).sum()


def train_epoch(model, device, train_loader, optimizer, epoch, criterion,
                scheduler=None, L1=False):
    """
    main training code
    """
    global args
    model.train()
    pbar = tqdm(train_loader, position=0)
    correct = 0
    processed = 0
    for batch_idx, (data, target) in enumerate(pbar):
        # get samples
        data, target = data.to(device), target.to(device)
        # Init
        optimizer.zero_grad()
        # In PyTorch, we need to set the gradients to zero before starting to
        # do back-propagation because PyTorch accumulates the gradients on
        # subsequent backward passes. Because of this, when you start your
        # training loop, ideally you should zero out the gradients so that
        # you do the parameter update correctly.

        # Predict
        y_pred = model(data)
        if L1:
            to_reg = []
            for param in model.parameters():
                to_reg.append(param.view(-1))
            l1 = args.l1_weight * l1_penalty(torch.cat(to_reg))
        else:
            l1 = 0
        # Calculate loss
        # L1 regularization adds an L1 penalty equal to the
        # absolute value of the magnitude of coefficients
        # torch.nn.CrossEntropyLoss:criterion combines
        # nn.LogSoftmax() and nn.NLLLoss() in one single class.
        loss = criterion(y_pred, target) + l1
        # Backpropagation
        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        # Capture the momentum and learning rate values
        # if args.optimizer == optim.Adam or args.optimizer == optim.AdamW:
        #     if optimizer.param_groups[0]['momentum'] is not None:
        #         cfg.momentum_values.append(optimizer.param_groups[0]['momentum'])
        # Update pbar-tqdm
        pred = y_pred.argmax(dim=1, keepdim=True)  # get the index of the max
        # log-probability
        correct += pred.eq(target.view_as(pred)).sum().item()
        processed += len(data)
        pbar.set_description(
            desc=f'Loss={loss.item()} Batch_id={batch_idx} Accuracy={100 * correct / processed:0.2f}',
            refresh=False)


def save_checkpoint(state, filename='model_checkpoint.pth'):
    """
    Save the model to the path
    """
    global args
    torch.save(state, filename)


def test_epoch(model, device, test_loader, optimizer, epoch, criterion):
    """
    main test code
    """
    # global current_best_acc, last_best_acc
    global args
    model.eval()
    test_loss = 0
    correct = 0
    acc1 = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()  # sum up batch loss
            pred = output.argmax(dim=1,
                                 keepdim=True)  # get the index of the max
            # log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print(
        '\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
            test_loss, correct, len(test_loader.dataset),
            100. * correct / len(test_loader.dataset)))
    acc1 = 100. * correct / len(test_loader.dataset)
    # Prepare model saving directory.
    save_dir = os.path.join(os.getcwd(), args.best_model_path)
    model_name = 'model_checkpoint.pth'
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    filepath = os.path.join(save_dir, model_name)
    save_checkpoint({
        'epoch': epoch + 1,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
    }, filename=filepath)


def main():
    global args
    args = get_args()
    resume = args.resume
    SEED = args.SEED
    cuda = args.cuda
    print("CUDA Available?", cuda)

    # For reproducibility
    torch.manual_seed(SEED)
    if cuda:
        torch.cuda.manual_seed(SEED)

    # Print the default and/or user supplied arg values, if any.
    print("\n\tHere are the different args values for this run:")
    for arg in vars(args):
        print("\t" + arg, ":", getattr(args, arg))

    # Create the required folder paths.
    best_model_path = args.best_model_path
    if not os.path.exists(best_model_path):
        os.makedirs(best_model_path)
    data_path = args.data
    if not os.path.exists(data_path):
        os.makedirs(data_path)

    # Get the image size tuple
    img_size = args.img_size

    # Calculate mean & std for Normalization
    mean, std, class_names = get_dataset_mean_std()
    mean_tuple = (mean,)
    std_tuple = (std,)

    # Dataloader Arguments & Test/Train Dataloaders
    train_dataset, test_dataset, train_loader, test_loader = \
        preprocess_data(mean_tuple, std_tuple)

    # Using L1-regularization here (l1_weight = 0.000025, reusing some older
    # assignment values, but works OK here too)
    L1 = args.L1
    L2 = args.L2

    if L2:
        weight_decay = args.l2_weight_decay
    else:
        weight_decay = 0

    criterion = args.criterion  # nn.CrossEntropyLoss()

    optimizer = args.optimizer

    device = torch.device("cuda" if args.cuda else "cpu")
    print(device)

    # Get the model loaded with summary(10 classes)
    model = Net().to(device)
    summary(model, input_size=(1, 28, 28))

    best_lr = args.best_lr
    # Setup optimizer & scheduler parameters for OCP
    CYCLE_MOMENTUM = args.cycle_momentum  # If True, momentum value cycles
    # from base_momentum of 0.85 to max_momentum of 0.95 during OCP cycle
    MOMENTUM = 0.9
    WEIGHT_DECAY = weight_decay
    DIV_FACTOR = args.div_factor  # default 10
    # final_div_factor = div_factor for NO annihilation
    FINAL_DIV_FACTOR = DIV_FACTOR
    EPOCHS = args.epochs  # 24 here
    MAX_LR_EPOCHS = args.max_lr_epochs  # 5 here
    NUM_OF_BATCHES = len(train_loader)
    PCT_START = MAX_LR_EPOCHS / EPOCHS
    # Based on above found maximum LR, initialize LRMAX and LRMIN
    LRMAX = best_lr
    LRMIN = LRMAX / DIV_FACTOR

    # Initialize optimizer and scheduler parameters
    optim_params = {"lr": LRMIN,
                    "momentum": MOMENTUM,
                    "weight_decay": WEIGHT_DECAY}
    scheduler_params = {"max_lr": LRMAX,
                        "steps_per_epoch": NUM_OF_BATCHES,
                        "epochs": EPOCHS,
                        "pct_start": PCT_START,
                        "anneal_strategy": "linear",
                        "div_factor": DIV_FACTOR,
                        "final_div_factor": FINAL_DIV_FACTOR,
                        "cycle_momentum": CYCLE_MOMENTUM}
    optimizer = optim.SGD(model.parameters(), **optim_params)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer,
                                                    **scheduler_params)

    if resume:
        # Load the pth checkpoint and start training forward
        print("Resuming from checkpoint...")
        save_dir = os.path.join(os.getcwd(), args.best_model_path)
        model_name = 'model_checkpoint.pth'
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)
        filepath = os.path.join(save_dir, model_name)
        if os.path.isfile(filepath):
            print("=> Loading checkpoint from '{}'".format(filepath))
            try:
                checkpoint = torch.load(filepath)
                model.load_state_dict(checkpoint['state_dict'])
                optimizer.load_state_dict(checkpoint['optimizer'])
                print("=> Resuming training from checkpoint")
            except Exception as e:
                print("Error loading checkpoint:")
                traceback.print_exc()
                # Handle the error or exit gracefully
            scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer,
                                                            **scheduler_params)
    else:
        # Normal training
        print("Starting normal training...")
        print("Model training starts on {} dataset".format(args.dataset))

    for epoch in range(EPOCHS):
        print("EPOCH:", epoch + 1)
        print('LR:', optimizer.param_groups[0]['lr'])
        train_epoch(model,
                    device,
                    train_loader,
                    optimizer,
                    epoch,
                    criterion,
                    scheduler=scheduler,
                    L1=L1)
        test_epoch(model, device, test_loader, optimizer, epoch, criterion)


if __name__ == "__main__":
    main()
