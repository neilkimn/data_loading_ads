# Based on: https://github.com/mulin88/compcars

import warnings

warnings.filterwarnings("ignore")

import time
import torch
import torchvision
import torch.optim as optim

from torch.utils import data as D
from torch.optim import lr_scheduler
from torch.autograd import Variable

import torch.profiler


NUM_DATALOADER_WORKERS = 1
miscPath = "data"
modle_csv_file = "labels.csv"
trainFilename = "data/train.txt"


def get_labels():
    ##############################
    # calculate the features size from paper's training dataset
    ##############################
    paperTrainFeatureSet = set()
    with open(trainFilename, newline="\n") as trainfile:
        for line in trainfile:
            feature = line.split("/")[1]
            paperTrainFeatureSet.add(feature)
    print("total train feature size: %s" % (len(paperTrainFeatureSet)))
    num_classes = len(paperTrainFeatureSet)
    print("num_classes:", num_classes)
    paperTrainFeatureList = sorted(paperTrainFeatureSet)

    return paperTrainFeatureList


def train_model(
    loader_train,
    loader_valid,
    model,
    criterion,
    optimizer,
    scheduler,
    epochs,
):
    dataset_sizes = {
        "train": len(loader_train.dataset),
        "valid": len(loader_valid.dataset),
    }
    print(
        f"Training on {len(loader_train.dataset)} samples. Validating on {len(loader_valid.dataset)} samples"
    )

    prof = torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        schedule=torch.profiler.schedule(wait=1, warmup=1, active=5, repeat=1),
        record_shapes=False,
        profile_memory=False,
        with_stack=True,
    )
    print("Running with PyTorch Profiling enabled! Profiler loaded")

    for epoch in range(epochs):
        epoc_time = time.time()

        ### Train
        scheduler.step()
        model.train(True)
        running_loss = 0.0
        train_running_corrects = 0

        for inputs, labels in loader_train:
            inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())

            optimizer.zero_grad()

            outputs = model.forward(inputs)

            loss = criterion(outputs, labels)
            preds = torch.max(outputs, 1)[1]

            train_running_corrects += torch.sum(preds == labels.data)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        train_epoch_acc = train_running_corrects.double() / dataset_sizes["train"] * 100

        train_time = time.time() - epoc_time

        ### Validation
        model.train(False)
        running_loss = 0.0
        valid_running_corrects = 0

        for inputs, labels in loader_valid:
            inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())

            optimizer.zero_grad()

            outputs = model.forward(inputs)

            loss = criterion(outputs, labels)
            _, preds = torch.max(outputs.data, 1)
            valid_running_corrects += torch.sum(preds == labels.data)

            running_loss += loss.item()

        valid_epoch_acc = valid_running_corrects.double() / dataset_sizes["valid"] * 100

        print(
            "Epoch [{}/{}] train acc: {:.4f}% "
            "valid  acc: {:.4f}% Time: {:.0f}s train corr: {:d}  valid corr: {:d}  ".format(
                epoch,
                epochs - 1,
                train_epoch_acc,
                valid_epoch_acc,
                (train_time),
                train_running_corrects,
                valid_running_corrects,
            )
        )
        prof.step()
    prof.stop()
    prof.export_chrome_trace("trace.json")

    return model


def run_experiment(loader_train, loader_valid, epochs=3):
    if not torch.cuda.is_available():
        raise Exception("CUDA not available to Torch!")

    device = torch.device("cuda:0")
    net = torchvision.models.resnet18(pretrained="imagenet")

    num_ftrs = net.fc.in_features  # num_ftrs = 2048
    print("features", num_ftrs, "classes", 431)
    net.fc = torch.nn.Linear(num_ftrs, 431)

    if torch.cuda.device_count() > 1:
        net = torch.nn.DataParallel(net)

    net.cuda()

    criterion = torch.nn.CrossEntropyLoss()
    criterion.cuda()

    optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9)
    my_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    if torch.cuda.device_count() > 1:
        net = torch.nn.DataParallel(net)
    net.cuda()

    start_time = time.time()
    model = train_model(
        loader_train,
        loader_valid,
        net,
        criterion,
        optimizer,
        my_scheduler,
        epochs=epochs,
    )
    print("Training time: {:10f} minutes".format((time.time() - start_time) / 60))