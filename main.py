from torch.utils.data import random_split
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import argparse
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import cv2

from utils.dataset import DrillDataset
from models.basicLeNet import LeNet5
from models.batchNormLeNet import LeNet5BatchNorm
from utils.modelPersistence import loadModel, saveModel
from utils.logger import NativeLogger


logger = NativeLogger().getLogger()

def parse_arguments(parser):
    """Function to add argument specifications for the input parameters of the program

    Args:
        parser (argparse.ArgumentParser): Parser object to set input parameters for the program

    Returns:
        argparse.ArgumentParser: Holds all information necessary for parsing command line input into python data types
    """
    # Parameters
    parser.add_argument('--mode', type=str, default='test', required=True, choices=['train', 'test'], help='Model Training or Testing')
    parser.add_argument('--pathToDataset', type=str, default='/home1/adiyer/coding-assessments/company/videos_frames/', required=False, help='Dataset folder location')
    parser.add_argument('--savedModelLocation', type=str, default='./model.pth', required=False, help='Absolute location of trained model.pth')
    parser.add_argument('--numEpochs', type=int, default=1, required=False, help='Number of epochs to train the model for')
    parser.add_argument('--batchSize', type=int, default=32, required=False, help='Batch size for train/val/test')    
    parser.add_argument('--optimizer', type=str, default='adam', required=False, choices=['adam', 'adamw', 'sgdm', 'rmsprop'], help='Optimizer to utilize for training the model')
    parser.add_argument('--learningRate', type=float, default=1e-3, required=False, help='Learning rate for the model')    
    parser.add_argument('--weightDecay', type=float, default=0, required=False, help='Weight decay to be used for training the model')
    parser.add_argument('--momentum', type=float, default=0, required=False, help='Momentum to be used for training the model')
    parser.add_argument('--savedModelName', type=str, default='model1', required=True, help='Save trained model with this name')
    parser.add_argument('--modelVersion', type=str, default='basic', required=False, choices=['basic', 'batchnorm'], help='Model version to use - Either basic LeNet or LeNet with Batch normalization')
    parser.add_argument('--lossPlotName', type=str, default='lossPlot1', required=False,  help='File name for train vs val loss')
    parser.add_argument('--confusionMatrixName', type=str, default='confMat1', required=False, help='File name for confusion matrix')


    args = parser.parse_args()
    # Print out the arguments
    for k in args.__dict__:
        logger.info(k + ": " + str(args.__dict__[k]))
    return args

def train(model, device, trainLoader, valLoader, optimizer, numEpochs):
    """Training loop for the model

    Args:
        model (LeNet5 or LeNet5BatchNorm): Model to train
        device (torch.device): Device to train the model on
        trainLoader (DataLoader): DataLoader which contains data for training the model
        valLoader (DataLoader): DataLoader which contains data for validating the model
        optimizer (torch.optim): Optimizer for updating parameters of the model
        numEpochs (_type_): Number of epochs to train the model for

    Returns:
        LeNet5 or LeNet5BatchNorm: Trained model
        list: List of train losses for each step
        list: List of validation losses for every epoch
        list: List of train losses for every epoch
    """
    ######### MODEL CREATION ###########
    criterion = nn.CrossEntropyLoss()
    # model = model.to(device)
    train_losses_step = []
    val_losses = []
    train_losses_epoch = []
    ######### MODEL TRAINING ###########
    for epoch in range(1, numEpochs + 1):
        trainLoss = 0.0
        validationLoss = 0.0

        model.train()
        for idx, (data, label) in enumerate(trainLoader, 0):
            data, label = data.to(device), label.to(device)
            optimizer.zero_grad()
            output = model(data.float())
            loss = criterion(output, label)
            loss.backward()
            optimizer.step()
            trainLoss += loss.item()
            train_losses_step.append(loss.item())
        trainLoss = trainLoss/len(trainLoader)

        
        with torch.no_grad():
            for input, labels in valLoader:
                input, labels = input.to(device), labels.to(device)
                y_pred = model(input.float()) 
                loss = criterion(y_pred, labels)
                validationLoss += loss.item()

            validationLoss = validationLoss / len(valLoader)
            val_losses.append(validationLoss)
            train_losses_epoch.append(trainLoss)

            logger.info('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f} \tNumber of training dataloaders: {} \tNumber of validation dataloaders: {}'.format(epoch, trainLoss, validationLoss, len(trainLoader), len(valLoader)))
    return model, train_losses_step, val_losses, train_losses_epoch

def predict(model, dataloader, device, classNames, fileName):
    """Generate predictions of a model for the given dataloader

    Args:
        model (LeNet5 or LeNet5BatchNorm): Trained LeNet5/LeNet5BatchNorm model object to generate predictions
        dataloader (DataLoader): Custom dataloader for testing dataset
        device (torch.device): Device to run the model on
        classNames (dict): Dictionary of integer to class name mapping. Example - {0: '2.0 mm x 26 mm', 1: '2.0 mm x 28 mm'...}
        fileName (string): Output file name for confusion matrix
    """
    model.eval()
    correct = 0
    allTargets = []
    allPredictions = []
    for inputs, targets in dataloader:
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs.float())
        _, predictions = torch.max(outputs.data, 1)
        correct += (predictions == targets).sum().item()
        targets, predictions = targets.cpu().numpy(), predictions.cpu().numpy()
        allTargets.extend(targets)
        allPredictions.extend(predictions)

    confusionMatrix = confusion_matrix(allTargets, allPredictions)
    disp = ConfusionMatrixDisplay(confusion_matrix=confusionMatrix, display_labels=classNames.values())

    logger.info(f'Correct: {correct}, dataloader.dataset: {len(dataloader.dataset)}')
    logger.info(f'Overall Accuracy: {correct/len(dataloader.dataset)}')
    fig, ax = plt.subplots(figsize=(20,30))
    disp.plot(ax=ax)
    plt.savefig('output_files/'+fileName+'.jpg')

def createDataLoader(batchSize, pathToDataset):
    """Function to create train/test/val dataloaders and get class names dictionary. Example - {0: '2.0 mm x 26 mm', 1: '2.0 mm x 28 mm'...}

    Args:
        batchSize (int): Batch size for data loaders
        pathToDataset (string): Path to dataset which has to be used for creating train/test/val split.

    Returns:
        DataLoader: Training dataset loader
        DataLoader: Validation dataset loader
        DataLoader: Testing dataset loader
        dict: Dictionary of integer to class name mapping. Example - {0: '2.0 mm x 26 mm', 1: '2.0 mm x 28 mm'...}
    """
    drillData = DrillDataset(imgPath = pathToDataset)
    drillDatasetLen = drillData.__len__()
    logger.info(f'Drill dataset length: {drillDatasetLen}') # 20274
    
    # 80-10-10 split
    trainSplit, valSplit, testSplit = 16219, 2028, 2027
    trainDataset, valDataset, testDataset = random_split(drillData, [trainSplit, valSplit, testSplit], generator=torch.Generator().manual_seed(42))
    
    trainLoader = DataLoader(trainDataset, batch_size = batchSize, shuffle=True)
    valLoader = DataLoader(valDataset, batch_size = batchSize, shuffle=False)
    testLoader = DataLoader(testDataset, batch_size = batchSize, shuffle=False)

    return trainLoader, valLoader, testLoader, drillData.getClassNames()

def getOptimizer(model, args):
    """Function to get optimizer based on input parameters of the program

    Args:
        model (LeNet or LeNet5BatchNorm): Model which we want to train
        args (argparse.ArgumentParser): Input arguments to the program

    Returns:
        torch.optim: Returns API to Adam / AdamW/ SGD with momentum/ RMSProp based on input argument
    """
    if args.optimizer == 'adam':
        return torch.optim.Adam(model.parameters(), lr=args.learningRate, weight_decay=args.weightDecay)
    elif args.optimizer == 'adamw':
        return torch.optim.AdamW(model.parameters(), lr=args.learningRate, weight_decay=args.weightDecay)
    elif args.optimizer == 'sgdm':
        return torch.optim.SGD(model.parameters(), lr=args.learningRate, momentum=args.momentum, weight_decay=args.weightDecay)
    elif args.optimizer == 'rmsprop':
        return torch.optim.RMSprop(model.parameters(), lr=args.learningRate, momentum=args.momentum, weight_decay=args.weightDecay)

def plot_losses(train_losses_step, val_losses, train_losses_epoch, fileName):    
    """Function to plot the training and validation losses 

    Args:
        train_losses_step (list): List of train losses for each step
        val_losses (list): List of validation losses for every epoch
        train_losses_epoch (list): List of train losses for every epoch
    """
    fig = plt.figure(figsize = (30, 5))
    xvalues = list(range(1, len(val_losses)+1))
    
    ax2 = plt.subplot(121)
    ax2.plot(xvalues, val_losses, label="val_loss")
    ax2.plot(xvalues, train_losses_epoch, label="train_loss")
    ax2.title.set_text("Loss for every epoch")
    ax2.legend()
    ax2.set_ylabel("Loss")
    ax2.set_xlabel("Epochs")

    plt.savefig('output_files/'+fileName+'.jpg')

def main():
    """
    Main function for program execution
    """
    parser = argparse.ArgumentParser(description="Drill bit training")
    args = parse_arguments(parser)
    trainLoader, valLoader, testLoader, classNames = createDataLoader(args.batchSize, args.pathToDataset)
    device = torch.device('cpu')
    model = LeNet5()

    if args.modelVersion == 'batchnorm':
        model = LeNet5BatchNorm()

    if torch.cuda.is_available():
        device = torch.device('cuda')
        # model = nn.DataParallel(model)

    model = model.to(device)

    if args.mode == 'train':
        logger.info('Training stage')
        optimizer = getOptimizer(model=model, args=args)
        model, train_losses_step, val_losses, train_losses_epoch = train(model, device, trainLoader, valLoader, optimizer=optimizer, numEpochs=args.numEpochs)
        saveModel(model, args.savedModelName)
        predict(model, testLoader, device, classNames, args.confusionMatrixName)
        plot_losses(train_losses_step, val_losses, train_losses_epoch, args.lossPlotName)

    elif args.mode == 'test':
        logger.info('Testing stage')
        model = loadModel(model, args.savedModelLocation, device)
        predict(model, testLoader, device, classNames)

if __name__ == '__main__':
    main()
