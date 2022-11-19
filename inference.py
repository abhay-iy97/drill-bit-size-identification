import glob
from collections import Counter
import cv2
import torch
import argparse

from utils.logger import NativeLogger
from models.basicLeNet import LeNet5
from models.batchNormLeNet import LeNet5BatchNorm
from utils.modelPersistence import loadModel

logger = NativeLogger().getLogger()
classNames = {0: '2.0 mm x 26 mm', 1: '2.0 mm x 28 mm', 2: '2.8 mm x 22 mm', 3: '3.5 mm x 19 mm', 4: '3.5 mm x 22 mm', 5: '3.5 mm x 28 mm', 6: '3.5 mm x 30 mm', 7: '4.2 mm x 22 mm', 8: '4.2 mm x 30 mm'}
  
def parseInferenceArguments(parser: argparse.ArgumentParser):
    """Function to add argument specifications for the input parameters of the program which runs the inference using the trained model

    Args:
        parser (argparse.ArgumentParser): Parser object to set input parameters for the program

    Returns:
        argparse.ArgumentParser: Holds all information necessary for parsing command line input into python data types
    """
    # Parameters
    parser.add_argument('--model', type=str, default='./batchnorm_adamw.pth', required=False, help='Absolute location of trained model.pth')
    parser.add_argument('--inferenceDataset', type=str, default='./inference_images/', required=False, help='Inference dataset folder location')

    args = parser.parse_args()
    # Print out the arguments
    for k in args.__dict__:
        logger.info(k + ": " + str(args.__dict__[k]))
    return args

def runInferences(model, inferenceDatasetPath, device, classNames):
    """Function to run inference on the unseen dataset

    Args:
        model (LeNet5 or LeNet5BatchNorm): Trained model
        inferenceDatasetPath (string): Directory location of dataset for which we need to run inferences
        device (torch.device): Device to run model inference
        classNames (dict): Dictionary of integer to class name mapping. Example - {0: '2.0 mm x 26 mm', 1: '2.0 mm x 28 mm'...}
    """
    logger.info('Running inferences')
    model.eval()
    allPredictions = {}
    fileList = glob.glob(inferenceDatasetPath + '*')
    for fileName in fileList:
        img = cv2.imread(fileName)
        img = cv2.resize(img, (1280, 960))
        imgTensor = torch.from_numpy(img)
        imgTensor = imgTensor.permute(2, 0, 1)
        imgTensor = torch.unsqueeze(imgTensor, 0)
        imgTensor = imgTensor.to(device)
        outputs = model(imgTensor.float())
        _, predictions = torch.max(outputs.data, 1)
        predictions = predictions.cpu().numpy()
        allPredictions[fileName] = classNames[predictions[0]]
    
    for key, value in allPredictions.items():
        logger.info(f'FileName: {key} - Predicted Class: {value}')
    
    counterPrediction = Counter(allPredictions.values())
    logger.info('\n\nPrinting table for each class found and number of images associated with the class\n====================================================')
    for key, value in counterPrediction.items():
        logger.info(f'Class Name: {key} - Number of examples predicted: {value}')

    with open("output.txt", "w") as f:
        
        for key, value in allPredictions.items():
            f.write(f'FileName: {key} - Predicted Class: {value}\n')
        f.write('\n\nPrinting table for each class found and number of images associated with the class\n====================================================\n')
        for key, value in counterPrediction.items():
            f.write(f'Class Name: {key} - Number of examples predicted: {value}\n')
        

def main():
    """
    Main function for program execution
    """
    parser = argparse.ArgumentParser(description="Drill bit inference")
    args = parseInferenceArguments(parser)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = LeNet5BatchNorm()
    logger.info('Inference stage')
    model = loadModel(model, args.model, device)
    model = model.to(device)
    runInferences(model, args.inferenceDataset, device, classNames)


if __name__ == '__main__':
    main()
