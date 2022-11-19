import torch
from utils.logger import NativeLogger

logger = NativeLogger().getLogger()

def saveModel(model, modelName):
    """Save trained model to local directory for persistence and reusage.

    Args:
        model (LeNet5 or LeNet5BatchNorm): Trained model to be saved
        modelName (string): File name for trained model
    """
    logger.info('Saving model')
    torch.save(model.state_dict(), modelName + '.pth')

def loadModel(model, modelLocation, device):
    """Load model from local directory

    Args:
        model (LeNet5 or LeNet5BatchNorm): Trained model to be saved
        modelLocation (string): File location where model can be loaded from
        device (torch.device): Device to load the model on

    Returns:
        LeNet5 or LeNet5BatchNorm: Trained model
    """
    logger.info('Loading model')
    model.load_state_dict(torch.load(modelLocation, map_location=device))
    model.eval()
    return model