from model import DeeperCNN as TheModel
from train import train as the_trainer  # Your main training loop function is likely named `train`
from predict import predict_single_image as the_predictor  # Replace with your actual function for prediction
from dataset import SyllableDataset as TheDataset
from torch.utils.data import DataLoader as the_dataloader
from config import batch_size as the_batch_size
from config import epochs as total_epochs
