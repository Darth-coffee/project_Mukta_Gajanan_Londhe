from model import DeeperCNN as TheModel
from train import train_model as the_trainer  # Your main training loop function is likely named `train`
from predict import main as the_predictor  # Replace with your actual function for prediction
from dataset import SyllableDataset as TheDataset
from torch.utils.data import DataLoader as the_dataloader
from config import BATCH_SIZE as the_batch_size
from config import EPOCHS as total_epochs
