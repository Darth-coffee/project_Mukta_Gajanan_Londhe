from LV_model import DeeperCNN as TheModel
from LV_train import train_model as the_trainer  # Your main training loop function is likely named `train`
from LV_predict import main as the_predictor  # Replace with your actual function for prediction
from LV_dataset import SyllableDataset as TheDataset
from torch.utils.data import DataLoader as the_dataloader
from LV_config import BATCH_SIZE as the_batch_size
from LV_config import EPOCHS as total_epochs
