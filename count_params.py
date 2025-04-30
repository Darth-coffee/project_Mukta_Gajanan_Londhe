import torch
import argparse

def count_parameters(pth_file):
    # Load the model weights
    checkpoint = torch.load(pth_file, map_location='cpu')

    # If the checkpoint has a 'state_dict' key, use it
    if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint  # assume it is already the state_dict

    total_params = 0
    for param_tensor in state_dict.values():
        total_params += param_tensor.numel()

    print(f"Total number of parameters: {total_params}")

if __name__ == "__main__":
    #parser = argparse.ArgumentParser(description="Count parameters in a .pth file")
    #parser.add_argument("pth_file", type=str, help="Path to the .pth model file")
    #args = parser.parse_args()

    #count_parameters(args.pth_file)

    count_parameters("checkpoints/final_weights.pth")
