# Neural Basics

A simple PyTorch project demonstrating a basic neural network for classification.

## Project Structure

- `model.py`  
  Defines a simple feedforward neural network (`MySimpleModel`) using PyTorch.

- `model_loader.py`  
  Generates random data, creates PyTorch datasets and data loaders for training and validation.

- `train.py`  
  Trains the model on the generated data, evaluates on the validation set, and saves the trained model to `model.pth`.

- `predict.py`  
  Loads the saved model and evaluates its accuracy on the validation set.

## Setup

1. **Create a virtual environment (optional but recommended):**
   ```bash
   python -m venv .venv
   ```
   Activate it:
   - On Windows:
     ```bash
     .venv\Scripts\activate
     ```
   - On macOS/Linux:
     ```bash
     source .venv/bin/activate
     ```

2. **Install dependencies:**
   ```bash
   pip install torch
   ```

## Usage

1. **Train the model:**
   ```bash
   python train.py
   ```
   This will train the model and save it as `model.pth`.

2. **Evaluate the model:**
   ```bash
   python predict.py
   ```
   This will load the saved model and print the validation accuracy.

## Notes

- The data is randomly generated for demonstration purposes.
- You can modify `model_loader.py` to use your own dataset.
- The model is a simple two-layer neural network suitable for basic classification tasks.

## License

This project is for educational purposes.