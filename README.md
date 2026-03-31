# Transformer-Based Addition Project

This project implements a simple **Transformer-based model** (encoder-only architecture) from scratch to perform the addition of two-digit numbers. The model is designed to handle mixed formats, including English digits (0-9) and Urdu digits (۰-۹), by mapping them to the same token IDs.

## Features

- **Custom Transformer Architecture**: Built from scratch using an encoder-only approach (no decoder).
- **Mixed Digit Support**: Handles addition like `17+33=`, `۲۳+۱۵=`, and `23+۱۵=`.
- **Signed Operand Support**: Handles negative numbers and explicit `+` signs — e.g. `-20+30=`, `-20+-30=`, `+5++95=`.
- **Token Classification**: The model classifies each token into one of 7 roles: `Sign_1`, `Operand_1`, `Plus Operator`, `Sign_2`, `Operand_2`, `Equals`, `PAD`.
- **Programmable Dataset**: Generates a dataset covering all integer combinations from `-99` to `99` for both operands, including explicit-plus variants.
- **High Accuracy**: Achieves near 100% token accuracy within 10 training epochs.
- **Streamlit Integration**: Includes a dark-theme web app with a per-token role grid, confidence scores, example buttons, and a session history panel.

## Project Structure

- `model.py`: Defines the `AdditionTransformer` and `PositionalEncoding`.
- `dataset.py`: Handles programmatic dataset generation and PyTorch `Dataset` loading.
- `train.py`: contains the training loop and model saving logic (`model.pth`).
- `predict.py`: CLI for running inference on single mathematical expressions.
- `streamlit_app.py`: Streamlit web application for an interactive user interface.
- `tokenizer.py`: Handles mapping characters to token IDs and role labels.
- `requirements.txt`: Lists the necessary Python dependencies.

## Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd Transformer_addition_project
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### 1. Training the Model
To generate the dataset and train the Transformer model, run:
```bash
python train.py
```
This will save the trained weights to `model.pth`.

### 2. Running Inference (CLI)
To test the model on a specific addition expression via the command line:
```bash
python predict.py
```

### 3. Launching the Web App
To run the interactive Streamlit application:
```bash
streamlit run streamlit_app.py
```

## How it Works

1. **Tokenization**: Each character in the expression is mapped to a token ID (0–9 for digits, 10 for `+`, 11 for `-`, 12 for `=`, 13 for `<PAD>`). English and Urdu digits share the same IDs.
2. **Role Classification**: The Transformer classifies every token position into one of 7 roles: `Sign_1` (optional sign before first number), `Operand_1` (first number's digits), `Plus Operator` (the `+` between operands), `Sign_2` (optional sign before second number), `Operand_2` (second number's digits), `Equals`, and `PAD`.
3. **Post-Processing**: Predicted roles are used to reconstruct both signed operands, which are then added with Python to produce the final result.
