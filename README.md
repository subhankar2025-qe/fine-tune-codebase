# Fine-Tune Codebase Model

This repository provides a **scalable and efficient solution** for fine-tuning large language models (LLMs) on codebases. It supports advanced features like **LoRA (Low-Rank Adaptation)**, **mixed precision training**, **quantization**, and **interactive testing**. Whether you're working on a small project or a large-scale codebase, this tool is designed to help you fine-tune models with ease.

---

## Features

- **LoRA Support**: Fine-tune models efficiently with fewer trainable parameters.
- **Mixed Precision Training**: Speed up training and reduce memory usage with FP16.
- **Quantization**: Reduce model size with 8-bit quantization for deployment.
- **Dataset Preprocessing**: Automatically preprocess code (e.g., remove comments).
- **Interactive Testing**: Test the fine-tuned model interactively in the terminal.
- **TensorBoard Logging**: Monitor training metrics with TensorBoard.
- **Early Stopping**: Automatically stop training if validation performance plateaus.
- **Multi-GPU Support**: Scale training across multiple GPUs for faster results.
- **Custom Tokenizers**: Use custom tokenizers for specific programming languages.

---

## Getting Started

### Prerequisites

- Python 3.8 or higher
- GPU with CUDA support (recommended for faster training)
- Required Python libraries (see `requirements.txt`)

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/ayminovitch/fine-tune-codebase.git
   cd fine-tune-codebase
   ```

2. Install the dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. (Optional) Install TensorBoard for logging:
   ```bash
   pip install tensorboard
   ```

---

## Usage

### Prepare Your Dataset

Place your codebase files in a directory (e.g., `my_project`). You can exclude specific files or folders using a `.codeignore` file (similar to `.gitignore`).

Example `.codeignore`:
```
node_modules/
venv/
*.log
*.cache
```

### Run the Script

Fine-tune a model on your codebase:
```bash
python fine_tune_codebase.py --input_dir my_project --file_extensions .py .js --fp16 --quantize --interactive
```

### Command-Line Arguments

| Argument                        | Description                                                                 | Default Value                     |
|---------------------------------|-----------------------------------------------------------------------------|-----------------------------------|
| `--input_dir`                   | Directory containing the codebase files.                                    | **Required**                      |
| `--output_file`                 | Output file for the concatenated dataset.                                   | `codebase.txt`                    |
| `--model_name`                  | Base model to fine-tune.                                                   | `codellama/CodeLlama-7b-hf`       |
| `--output_dir`                  | Directory to save the fine-tuned model.                                    | `./fine_tuned_model`              |
| `--test_prompt`                 | Prompt to test the fine-tuned model.                                       | `// Create a new function`        |
| `--ignore_file`                 | File containing ignore patterns.                                           | `.codeignore`                     |
| `--file_extensions`             | File extensions to include (e.g., `.py .js`).                              | `None` (include all files)        |
| `--learning_rate`               | Learning rate for training.                                                | `5e-5`                            |
| `--batch_size`                  | Batch size for training.                                                   | `4`                               |
| `--num_epochs`                  | Number of training epochs.                                                 | `3`                               |
| `--early_stopping_patience`     | Number of epochs to wait before early stopping.                            | `3`                               |
| `--gradient_accumulation_steps` | Number of steps to accumulate gradients.                                   | `1`                               |
| `--fp16`                        | Enable mixed precision training.                                           | `False`                           |
| `--resume_from_checkpoint`      | Resume training from the latest checkpoint.                                | `False`                           |
| `--tensorboard`                 | Log metrics to TensorBoard.                                                | `False`                           |
| `--preprocess`                  | Preprocess the dataset (e.g., remove comments).                            | `False`                           |
| `--quantize`                    | Quantize the model to 8-bit.                                               | `False`                           |
| `--interactive`                 | Enable interactive testing mode.                                           | `False`                           |

---

## Advanced Features

### LoRA (Low-Rank Adaptation)
LoRA reduces the number of trainable parameters, making fine-tuning faster and more memory-efficient. It's enabled by default in the script.

### Mixed Precision Training
Mixed precision (FP16) speeds up training and reduces memory usage. Enable it with the `--fp16` flag.

### Quantization
Quantize the model to 8-bit for reduced size and memory usage. Enable it with the `--quantize` flag.

### Interactive Testing
Test the fine-tuned model interactively in the terminal. Enable it with the `--interactive` flag.

### TensorBoard Logging
Monitor training metrics (e.g., loss, perplexity) with TensorBoard. Enable it with the `--tensorboard` flag.

---

## Example Workflow

1. Prepare your codebase:
   ```bash
   mkdir my_project
   cp -r /path/to/your/codebase/* my_project/
   ```

2. Fine-tune the model:
   ```bash
   python fine_tune_codebase.py --input_dir my_project --file_extensions .py .js --fp16 --quantize
   ```

3. Test the model interactively:
   ```bash
   python fine_tune_codebase.py --input_dir my_project --interactive
   ```

4. Monitor training with TensorBoard:
   ```bash
   tensorboard --logdir=./logs
   ```

---

## Contributing

We welcome contributions! If you'd like to contribute, please follow these steps:

1. Fork the repository.
2. Create a new branch for your feature or bugfix.
3. Commit your changes and push to the branch.
4. Submit a pull request with a detailed description of your changes.

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE.md) file for details.

---

## Acknowledgments

- [Hugging Face](https://huggingface.co/) for the `transformers` library.
- [LoRA](https://arxiv.org/abs/2106.09685) for parameter-efficient fine-tuning.
- [TensorBoard](https://www.tensorflow.org/tensorboard) for training visualization.

---

## Contact

For questions or feedback, please open an issue or contact:

- **Aymen Hammami**  
  Email: hello@aymen-hammami.com  
  GitHub: [ayminovitch](https://github.com/ayminovitch)

