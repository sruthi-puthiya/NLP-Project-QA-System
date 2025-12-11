# NLP Final Project: Transformer-Based Question Answering System

## Project Overview

This project implements a complete Question Answering (QA) system using state-of-the-art Transformer models:
- **Extractive QA**: Fine-tuned DistilBERT to extract answer spans from passages
- **Response Generation**: Fine-tuned DistilGPT-2 to generate explanatory responses
- **Interactive Demo**: Gradio interface for real-time question answering

**Dataset**: SQuAD 2.0 (Stanford Question Answering Dataset)

## Project Requirements Met

**Data Preparation (15%)**: Load SQuAD 2.0, explore, preprocess, visualize  
**Extractive QA (25%)**: Fine-tune BERT, evaluate with EM/F1 scores  
**Response Generation (25%)**: Fine-tune GPT-2, evaluate with perplexity  
**Advanced Exploration (15%)**: Attention visualization, zero-shot comparison, LoRA  
**Integrated Demo (10%)**: Gradio interface  
**Analysis & Reflection (10%)**: Model comparison, discussion, reflection  

## Quick Start

### Prerequisites

```bash
# Python 3.8 or higher
# GPU recommended (but not required)
```

### Installation

```bash
# Install required packages
pip install transformers datasets torch torchvision torchaudio
pip install gradio bertviz wordcloud matplotlib seaborn pandas numpy
pip install accelerate evaluate scikit-learn peft
```

### Running the Notebook

1. Open `NLP_Final_Project_QA_System.ipynb` in Jupyter Notebook or JupyterLab
2. Run cells sequentially from top to bottom
3. The notebook is designed to run end-to-end

**Note**: If you have limited computational resources:
- Set `USE_SUBSET = True` in the data preparation section (already set by default)
- This uses 10,000 training samples instead of the full 87,000
- Results will still be excellent!

## Expected Results

### Extractive QA Performance
- **Exact Match (EM)**: ~70-80%
- **F1 Score**: ~75-85%
- **Training Time**: ~15-30 minutes (with GPU, subset)

### Generation Performance
- **Perplexity**: ~15-25 (lower is better)
- **Training Time**: ~10-20 minutes (with GPU, subset)

### Efficiency Gains
- **LoRA**: 99% reduction in trainable parameters
- **Performance**: Similar to full fine-tuning

## Project Structure

```
NLP_Final_Project_QA_System.ipynb    # Main notebook with all code
README.md                             # This file
Final.pdf                         # Project requirements

# Generated during execution:
data_exploration.png                  # Dataset visualizations
qa_training_curves.png               # BERT training progress
gpt2_training_curves.png             # GPT-2 training progress
attention_heatmap.png                # Attention visualization
performance_comparison.png           # Model comparison charts
model_comparison.csv                 # Detailed metrics table
qa_model_final/                      # Fine-tuned BERT model
gpt2_generation_final/               # Fine-tuned GPT-2 model
```

## Key Features

### 1. Data Preparation
- Comprehensive SQuAD 2.0 exploration
- Answerable vs unanswerable question analysis
- Passage/question length distributions
- Word cloud visualizations

### 2. Extractive QA
- DistilBERT fine-tuning with Hugging Face Trainer
- Handles unanswerable questions (SQuAD 2.0 feature)
- Evaluation with Exact Match and F1 scores
- Training/validation loss curves

### 3. Response Generation
- DistilGPT-2 fine-tuning for explanations
- Prompt-response pair creation
- Perplexity evaluation
- Sample generation with temperature sampling

### 4. Advanced Exploration
- **Attention Visualization**: BertViz integration for interpretability
- **Zero-Shot Comparison**: Pre-trained vs fine-tuned performance
- **LoRA**: Parameter-efficient fine-tuning demonstration

### 5. Interactive Demo
- Gradio web interface
- Real-time answer extraction and explanation generation
- Example questions included
- Deployable to Hugging Face Spaces

## Customization Options

### Hyperparameters
```python
# In the notebook, you can adjust:
learning_rate = 2e-5          # Learning rate
batch_size = 16               # Batch size
num_epochs = 3                # Training epochs
max_length = 384              # Maximum sequence length
doc_stride = 128              # Sliding window stride
```

### Model Variants
```python
# Try different models:
model_checkpoint = "distilbert-base-uncased"  # Current
# model_checkpoint = "bert-base-uncased"      # Larger, more accurate
# model_checkpoint = "roberta-base"           # Alternative architecture

gpt2_model_name = "distilgpt2"                # Current
# gpt2_model_name = "gpt2"                    # Larger, better generation
```

## Deployment

### Local Deployment
The Gradio demo runs locally in the notebook. Simply execute the demo cells.

### Hugging Face Spaces (Free)
1. Create account at https://huggingface.co
2. Create new Space with Gradio SDK
3. Upload models and create `app.py` with the Gradio code
4. Your demo will be live at: `https://huggingface.co/spaces/YOUR_USERNAME/qa-system`

## Performance Tips

### For Better Accuracy
- Use full dataset (`USE_SUBSET = False`)
- Train for more epochs (5-10)
- Use larger models (bert-base-uncased, gpt2)
- Increase max_length for longer contexts

### For Faster Training
- Keep `USE_SUBSET = True`
- Use DistilBERT/DistilGPT-2 (current setup)
- Enable FP16 training (automatic with GPU)
- Use LoRA for parameter-efficient fine-tuning

## Troubleshooting

### Out of Memory Error
```python
# Reduce batch size
per_device_train_batch_size = 8  # or 4

# Reduce max_length
max_length = 256  # instead of 384
```

### Slow Training
```python
# Use smaller subset
train_dataset = dataset["train"].select(range(5000))

# Reduce epochs
num_train_epochs = 2
```

### Import Errors
```bash
# Reinstall packages
pip install --upgrade transformers datasets torch
```

## Learning Resources

- [Hugging Face Transformers Documentation](https://huggingface.co/docs/transformers)
- [SQuAD 2.0 Paper](https://arxiv.org/abs/1806.03822)
- [BERT Paper](https://arxiv.org/abs/1810.04805)
- [GPT-2 Paper](https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)
- [LoRA Paper](https://arxiv.org/abs/2106.09685)



## Author

**Sruthi Puthiyandy**  
Natural Language Processing Course  
December 2025



