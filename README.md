# README: AI Alignment with Shared Human Values

## Overview

This project is based on the "Aligning AI with Shared Human Values" classification task. The goal is to fine-tune a pre-trained large language model (LLM), specifically BERT, to classify data based on ethical alignment and ensure that the model produces outputs that align with human values. This process involves loading, tokenizing, training, and evaluating a model on ethical data to help AI systems better understand and classify morally aligned content.

## Prerequisites

Before running the notebook, ensure that the following dependencies are installed:

- Python 3.x
- PyTorch
- Hugging Face's Transformers library
- pandas
- scikit-learn
- numpy
- matplotlib (for plotting the ROC curve)

You can install the necessary packages using the following command:

```bash
pip install torch transformers pandas scikit-learn numpy matplotlib
```
Project Structure
The notebook follows these main steps:

Setup and Install Dependencies:

Install the required Python libraries for working with BERT and handling the dataset.
Importing Required Libraries:

Import libraries like PyTorch for deep learning, Hugging Face's transformers for BERT, pandas for data handling, and other utility libraries for data processing.
Loading the Dataset:

The dataset is loaded from a CSV file that contains ethically labeled data, which is used for training and evaluation.
Data Processing:

Tokenize the input text using BERT's tokenizer and prepare the data for model training by converting text into numerical format.
Model Training:

Fine-tune a pre-trained BERT model for ethical classification using the provided data. The model is trained to distinguish between morally aligned and non-aligned content.
Evaluation:

Evaluate the model's performance using the ROC (Receiver Operating Characteristic) curve and calculate the AUC (Area Under the Curve) score to measure classification accuracy.
Discussion Task:

Reflect on whether this form of "alignment" training is sufficient for large language models to understand the difference between "right" and "wrong." The discussion is open-ended, exploring whether AI can truly grasp ethical concepts beyond probabilistic outputs.
Key Ethical Considerations
Bias: Large language models are often trained on datasets that may contain biases, leading to the propagation of these biases in the outputs.
Environmental Impact: Training large models like BERT consumes significant computational resources, leading to high energy consumption and carbon emissions.
Usage
Clone the repository.
Install the necessary dependencies.
Run the Jupyter Notebook (Lab2.ipynb) to train and evaluate the model on the ethical classification task.
Reflection
After running the notebook, please consider the open-ended question posed in the notebook:

Is this form of "alignment" training enough to ensure that large language models can "understand" the difference between "right" and "wrong"?

Provide a brief reflection on this topic based on your understanding of the challenges and limitations of large language models in ethical alignment.

License
This project is licensed under the MIT License.

Acknowledgments
This project is inspired by the "Aligning AI with Shared Human Values" classification task by Dan Hendrycks et al., published at ICLR 2021.
