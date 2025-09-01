### 05 PyTorch Going Modular

We're going to turn the most useful code cell in notebook 04 into a series of Python scripts saved to a directory call `going_modular`.

---

#### What is going modular

For example, you might be instructed to run code like the following in a terminal/command line to train a model:
```
python train.py --model MODEL_NAME --batch_size BATCH_SIZE --lr LEARNING_RATE --num_epochs NUM_EPOCHS
```
---

#### What's we're going to cover

The main concept of this section is: **turn useful notebook code cells into reusable Python files**.

Doing this will save us writing the same code over and over again.

THere are two notebooks for this section:
1. `05. Going Mudular: Part 1 (cell mode)` - this notebook is run as a traditional Jupyter Notebook/Google Colab notebook and is a condensed version of notebook 04

1. `05. Going Mudular: Part2 (script mode)` - this notebook is the same as number 1 but with added functionality to turn each of the major sections into Python scripts, such as, `data_setup.py` and `train.py`.

This text in this document focuses on the code cells 05. Going Modular: Part 2 (script mode), the ones with `%%writefile`... at the top