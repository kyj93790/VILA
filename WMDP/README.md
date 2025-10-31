# WMDP

## Installation

Install the Python dependencies with pip:
```bash
pip install -r pip_requirements.txt
```

## Get the Data

Follow the [link](https://github.com/centerforaisafety/wmdp?tab=readme-ov-file) to download the WMDP-Bio dataset and place it in the `./WMDP/files/data` directory.

### More Details:

- The **cyber** dataset can be downloaded from Google Drive: [link](https://drive.google.com/drive/folders/1qUJWErep12y9X7wvrr4x1snjdA97IFh9)

- The **bio** dataset requires a request via Google Form: [link](https://docs.google.com/forms/d/e/1FAIpQLSdnQc8Qn0ozSDu3VE8HLoHPvhpukX1t1dIwE5K5rJw9lnOjKw/viewform)

  After submitting the form, the dataset will be sent to you via email.


## Get the importance
### FILA method
1. Run the command `./measure_imp_fila.sh`
2. After the command is complete. The importance will be stored in `./WMDP/files/results/importances/fila`
### VILA method
1. Run the comman `./measure_imp_vila.sh`
2. After the command is complete. The importance will be stored in `./WMDP/files/results/importances/vila`

## Get the unlearned model
1. Run the command `./run_wmdp_unlearn.sh`.
2. After the command is complete, the checkpoints and results will be stored in `./WMDP/files/results/unlearn_wmdp_all/NPO+FT+xxx`.


## Project Structure

- `configs/`  
  Contains configuration files for running the code, including hyperparameters.

- `files/`  
  - `data/`: Stores training datasets.  
  - `importances/`: Contains importance maps for FILA and VILA.  
  - `results/`: Saves evaluation results.

- `src/`  
  - `dataset/`: Handles dataset loading and processing.  
  - `exec/`: Entry points for running models and experiments.  
  - `loggers/`: Logging utilities.  
  - `metrics/`: Evaluation metrics implementations.  
  - `model/`: Core unlearning model code.  
    - `measure_imp.py`: Computes importance scores.  
    - `unlearn.py`: Performs unlearning with evaluation every 25 steps.  
    - `unlearn_one_step.py`: Performs unlearning and evaluates at a specified step.  
  - `unlearn/`: Implements unlearning loss functions.


## Code Base

This code is based on the [Unlearn-Smooth](https://github.com/OPTML-Group/Unlearn-Smooth) repository.  
Many thanks to the authors for their great work!
