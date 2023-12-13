# PrivateTinyBertGlue
Training and evaluation of bert-tiny model on GLUE benchmark with differential privacy

### Models

The model aims to train and evaluate the tiny-bert available at https://huggingface.co/prajjwal1/bert-tiny using 5 different fine-tuning techniques : 
 - soft-prompt 
 - prefix-tuning
 - LoRa-tuning
 - full fine-tuning
 - last-linear-layer finetuning

### Evaluated tasks

The tasks used on the GLUE benchmark are 'mnli', 'qqp', 'sst2' and 'qnli'.

### How to use

To start using the repository, install required librairies using :

```pip install -r requirements```

To execute the training, import main.ipynb on google colab, choose 'learning_way' (the wanted type of fine-tuning) and execute the file. The training will start on all previously mentioned tasks of the GLUE benchmark.