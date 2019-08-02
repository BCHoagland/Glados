# Glados


## Running the Code
From the top level of `Glados`, run `visdom` in one terminal window and then `python train.py {filename}` in another

The file must be in the `data` directory. If no filename is given, the script defaults to training on the complete works of Shakespeare

Training progress is reported in the terminal, but loss graphs are sent to `localhost:8097`. 99.9% confidence intervals are plotted for both training and validation loss


## Repo Structure
├ **data**
    ├ hamlet
    ├ shakespeare
    ├ vonnegut
    └ *put your own text files in here*
├ **saved_models** *(created upon first model save)*
├ model.py
├ train.py
├ utils.py
└ visualize.py