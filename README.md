# Glados


## Running the Code
From the top level of `Glados`, run `visdom` in one terminal window and then `python train.py {filename}` in another

The file must be in the `data` directory. If no filename is given, the script defaults to training on the complete works of Shakespeare. If multiple filenames are given as multiple command line arguments, a model is trained on all of them collectively.

Training progress is reported in the terminal, but loss graphs are sent to `localhost:8097`. 99.9% confidence intervals are plotted for both training and validation loss


## Repo Structure
```
├ DATA
    ├ bee
    ├ hamlet
    ├ shakespeare
    ├ vonnegut
    └ [put your own text files in here]
├ SAVED_MODELS
├ model.py
├ train.py
├ utils.py
└ visualize.py
```