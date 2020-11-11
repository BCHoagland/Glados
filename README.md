# Glados


## Running the Code
Navigate to wherever you downloaded this repo. If this is your first time using it, run `pipenv install` to install all the necessary dependencies. Then run `pipenv shell` to enter into a virtual environment with all the dependencies. Now run the command `visdom` (if this throws an error about not being a recognized command, run `python -m visdom.server` instead).

Open another terminal window and navigate to the repo again, then `pipenv shell` again. From there, run `python train.py {filename(s)}`, where `{filename(s)}` is either
1. the name of a single file from the `data` folder of the Glados repo, or
2. a space-separated list of file names from the `data` folder.
If no filename is given, the script defaults to training on the complete works of Shakespeare. If multiple filenames are given (case 2 above), a model is trained on all of them collectively.

Once a model has finished training, it should output results to the `results` folder of the repo (this folder will be created if it doesn't already exist).

### Examples
Training on the novel "Slaughterhouse-Five" by Kurt Vonnegut: `python train.py vonnegut`.

Training on both Hamlet and the Bee Movie collectively: `python train.py bee hamlet`.

Training progress is reported in the terminal, but loss graphs are sent to `localhost:8097` in the browser. 99.9% confidence intervals are plotted for both training and validation loss.


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
