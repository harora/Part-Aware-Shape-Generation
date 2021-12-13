## Instruction for running mixer

### Install dataset
The preprocessed files can be found [here](https://www.dropbox.com/sh/hxfnhwn4r2sp33e/AADeIqSBO00wXuzFJQYETopKa?dl=0).

### Folder structure
```
|--mixer
|--scorer
|--dataset
    |--processed_pcd
```

### Run training
```
python training.py
```
### Run Scorer pipeline

Input directory parameter is the parent directory where to find the .npz files.
```
$ scorer/run_scorer.sh ${INPUT_DIRECTORY}
```
