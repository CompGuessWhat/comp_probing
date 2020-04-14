# Probing classifiers for Attribute prediction task

In the GroLLA (Grounded Language Learning with Attributes) framework we support the goal-oriented evaluation with the 
attribute prediction auxiliary task related to assessing the degree of compositionality of the representations learned 
for a specific task. In this repository, we provide the implementation of the probing classifiers used in the 
*CompGuessWhat?!* evaluation. We believe that the code is generic enough to be reused for other datasets as well.

## Requirements

All the library requirements are stored in the file `requirements.txt`. Please install them using the command:

```
pip install -r requirements.txt
```

## Code structure

The current framework is implemented following the library skeleton provided by the AllenNLP framework.
This means that the code supports all the commands and features of an AllenNLP library. We encourage the 
reader to follow the official [AllenNLP tutorial](http://docs.allennlp.org/master/tutorials/getting_started/predicting_paper_venues/predicting_paper_venues_pt1/) as well as check the [AllenNLP as a library](https://github.com/allenai/allennlp-as-a-library-example/) starter code.

The code is organised as follows:

1. `datasets`: contains the `DatasetReader`s which are used to read the data;
2. `experiments`: contains the JSON experiments configuration files that have been used for the CompGuessWhat?! paper;
3. `metrics`: implementation of F1, Precision and Recall measure for Multi-label classification;
4. `predictors`: AllenNLP Predictor implementation for the attribute classifiers;
5. `tests`: basic AllenNLP tests for the codebase.   

## Pretrained features

We provide several features that have been used in the original paper for CompGuessWhat?!. For the sake of reproducibility
each model features are stored in a Numpy compressed format. 

### Attributes

We have a single "npz" file for every image. We assume that we have `N` objects, `N_a` abstract attributes
and `N_s` situated attributes. In this file we have three main fields:

- `objects2ids`: a one-dimensional `(N, )` array that contains all the object ids in the dataset. The index
of the object id represents the position in the corresponding attribute vectors. 
- `abstract_attributes`: `(N, N_a)` matrix where every row corresponds to the i-th object abstract attributes vector
- `situated_attributes`: `(N, N_s)` matrix where every row corresponds to the i-th object situated attributes vector

### Dialogue state features

We have a single "npy" file for every dialogue. The file contains the features for the dialogue target object.

### Download links

The features used for the experiments can be found at the following links:

- [attributes](https://www.dropbox.com/s/h5vg01nx2ai5588/compguesswhat_attributes_vectors.zip?dl=0)
- [dialogue state features](https://www.dropbox.com/s/xszhmejs88v3af6/compguesswhat_probing_features.zip?dl=0)

In the configuration files we assume that all the dialogue state features are
contained in a folder `data/comp_guesswhat/probes` whereas the attribute features
are stored in a folder `data/vg_data/vectors/`.

## Run experiments

In order to train the probing classifiers we use the AllenNLP command-line interface. For instance, in order 
to train a probing classifier that uses the DeVries et al. supervised learning representations you can use
the following command:

```
allennlp train experiments/devries_sl.json -s probes/situated/devries_sl --include-package comp_probing
``` 

This command will start the experiments using the configuration file `experiments/devries_sl.json` and it will save
the model files in the directory `probes/situated/devries_sl`.

## Citation

Please remember to cite the following paper if you're using this code:

```
CompGuessWhat?! bibtex here
```

 