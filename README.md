# Neural-EDU-Segmentation
A toolkit for segmenting Elementary Discourse Units (clauses).
This is a refactoring of the paper: [Toward Fast and Accurate Neural Discourse Segmentation](http://www.aclweb.org/anthology/D18-1116)


### Requirements
This version runs under Python 3.7. The AllenNLP dependency is replaced by Tensorflow-Hub.

### Data

We cannot provide the complete [RST-DT corpus](https://catalog.ldc.upenn.edu/products/LDC2002T07) due to the LDC copyright.
So we only put several samples in `./data/rst/` to test the our code and show the data structure.

If you want to train or evaluate our model on RST-DT, you need to download the data manually and put it in the same folder. Then run the following command to preprocess the data and create the vocabulary:

```
python run.py --prepare
```


### Evaluate the model on RST-DT:

You can evaluate the performance of a model after downloading and preparing the RST-DT data as mentioned above:

```
python run.py --evaluate --test_files ../data/rst/preprocessed/test/*.preprocessed
```

### Train a new model

You can use the following command to train the model from scratch:

```
python run.py --train
```

Hyper-parameters and other training settings can be modified in `config.py`.

### Segmenting raw text into EDUs

You can segment files with raw text into EDUs:

```
python run.py --segment --input_files ../data/rst/TRAINING/wsj_110*.out --result_dir ../data/results/
```

### Citation

Please cite the following paper if you use this toolkit in your work:

```
@inproceedings{wang2018edu,
  title={Toward Fast and Accurate Neural Discourse Segmentation},
  author={Wang, Yizhong and Li, Sujian and Yang, Jingfeng},
  booktitle={Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing},
  pages={962--967},
  year={2018}
}
```
