# An Aspect-Centralized Graph Convolutional Network for Aspect-based Sentiment Classification

**ACGCN** - **A**spect **C**entralized **G**raph **C**onvolutional **N**etwork
* Code for [NLPCC 2021](http://tcci.ccf.org.cn/conference/2021/index.php) accepted paper titled "An Aspect-Centralized Graph Convolutional Network for Aspect-based Sentiment Classification"
* Weixiang Zhao, Yanyan Zhao, Xin Lu and Bing Qin.

## Requirements

* Python 3.7
* PyTorch 1.6.0
* Supar 1.0.0

## Usage

* Install [SpaCy](https://spacy.io/) package and language models with
```bash
pip install spacy
```
and
```bash
python -m spacy download en
```
* Install Biaffine parser with
```bash
pip install -U supar
```
* Generate Aspect-Centralized Graph with
```bash
python generate_acg.py
```
* Download pretrained GloVe embeddings with this [link](http://nlp.stanford.edu/data/wordvecs/glove.840B.300d.zip) and extract `glove.840B.300d.txt` into `glove/`.

## Train
* optional arguments could be found in [train.py](/train.py)
* For dataset Lap14
```bash
python train.py --model_name acgcn --embed_type glove --layernorm True --highway True --batch_size 16 --dataset lap14
```
```bash
python train.py --model_name acgcn_bert --embed_type bert --hidden_dim 768 --learning_rate 5e-5 --dataset lap14
```
* For dataset Rest14
```bash
python train.py --model_name acgcn --embed_type glove --layernorm True --highway True --batch_size 16 --dataset rest14
```
```bash
python train.py --model_name acgcn_bert --embed_type bert --hidden_dim 768 --learning_rate 5e-5 --dataset rest14
```
* For dataset Rest15
```bash
python train.py --model_name acgcn --embed_type glove --dataset rest15
```
```bash
python train.py --learning_rate 5e-5 --model_name asgcn_bert --embed_type bert --hidden_dim 768 --dataset rest15
```
* For dataset Rest16
```bash
python train.py --model_name acgcn --embed_type glove --highway True --dataset rest16
```
```bash
python train.py --learning_rate 5e-5 --model_name asgcn_bert --embed_type bert --hidden_dim 768 --dataset rest16
```
* For dataset Twitter
```bash
python train.py --model_name acgcn --embed_type glove --layernorm True --dataset twitter
```
```bash
python train.py --learning_rate 5e-5 --model_name asgcn_bert --embed_type bert --hidden_dim 768 --layernorm True --dataset twitter
```

## Model

An overview of proposed ACGCN model is given below

![image](https://github.com/circle-hit/ACGCN/blob/master/models/model.jpg)

## Credits
* The code of this repository partly relies on [ASGCN](https://github.com/GeneZC/ASGCN) and I would like to show my sincere gratitude to authors of it.