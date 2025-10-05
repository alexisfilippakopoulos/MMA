# Mixture of Multimodal Adapters for Sentiment Analysis

This repository contains code of Mixture of Multimodal Adapters for Sentiment Analysis, accepted at 2025 Annual Conference of the Nations of the Americas Chapter of the Association for Computational Linguistics (NAACL 2025).

## Introduction

MMA is a plug-and-play module, which can be flexibly applied to various pre-trained language models and transform these models into a multi-modal model that can handle MSA tasks.
![MMA](MMA_v1.1.png)

## Usage

1. Download the word-aligned CMU-MOSI dataset from [MMSA](https://github.com/thuiar/MMSA). Download the pre-trained BERT model from [Huggingface](https://huggingface.co/google-bert/bert-base-uncased/tree/main).
2. Set up the environment.
```
conda create -n MMA python=3.7
conda activate MMA
pip install -r requirements.txt
```
3. Start training.

Training on CMU-MOSI:

```
python main.py --dataset mosi --data_path [your MOSI path] --bert_path [your bert path]
```
Training on other dataset and LLM:

coming soon.
## Citation
```
@inproceedings{chen2025mixture,
  title={Mixture of Multimodal Adapters for Sentiment Analysis},
  author={Chen, Kezhou and Wang, Shuo and Ben, Huixia and Tang, Shengeng and Hao, Yanbin},
  booktitle={Proceedings of the 2025 Conference of the Nations of the Americas Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 1: Long Papers)},
  pages={1822--1833},
  year={2025}
}
```

## Contact 
If you have any question, feel free to contact me through chenkezhou@mail.ustc.edu.cn.
