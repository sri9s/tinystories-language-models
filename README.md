<div align="center">

# Tiny stories language models

<a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white"></a>
<a href="https://pytorchlightning.ai/"><img alt="Lightning" src="https://img.shields.io/badge/-Lightning-792ee5?logo=pytorchlightning&logoColor=white"></a>
<a href="https://hydra.cc/"><img alt="Config: Hydra" src="https://img.shields.io/badge/Config-Hydra-89b8cd"></a>
<a href="https://github.com/ashleve/lightning-hydra-template"><img alt="Template" src="https://img.shields.io/badge/-Lightning--Hydra--Template-017F2F?style=flat&logo=github&labelColor=gray"></a><br>
[![Paper](http://img.shields.io/badge/paper-arxiv.1001.2234-B31B1B.svg)](https://www.nature.com/articles/nature14539)
[![Conference](http://img.shields.io/badge/AnyConference-year-4b44ce.svg)](https://papers.nips.cc/paper/2020)

</div>

## Description

What it does

## Dataset and format

Tiny stories can be found at https://huggingface.co/datasets/roneneldan/TinyStories.

Data format:
dict_keys(['story', 'instruction', 'summary', 'source'])
eg:
['story']
'Once upon a time, there was a big, red ball that could bounce very high. The ball was very intelligent and could speak a special language that only kids could understand. One sunny day, the ball met a little girl named Lucy.\n"Hello, Lucy!" said the ball in its special language. Lucy was surprised and asked, "Can you really talk?" The ball smiled and said, "Yes, I can! I am an intelligent ball that can bounce and talk."\nLucy and the ball became good friends. They played together every day, and the ball would bounce high into the sky, making Lucy laugh. They had lots of fun and were very happy together.\nIn the end, Lucy and the ball taught each other many new things. Lucy learned the ball\'s special language, and the ball learned how to play new games with Lucy. They were the best of friends and had many more fun days together, filled with laughter and lots of bouncing.'

['instruction']
{'prompt:': 'Write a short story (3-5 paragraphs) which only uses very simple words that a 3 year old child would understand. The story should use the verb "bounce", the noun "language" and the adjective "intelligent". The story has the following features: the story should contain at least one dialogue. Remember to only use simple words!', 'words': ['bounce', 'language', 'intelligent'], 'features': ['Dialogue']}

['summary']
'A big, red ball that could bounce high and speak a special language met a little girl named Lucy, and they became good friends, learning new things from each other and having lots of fun together.'

['source']
'GPT-4'

## Installation

#### Pip

```bash
# clone project
git clone https://github.com/YourGithubName/your-repo-name
cd your-repo-name

# [OPTIONAL] create conda environment
conda create -n myenv python=3.9
conda activate myenv

# install pytorch according to instructions
# https://pytorch.org/get-started/

# install requirements
pip install -r requirements.txt
```

#### Conda

```bash
# clone project
git clone https://github.com/YourGithubName/your-repo-name
cd your-repo-name

# create conda environment and install dependencies
conda env create -f environment.yaml -n myenv

# activate conda environment
conda activate myenv
```

## How to run

Train model with default configuration

```bash
# train on CPU
python src/train.py trainer=cpu

# train on GPU
python src/train.py trainer=gpu
```

Train model with chosen experiment configuration from [configs/experiment/](configs/experiment/)

```bash
python src/train.py experiment=experiment_name.yaml
```

You can override any parameter from command line like this

```bash
python src/train.py trainer.max_epochs=20 data.batch_size=64
```
