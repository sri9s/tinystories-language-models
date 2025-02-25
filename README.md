<div align="center">

# 📚 Tiny Stories Language Models

<a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white"></a>
<a href="https://pytorchlightning.ai/"><img alt="Lightning" src="https://img.shields.io/badge/-Lightning-792ee5?logo=pytorchlightning&logoColor=white"></a>
<a href="https://hydra.cc/"><img alt="Config: Hydra" src="https://img.shields.io/badge/Config-Hydra-89b8cd"></a>
<a href="https://github.com/ashleve/lightning-hydra-template"><img alt="Template" src="https://img.shields.io/badge/-Lightning--Hydra--Template-017F2F?style=flat&logo=github&labelColor=gray"></a><br>
[![Paper](http://img.shields.io/badge/paper-arxiv.1001.2234-B31B1B.svg)](https://www.nature.com/articles/nature14539)
[![Conference](http://img.shields.io/badge/AnyConference-year-4b44ce.svg)](https://papers.nips.cc/paper/2020)

</div>

## 🎯 Description

This repository implements language models trained on the TinyStories dataset - a collection of simple, child-friendly stories generated using GPT-4. 

## 📊 Dataset and Format

TinyStories can be found at [HuggingFace Datasets](https://huggingface.co/datasets/roneneldan/TinyStories).

### Data Fields:

Each story entry contains:
- `story`: The main story text
- `instruction`: Prompt and constraints used to generate the story
- `summary`: Brief summary of the story
- `source`: The model used to generate the story (GPT-4)

<details>
<summary>📝 Click to see example story</summary>

**Story:**
```
Once upon a time, there was a big, red ball that could bounce very high...
```
[Rest of the example story]

**Instruction:**
- Prompt: 'Write a short story (3-5 paragraphs)...'
- Required words: ['bounce', 'language', 'intelligent']
- Features: ['Dialogue']

**Summary:**
'A big, red ball that could bounce high and speak a special language...'

**Source:** GPT-4
</details>

## 🚀 Installation

<details>
<summary>📦 Pip Installation</summary>

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
</details>

<details>
<summary>🐍 Conda Installation</summary>

```bash
# clone project
git clone https://github.com/YourGithubName/your-repo-name
cd your-repo-name

# create conda environment and install dependencies
conda env create -f environment.yaml -n myenv

# activate conda environment
conda activate myenv
```
</details>

## 🏃 How to Run

Train model with default configuration:
```bash
# train on CPU
python src/train.py trainer=cpu

# train on GPU
python src/train.py trainer=gpu
```

Train with specific experiment configuration:
```bash
python src/train.py experiment=experiment_name.yaml
```

Override parameters from command line:
```bash
python src/train.py trainer.max_epochs=20 data.batch_size=64
```