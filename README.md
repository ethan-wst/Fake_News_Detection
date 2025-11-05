 
# Fake News Detection

This repository contains code and notebooks for a fake news detection project using the LIAR dataset. 

## Contents

- `CS4371_Proj.ipynb` — main working notebook
- `liar_dataset/` — expected location for LIAR dataset TSV files (`train.tsv`, `valid.tsv`, `test.tsv`)
- `legacy_code/` — older scripts from original repo for reference
  - `classifier.py` — example classifier script
  - `prediction.py` — example prediction script


## Prerequisites

- Conda or plain `venv` + `pip`

Install requirements as listed in `requirements.txt`

## Current Work

- Ethan West
  - Main notebook `CS4371_Proj.ipynb` contains the latest code and experiments.
    - Implemented some data cleaning and preperation for feeding into CNN
    - Using one hot encoding for catagories and affiliation
    - Mapped labels to a probability

## Future Work

- Need to implement CNN training and inference scripts (hard)
- Implement threshold/bin that takes CNN probability output (moderate)
- Create some kind of dashboard for demo
  - Intake a small dataset of values and some user defined variables, outputs three datasets (kept, review, reject)


## Comments

- We need to determine what attributes we want to use, currently using the label, statement, topic, and affiliation. More attributes will make for a more complex model. The choosen attributes are those with the most consistency, other attributes (ei. speaker name, position, or orgin) may be hard to encode for a CNN as they are mostly unique and will have no pattern for a CNN to recognize/train for.