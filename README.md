# OpenRE
The source code of Relational Siamese Network

# Preparation
You only need TensorFlow(>=1.12.0) to run this code.
`pip install tensorflow-gpu==1.12`<br>
Then you need to download data from
https://drive.google.com/file/d/10Q7M209DVO5zWUEZ1Ir1pbCNZq3ZxZzg/view?usp=sharing
The data is already preprocessed from the original FewRel and Glove data. Unzip it and get your very easy start.

# Running CNN
A standard CNN for supervised RE can be trained by
`cd CNN`<br>
`python train_CNN.py`<br>

# Running RSN
The Relational Siamese Network for OpenRE can be trained by
`cd RSN`<br>
`python train_RSN.py`<br>
By default it will be trained as a semi-supervised RSN. A supervised RSN without utilizing unlabeled data can be trained by
`python train_RSN.py --trainset_loss_type cross --testset_loss_type none`<br>

# Future Work
The FewRel-distant will be released in the future. I'll add models and data when they are available.