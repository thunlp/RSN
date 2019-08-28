# OpenRE
The source code of Relational Siamese Network

# Preparation
You need TensorFlow(>=1.12.0) to run this code.
`pip install tensorflow-gpu==1.12`<br>
Then you need to download the public FewRel dataset from www.zhuhao.me/fewrel.
`cd data-bin`<br>
`wget https://thunlp.oss-cn-qingdao.aliyuncs.com/fewrel/fewrel_train.json`<br>
`wget https://thunlp.oss-cn-qingdao.aliyuncs.com/fewrel/fewrel_val.json`<br>
Then you can preprocess them to get split data.
`python data_split.py`<br>

# environment
`export PATH=/usr/local/cuda10.1/bin:$PATH`<br>
`export PYTHONPATH=your path/ecloud:$PYTHONPATH`


# dataset download and preprocess
Only implement QM8, QM9 and Alchemy auto-process yet.
`cd preprocess`<br>
`python build.py`<br>
`cd ..`

# install pyGPGO (optional for optim_train.py)
`pip install pyGPGO`

# install apex (optional for train -apex 1)
`git clone https://github.com/NVIDIA/apex`<br>
`cd apex`<br>
`pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./`<br>
or<br>
`pip install -v --no-cache-dir ./`
