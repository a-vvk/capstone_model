capstone model setup

whats where:

~/capstone_model/
├── train.py       — run this to train, config is at the top
├── model.py       — the neural net architecture
├── dataset.py     — loads the pickle files and batches them
├── best_model.pt  — saved weights after training
└── run_history.json — logs every training run

~/MISA/datasets/MOSEI/
├── train_bert.pkl   — training data (2000 samples)
├── dev_bert.pkl     — validation (300 samples)
├── test_bert.pkl    — test (500 samples)
└── *.csd.bak        — raw dataset files, dont touch these


-----setup on a new machine-----

bash
# 1. install conda if you dont have it
brew install --cask miniconda

2. create env
conda create -n multimodal python=3.11
conda activate multimodal

3. install everything
pip install torch transformers==4.28.0 pytest==7.4.0 gensim==4.3.3 scipy==1.10.1
pip install numpy matplotlib pandas seaborn scikit-learn tensorboardx
pip install h5py validators tqdm colorama huggingface_hub "setuptools<68"

4. clone the SDK (dont pip install it, it breaks)
git clone https://github.com/CMU-MultiComp-Lab/CMU-MultimodalSDK.git ~/CMU-MultimodalSDK

Note: use python 3.11 not 3.14, 3.14 breaks everything

-----running it-----

bash
conda activate multimodal
cd ~/capstone_model
python train.py


takes about 60-90 mins. each epoch prints progress. results print at the end and save to `results.txt` and `run_history.json`.

to change hyperparameters edit the `CONFIG` dict at the top of `train.py`.

-----regenerating data-----

if you need to change the training set size or re-extract features, theres two steps:

1. generate base pickles from the .csd files (use the notebook code from our chat history)
2. run distilbert extraction on them (also in the chat history)

the _bert.pkl files are whats actually used for training. the base .pkl files are intermediate.

-----Extra stuff-----

- CMU server is dead, all data comes from huggingface mirror
- the .csd files are renamed to .csd.bak on purpose — if they have .csd extension the code tries to realign everything and crashes on 8gb ram
- if you get `weights_only` errors from torch.load, add `weights_only=False`
- visual features are 713-dim not 35 (the mirror has different features than the paper describes)