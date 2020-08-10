# AutoSUM: Automating Feature Extraction and Multi-user Preference for Entity Summarization
PAKDD'20 paper: "[AutoSUM: Automating Feature Extraction and Multi-user Preference for Entity Summarization](https://arxiv.org/pdf/2005.11888.pdf)"
## ENVIRONMENT AND DEPENDENCY
### Environment
- Ubuntu 16.04
- python 3.5+
- pytorch 1.0.1
- java 8
### Dependency
```python
pip install numpy
pip install tqdm
```
## USAGE
### Train
```linux
git clone git@github.com:WeiDongjunGabriel/ESA.git
cd .../AutoSUM
cd model
python main.py
```
we also provide a commandline tool for training the AutoSUM model, you can also run the following command for more details:
```linux
python main.py -h
```
for example, if you want to train the model in dbpedia, the commands are as follows:
```linux
python main.py \
    --db_name dbpedia \
    --mode train \
    --transE_dim 100 \
    --pred_embedding_dim 100 \
    --lr 0.0001 \
    --clip 50 \
    --loss_function BCE \
    --regularization False \
    --n_epoch 50 \
    --save_every 2
```
if you want to test the model and generate entity summarization results, the commands are as follows:
```linux
python main.py \
    --db_name dbpedia \
    --model test \
    --use_epoch 48
```
we also provdie a mode called "all" to train and test the model at the same time, the commands are as follows:
```linux
python main.py \
    --db_name dbpedia \
    --mode all \
    --transE_dim 100 \
    --pred_embedding_dim 100 \
    --lr 0.0001 \
    --clip 50 \
    --loss_function BCE \
    --regularization False \
    --n_epoch 50 \
    --save_every 2 \
    --model test \
    --use_epoch 48
```
### Test
```linux
cd .../AutoSUM
cd test
sh run.sh
```
## CITATION
If you use our code or model, please kindly cite the following paper:

```
@inproceedings{AutoSUM,
  author    = {Dongjun Wei and
               Yaxin Liu and
               Fuqing Zhu and
               Liangjun Zang and
               Wei Zhou and
               Yijun Lu and 
               Songlin Hu},
  title     = {AutoSUM: Automating Feature Extraction and Multi-user Preference for Entity Summarization},
  booktitle = {PAKDD},
  year      = {2020}
}
