# Pointer Generator Networks with pytorch_lightning 
This is a ***pytorch lightning*** version of **<u>Pointer Generator Networks</u>** 
------
heavily base on the paper **<a herf=https://arxiv.org/pdf/1704.04368.pdf>Get To The Point: Summarization with Pointer-Generator Networks</a>**

and the transformer version of PGN from **https://github.com/hquzhuguofeng/New-Pointer-Generator-Networks-for-Summarization-Chinese**
## Step 1: Prepare data

Here,  we create data file in ```.csv``` : ```train.csv/valid.csv/test.csv```

and the column names in data file are ```[0, "articles", "titles"]```

build vocab file by ```vocab.py```

```python
 # set the path to training datas & vocab file output path
    data_path = "train.csv"
    vocab_path = "vocab"
#  give the max size of vocab file here set 60000
   	makeVocabdict(data_path, vocab_path, 60000)
```

and launch ```python vocab.py```

## Step 2: Training

filling in all the in ```main.py```

and launch training by ```python main.py```

## Step 3: Decoding

```python main.py -load_model ./path/to/the/checkpoint.chkpt -mode test```

