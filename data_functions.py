import os
import torch
from torch.utils.data import DataLoader, RandomSampler, TensorDataset, Dataset
import pandas as pd
from tqdm import tqdm

from Vocab import SENTENCE_START, SENTENCE_END, PAD_TOKEN, UNKNOWN_TOKEN, START_DECODING, STOP_DECODING
from Vocab import article2ids, abstract2ids, Vocab

from multiprocessing import Manager

import numpy as np

class TestTensorDataset(Dataset):
    r"""Dataset wrapping tensors.

    Each sample will be retrieved by indexing tensors along the first dimension.

    Arguments:
        *tensors (Tensor): tensors that have the same size of the first dimension.
    """

    def __init__(self, *tensors, titles, oovs):
        assert all(tensors[0].size(0) == tensor.size(0) for tensor in tensors)
        manager = Manager()
        self.tensors = manager.list(tensors)
        self.titles = titles
        self.oovs = oovs

    def __getitem__(self, index):
        tensor = tuple(tensor[index] for tensor in self.tensors)
        title = self.titles[index]
        oov = self.oovs[index]
        return tensor, title, oov

    def __len__(self):
        return self.tensors[0].size(0)

class Example(object):
    '''
        Example:
            self.article
            self.title
            self.encoder_input
            self.encoder_mask
            self.encoder_input_with_oov
            self.decoder_input
            self.decoder_mask
            self.decoder_target
            self.decoder_target_with_oov
            self.article_oovs
            self.oov_len
    '''
    def __init__(self, article, title,
                 encoder_input, decoder_input, decoder_target,
                 encoder_input_with_oov, article_oovs, decoder_target_with_oov,
                 max_encoder_len, max_decoder_len, pad_idx=0):

        # articles & titles
        assert len(decoder_input) == len(decoder_target)
        self.article = article  # str
        self.title = title # str

        self.encoder_input ,self.encoder_mask = \
            self._add_pad_and_gene_mask(encoder_input, max_encoder_len, pad_idx=pad_idx)
        self.encoder_input_with_oov = \
            self._add_pad_and_gene_mask(encoder_input_with_oov, max_encoder_len, pad_idx=pad_idx, return_mask=False)

        self.decoder_input, self.decoder_mask = \
            self._add_pad_and_gene_mask(decoder_input, max_decoder_len, pad_idx=pad_idx)
        self.decoder_target = \
            self._add_pad_and_gene_mask(decoder_target, max_decoder_len, pad_idx=pad_idx, return_mask=False)
        self.decoder_target_with_oov = \
            self._add_pad_and_gene_mask(decoder_target_with_oov, max_decoder_len, pad_idx=pad_idx, return_mask=False)

        self.article_oovs = article_oovs
        self.oov_len = len(article_oovs)

    @classmethod
    def _add_pad_and_gene_mask(cls, x, max_len, pad_idx=0, return_mask=True):
        pad_len = max_len - len(x)
        assert pad_len >= 0
        if return_mask:
            mask = [1]*len(x)
            mask.extend([0] * pad_len)
            assert len(mask) == max_len

        x.extend([pad_idx] * pad_len)
        assert  len(x) == max_len

        if return_mask:
            return x, mask
        else:
            return x


def from_sample_covert_example(vocab, article, title, max_article_len, max_title_len,
                                use_pointer=True, print_details=False):
    if 0 == len(title) or 0 == len(article):
        return None

    if len(article) <= len(title):
        return None

    if len(article) > max_article_len:
        article = article[:max_article_len]

    encoder_input = [vocab.word2id(word) for word in article]
    # add <bos> and <eos>
    title = [START_DECODING] + title + [STOP_DECODING]
    # trunc when reach max_len
    title = title[:max_title_len+1]
    title_idx = [vocab.word2id(word) for word in title]
    decoder_input = title_idx[:-1]
    decoder_target = title_idx[1:]
    assert len(decoder_target) == len(decoder_input) <= max_title_len

    encoder_input_with_oov = None
    decoder_target_with_oov = None
    article_oovs = None

    if use_pointer:
        encoder_input_with_oov, article_oovs = article2ids(article, vocab)
        decoder_target_with_oov = abstract2ids(title[1:], vocab, article_oovs)


    example = Example(
        article = article,
        title = title,
        encoder_input = encoder_input,
        decoder_input = decoder_input,
        decoder_target = decoder_target,
        encoder_input_with_oov = encoder_input_with_oov,
        decoder_target_with_oov = decoder_target_with_oov,
        article_oovs = article_oovs,
        max_encoder_len = max_article_len,
        max_decoder_len = max_title_len,
        pad_idx=0
    )
    if print_details:
        print("encoder_input :[{}]".format(" ".join([str(i) for i in example.encoder_input])))
        print("encoder_mask  :[{}]".format(" ".join([str(i) for i in example.encoder_mask])))
        print("encoder_input_with_oov :[{}]".format(" ".join([str(i) for i in example.encoder_input_with_oov])))
        print("decoder_input :[{}]".format(" ".join([str(i) for i in example.decoder_input])))
        print("decoder_mask  :[{}]".format(" ".join([str(i) for i in example.decoder_mask])))
        print("decoder_target  :[{}]".format(" ".join([str(i) for i in example.decoder_target])))
        print("decoder_target_with_oov  :[{}]".format(" ".join([str(i) for i in example.decoder_target_with_oov])))
        print("oovs          :[{}]".format(" ".join(example.article_oovs)))
        print("\n")

    return example

def get_example_loader(data_path, vocab_path, max_article_len,
                       max_title_len, use_pointer, test_mode=False, test_num=1000):
    assert os.path.exists(data_path)
    assert os.path.exists(vocab_path)

    #TODO assume operating csv files
    # load datas and vocab
    print("[INFO] loading datas...")
    df = pd.read_csv(data_path)
    articles = df['articles'].tolist()
    titles = df['titles'].tolist()
    print("[INFO] loading vocab...")
    vocab = Vocab(vocab_path)

    example_loader = []

    # print_details = True if test_mode else False
    print_details=False

    for art, tit in tqdm(zip(articles, titles)):
        try:
            art = art.strip().split()
            tit = tit.strip().split()
        except:
            print(f"error with empty article.")
            continue
        example = from_sample_covert_example(
            vocab=vocab,
            article=art,
            title=tit,
            max_article_len=max_article_len,
            max_title_len=max_title_len,
            use_pointer=use_pointer,
            print_details=print_details
        )
        if example != None:
            example_loader.append(example)
        if test_mode:
            if len(example_loader) == test_num:
                break


    print("[INFO] all datas has been load...")
    print("[INFO] {} examples in total...".format(len(example_loader)))

    return example_loader

def covert_loader_to_dataset(example_loader):
    all_encoder_input = torch.tensor(np.array([ex.encoder_input for ex in example_loader]), dtype=torch.long)
    all_encoder_mask = torch.tensor(np.array([ex.encoder_mask for ex in example_loader]),dtype = torch.long)

    all_decoder_input = torch.tensor(np.array([ex.decoder_input for ex in example_loader]),dtype=torch.long)
    all_decoder_mask = torch.tensor(np.array([ex.decoder_mask for ex in example_loader]),dtype=torch.int)

    all_decoder_target = torch.tensor(np.array([ex.decoder_target for ex in example_loader]),dtype=torch.long)

    all_encoder_input_with_oov = torch.tensor(np.array([ex.encoder_input_with_oov for ex in example_loader]),dtype=torch.long )
    all_decoder_target_with_oov = torch.tensor(np.array([ex.decoder_target_with_oov for ex in example_loader]),dtype=torch.long )
    all_oov_len = torch.tensor(np.array([ex.oov_len for ex in example_loader]),dtype=torch.int)

    # dataset = MyTensorDataset(all_encoder_input, all_encoder_mask, all_decoder_input, all_decoder_mask,
    #                         all_decoder_target,all_encoder_input_with_oov,all_decoder_target_with_oov,all_oov_len)
    dataset = TensorDataset(all_encoder_input, all_encoder_mask, all_decoder_input, all_decoder_mask,
                            all_decoder_target,all_encoder_input_with_oov,all_decoder_target_with_oov,all_oov_len)

    return dataset

def covert_test_loader_to_dataset(example_loader):
    all_encoder_input = torch.tensor(np.array([ex.encoder_input for ex in example_loader]), dtype=torch.long)
    all_encoder_mask = torch.tensor(np.array([ex.encoder_mask for ex in example_loader]), dtype=torch.long)

    all_decoder_input = torch.tensor(np.array([ex.decoder_input for ex in example_loader]), dtype=torch.long)
    all_decoder_mask = torch.tensor(np.array([ex.decoder_mask for ex in example_loader]), dtype=torch.int)

    all_decoder_target = torch.tensor(np.array([ex.decoder_target for ex in example_loader]), dtype=torch.long)

    all_encoder_input_with_oov = torch.tensor(np.array([ex.encoder_input_with_oov for ex in example_loader]),
                                              dtype=torch.long)
    all_decoder_target_with_oov = torch.tensor(np.array([ex.decoder_target_with_oov for ex in example_loader]),
                                               dtype=torch.long)
    all_oov_len = torch.tensor(np.array([ex.oov_len for ex in example_loader]), dtype=torch.int)

    titles = np.array([f.title for f in example_loader])
    oovs = np.array([f.article_oovs for f in example_loader])

    dataset = TestTensorDataset(all_encoder_input, all_encoder_mask, all_decoder_input, all_decoder_mask,
                              all_decoder_target, all_encoder_input_with_oov, all_decoder_target_with_oov, all_oov_len, titles=titles, oovs=oovs)
    return dataset

def from_test_batch_get_model_input(batch, hidden_dim, use_pointer=True, use_coverage=True):
    (all_encoder_input, all_encoder_mask, all_decoder_input, all_decoder_mask,all_decoder_target,\
    all_encoder_input_with_oov, all_decoder_target_with_oov, all_oov_len), title, oov = batch

    max_encoder_len = all_encoder_mask.sum(dim=-1).max()
    max_decoder_len = all_decoder_mask.sum(dim=-1).max()

    all_encoder_input = all_encoder_input[:,:max_encoder_len]
    all_encoder_mask = all_encoder_mask[:,:max_encoder_len]
    all_decoder_input = all_decoder_input[:,:max_decoder_len]
    all_decoder_mask = all_decoder_mask[:,:max_decoder_len]
    all_decoder_target = all_decoder_target[:,:max_decoder_len]
    all_encoder_input_with_oov = all_encoder_input_with_oov[:,:max_encoder_len]
    all_decoder_target_with_oov = all_decoder_target_with_oov[:,:max_decoder_len]

    batch_size = all_encoder_input.shape[0]
    max_oov_len = all_oov_len.max().item()

    oov_zeros = None
    if use_pointer:                # when using pointer, decoder_target should be oov version
        all_decoder_target = all_decoder_target_with_oov
        if max_oov_len > 0:                # when using pointer and the oov exist, the oov_zeros are not None
            oov_zeros = torch.zeros((batch_size, max_oov_len),dtype= torch.float32)
    else:                                  # when not using pointer, it is not necessary to build all_decoder_target_with_oov and oov_zeros
        all_encoder_input_with_oov = None


    init_coverage = None
    if use_coverage:
        init_coverage = torch.zeros(all_encoder_input.size(),dtype=torch.float32)          # float32

    init_context_vec = torch.zeros((batch_size, 2 * hidden_dim),dtype=torch.float32)   # float32

    model_input = [all_encoder_input,all_encoder_mask,all_encoder_input_with_oov,oov_zeros,init_context_vec,
                   init_coverage,all_decoder_input,all_decoder_mask,all_decoder_target]
    model_input = [t.cuda() if t is not None else None for t in model_input]
    return model_input, title, oov


def from_batch_get_model_input(batch, hidden_dim, use_pointer=True, use_coverage=True):
    all_encoder_input, all_encoder_mask, all_decoder_input, all_decoder_mask,all_decoder_target,\
    all_encoder_input_with_oov, all_decoder_target_with_oov, all_oov_len = batch

    max_encoder_len = all_encoder_mask.sum(dim=-1).max()
    max_decoder_len = all_decoder_mask.sum(dim=-1).max()

    all_encoder_input = all_encoder_input[:,:max_encoder_len]
    all_encoder_mask = all_encoder_mask[:,:max_encoder_len]
    all_decoder_input = all_decoder_input[:,:max_decoder_len]
    all_decoder_mask = all_decoder_mask[:,:max_decoder_len]
    all_decoder_target = all_decoder_target[:,:max_decoder_len]
    all_encoder_input_with_oov = all_encoder_input_with_oov[:,:max_encoder_len]
    all_decoder_target_with_oov = all_decoder_target_with_oov[:,:max_decoder_len]

    batch_size = all_encoder_input.shape[0]
    max_oov_len = all_oov_len.max().item()

    oov_zeros = None
    if use_pointer:                # when using pointer, decoder_target should be oov version
        all_decoder_target = all_decoder_target_with_oov
        if max_oov_len > 0:                # when using pointer and the oov exist, the oov_zeros are not None
            oov_zeros = torch.zeros((batch_size, max_oov_len),dtype= torch.float32)
    else:                                  # when not using pointer, it is not necessary to build all_decoder_target_with_oov and oov_zeros
        all_encoder_input_with_oov = None


    init_coverage = None
    if use_coverage:
        init_coverage = torch.zeros(all_encoder_input.size(),dtype=torch.float32)          # float32

    init_context_vec = torch.zeros((batch_size, 2 * hidden_dim),dtype=torch.float32)   # float32

    model_input = [all_encoder_input,all_encoder_mask,all_encoder_input_with_oov,oov_zeros,init_context_vec,
                   init_coverage,all_decoder_input,all_decoder_mask,all_decoder_target]
    model_input = [t.cuda() if t is not None else None for t in model_input]
    return model_input

if __name__ == "__main__":
    data_path = "train.csv"
    vocab_path = "vocab"
    max_article_len = 200
    max_title_len = 200
    batch_size = 32
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    example_loader=get_example_loader(data_path, vocab_path,
                                      max_article_len, max_title_len,
                                      use_pointer=True, test_mode=False)

    example_dataset = covert_loader_to_dataset(example_loader)
    sampler = RandomSampler(example_dataset) # for training random shuffle
    #sampler = SequentialSampler(example_dataset) # for evaluating sequential loading
    train_dataloader = DataLoader(example_dataset, sampler=sampler, batch_size=batch_size)

    for batch in train_dataloader:
        print(batch)
        model_input = from_batch_get_model_input(batch, 256, device)
        print(model_input)
        break
