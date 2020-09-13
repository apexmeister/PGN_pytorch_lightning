import os
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
import random
import argparse
from data_functions import get_example_loader, covert_loader_to_dataset, from_batch_get_model_input, covert_test_loader_to_dataset
from models import PointerGeneratorNetworks
import pytorch_lightning as pl

SEED=1234
random.seed(SEED)
torch.manual_seed(SEED)

class DataLoaderModule(pl.LightningDataModule):
    def __init__(self, opt):
        super(DataLoaderModule, self).__init__()
        self.batch_size = opt.batch_size
        self.train_path = opt.train_path
        self.valid_path = opt.valid_path
        self.test_path = opt.test_path
        self.vocab_path = opt.vocab_path
        self.max_article_len = opt.max_article_len
        self.max_title_len = opt.max_title_len
        self.use_pointer = opt.use_pointer

    def setup(self, stage):
        if stage == 'fit':
            assert os.path.isfile(self.train_path)
            example_loader = get_example_loader(self.train_path, self.vocab_path, self.max_article_len,
                                                self.max_title_len, self.use_pointer, test_mode=True, test_num=10000)
            self.train_dataset = covert_loader_to_dataset(example_loader)
            self.train_sampler = RandomSampler(self.train_dataset)

            assert os.path.isfile(self.valid_path)
            example_loader = get_example_loader(self.valid_path, self.vocab_path, self.max_article_len,
                                                self.max_title_len, self.use_pointer, test_mode=True, test_num=500)
            self.valid_dataset = covert_loader_to_dataset(example_loader)
            self.valid_sampler = SequentialSampler(self.valid_dataset)

        if stage == 'test':
            assert os.path.isfile(self.test_path)
            example_loader = get_example_loader(self.test_path, self.vocab_path, self.max_article_len,
                                                self.max_title_len, self.use_pointer, test_mode=True, test_num=500)
            self.test_dataset = covert_test_loader_to_dataset(example_loader)
            self.test_sampler = SequentialSampler(self.test_dataset)

    def train_dataloader(self):
        train_loader = DataLoader(self.train_dataset, sampler=self.train_sampler, batch_size=self.batch_size)
        return train_loader

    def val_dataloader(self):
        valid_loader = DataLoader(self.valid_dataset, sampler=self.valid_sampler, batch_size=self.batch_size)
        return valid_loader

    def test_dataloader(self):
        test_loader = DataLoader(self.test_dataset, sampler=self.test_sampler, batch_size=1)
        return test_loader

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-mode", type=str, default="train")

    # ===============path config=================
    parser.add_argument("-train_path", type=str,
                        default="/home/disk2/lyj2019/workspace/my_paper/dataset/SAMsum/finished_csv_datas/train.csv")
    parser.add_argument("-valid_path", type=str,
                        default="/home/disk2/lyj2019/workspace/my_paper/dataset/SAMsum/finished_csv_datas/valid.csv")
    parser.add_argument("-test_path", type=str,
                        default="/home/disk2/lyj2019/workspace/my_paper/dataset/SAMsum/finished_csv_datas/test.csv")
    parser.add_argument("-vocab_path", type=str,
                        default="/home/disk2/lyj2019/workspace/my_paper/dataset/SAMsum/finished_csv_datas/vocab")
    parser.add_argument("-save_mode", type=str, default="best")
    parser.add_argument("-load_model", type=str, default=None)
    parser.add_argument("-result_dir", type=str,
                        default="/home/disk2/lyj2019/workspace/my_paper/PGN_TR_PLver/result/pred_SAMsum.txt")

    # ===============model config=================
    parser.add_argument("-n_heads", type=int, default=4)
    parser.add_argument("-vocab_size", type=int, default=50000)
    parser.add_argument("-n_layers", type=int, default=1)
    parser.add_argument("-encoder_lstm_num_layer", type=int, default=1)
    parser.add_argument("-decoder_lstm_num_layer", type=int, default=1)
    parser.add_argument("-hidden_dim", type=int, default=256)
    parser.add_argument("-linear_dim", type=int, default=512)
    parser.add_argument("-intermediate_dim", type=int, default=512)
    parser.add_argument("-max_article_len", type=int, default=400)
    parser.add_argument("-max_title_len", type=int, default=100)
    parser.add_argument("-min_title_len", type=int, default=15)

    parser.add_argument("-pad_idx", type=int, default=0)
    parser.add_argument("-unk_idx", type=int, default=1)
    parser.add_argument("-start_idx", type=int, default=2)
    parser.add_argument("-stop_idx", type=int, default=3)

    parser.add_argument("-eps", type=float, default=1e-08)
    parser.add_argument("-dropout", type=float, default=0.1)
    parser.add_argument("-coverage_loss_weight", type=float, default=1.0)
    parser.add_argument("-init_lr", type=float, default=1.0)
    parser.add_argument("-max_grad_norm", type=float, default=5.0)

    parser.add_argument("-use_pointer", type=bool, default=True)
    parser.add_argument("-use_coverage", type=bool, default=True)


    # ===============training&testing config=================
    parser.add_argument("-batch_size", type=int, default=28)
    parser.add_argument("-beam_size", type=int, default=4)
    parser.add_argument("-epoch", type=int, default=10)
    parser.add_argument("-max_epoch", type=int, default=100)
    parser.add_argument("-n_warmup_steps", type=int, default=2000)

    opt = parser.parse_args()
    print(opt)
    # ==============setup Training or testing================
    print("[INFO] Setting model and dataloader")
    model = PointerGeneratorNetworks(opt).cuda()
    DataModule = DataLoaderModule(opt)
    if opt.load_model:
        trainer = pl.Trainer(resume_from_checkpoint=opt.load_model, gpus=2, max_epochs=opt.max_epoch)
    else:
        trainer = pl.Trainer(gpus=2, max_epochs=opt.epoch)

    if opt.mode == "train":
        trainer.fit(model=model, datamodule=DataModule)
    elif opt.mode == "test":
        trainer.test(model=model, datamodule=DataModule)
    else:
        print("[MODE ERROR] mode should be in [\"train\", \"test\"]")

if __name__ == '__main__':
    main()
