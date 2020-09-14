import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch.nn.functional as F
import torch.optim as Optim
from data_functions import from_batch_get_model_input, from_test_batch_get_model_input
from Vocab import Vocab
import math
import pytorch_lightning as pl
from rouge import Rouge
import numpy
import numpy as np


# ===============define some function================
class Beam(object):
    def __init__(self,tokens,log_probs,status,context_vec,coverage):
        self.tokens = tokens
        self.log_probs = log_probs
        self.status = status
        self.context_vec = context_vec
        self.coverage = coverage

    def update(self,token,log_prob,status,context_vec,coverage):
        return Beam(
            tokens = self.tokens + [token],
            log_probs = self.log_probs + [log_prob],
            status = status,
            context_vec = context_vec,
            coverage = coverage
        )


    def latest_token(self):
        return self.tokens[-1]

    @property
    def avg_log_prob(self):
        #print(sum(self.log_probs) / len(self.tokens))
        return sum(self.log_probs) / len(self.tokens)

def sort_beams(beams):
    return sorted(beams, key=lambda beam:beam.avg_log_prob, reverse=True)

def idx_to_token(idx, oov_word, vocab):
    if idx < vocab.get_vocab_size():
        return vocab.id2word(idx)
    else:
        idx = idx - vocab.get_vocab_size()

        if idx < len(oov_word):
            return oov_word[idx]
        else:
            return "<unk>"


# ============define component of model==============

BertLayerNorm = torch.nn.LayerNorm

def gelu(x):
    """Implementation of the gelu activation function.
        For information: OpenAI GPT's gelu is slightly different (and gives slightly different results):
        0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
        Also see https://arxiv.org/abs/1606.08415
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))

ACT2FN = gelu

class BertEmbeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings.
    """

    def __init__(self, vocab_size, hidden_dim, pad_idx, max_article_len, dropout, eps):
        super(BertEmbeddings, self).__init__()
        self.word_embeddings = nn.Embedding(vocab_size, hidden_dim, padding_idx=pad_idx)
        self.position_embeddings = nn.Embedding(max_article_len, hidden_dim)

        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.LayerNorm = BertLayerNorm(hidden_dim, eps=eps)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_ids, position_ids=None):
        if input_ids.shape == torch.Size([0]):
            return
        seq_length = input_ids.size(1)
        if position_ids is None:
            position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
            position_ids = position_ids.unsqueeze(0).expand_as(input_ids)

        words_embeddings = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)

        embeddings = words_embeddings + position_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings

class BertSelfAttention(nn.Module):
    def __init__(self, hidden_dim, n_heads, dropout):
        super(BertSelfAttention, self).__init__()
        if hidden_dim % n_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (hidden_dim, n_heads))

        self.n_heads = n_heads
        self.attention_head_size = int(hidden_dim / n_heads)
        self.all_head_size = self.n_heads * self.attention_head_size

        self.query = nn.Linear(hidden_dim, self.all_head_size)
        self.key = nn.Linear(hidden_dim, self.all_head_size)
        self.value = nn.Linear(hidden_dim, self.all_head_size)

        self.dropout = nn.Dropout(dropout)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.n_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, attention_mask):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        # Apply the attention mask is (precomputed for all layers in BertModel forward() function)

        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)

        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and -10000.0 for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        attention_scores = attention_scores + extended_attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        return context_layer

class BertSelfOutput(nn.Module):
    def __init__(self, hidden_dim, dropout, eps):
        super(BertSelfOutput, self).__init__()
        self.dense = nn.Linear(hidden_dim, hidden_dim)
        self.LayerNorm = BertLayerNorm(hidden_dim, eps=eps)
        self.dropout = nn.Dropout(dropout)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states

class BertAttention(nn.Module):
    def __init__(self, hidden_dim, n_heads, dropout, eps):
        super(BertAttention, self).__init__()
        self.self = BertSelfAttention(hidden_dim, n_heads, dropout)
        self.output = BertSelfOutput(hidden_dim, dropout, eps)
        self.pruned_heads = set()

    def forward(self, input_tensor, attention_mask):
        self_outputs = self.self(input_tensor, attention_mask)
        attention_output = self.output(self_outputs, input_tensor)
        outputs = attention_output  # add attentions if we output them
        return outputs

class BertIntermediate(nn.Module):
    def __init__(self, hidden_dim, intermediate_dim):
        super(BertIntermediate, self).__init__()
        self.dense = nn.Linear(hidden_dim, intermediate_dim)
        self.intermediate_act_fn = ACT2FN

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states

class BertOutput(nn.Module):
    def __init__(self, intermediate_dim, hidden_dim, dropout, eps):
        super(BertOutput, self).__init__()
        self.dense = nn.Linear(intermediate_dim, hidden_dim)
        self.LayerNorm = BertLayerNorm(hidden_dim, eps=eps)
        self.dropout = nn.Dropout(dropout)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states

class BertLayer(nn.Module):
    def __init__(self, hidden_dim, intermediate_dim, n_heads, dropout, eps):
        super(BertLayer, self).__init__()
        self.attention = BertAttention(hidden_dim, n_heads, dropout, eps)
        self.intermediate = BertIntermediate(hidden_dim, intermediate_dim)
        self.output = BertOutput(intermediate_dim, hidden_dim, dropout, eps)

    def forward(self, hidden_states, attention_mask):
        attention_output = self.attention(hidden_states, attention_mask)
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output

class Encoder(nn.Module):
    def __init__(self, n_heads, encoder_lstm_num_layer, vocab_size, hidden_dim, intermediate_dim,
                 pad_idx, max_article_len, dropout, eps):
        super(Encoder, self).__init__()

        self.embedding = BertEmbeddings(vocab_size, hidden_dim, pad_idx, max_article_len, dropout, eps)
        self.bertLayer = BertLayer(hidden_dim, intermediate_dim, n_heads, dropout, eps)

        self.lstm = nn.LSTM(input_size=hidden_dim,
                            hidden_size=hidden_dim,
                            num_layers=encoder_lstm_num_layer,
                            dropout=dropout,
                            batch_first=True,
                            bidirectional=True)

        self.dropout = nn.Dropout(p=dropout)

        self.pad_idx = pad_idx

    def forward(self, x, mask):
        embedded = self.dropout(self.embedding(x))

        bert_output = self.bertLayer(hidden_states=embedded, attention_mask=mask)
        bool_mask = (mask == 0)

        bert_output_masked = bert_output.masked_fill(mask=bool_mask.unsqueeze(dim=2), value=0)

        seq_lens = mask.sum(dim=-1)
        packed = pack_padded_sequence(input=bert_output_masked, lengths=seq_lens, batch_first=True,
                                      enforce_sorted=False)
        output_packed, (h, c) = self.lstm(packed)

        # pad_packed_sequence(sequence, batch_first=False, padding_value=0.0, total_length=None):

        output, _ = pad_packed_sequence(sequence=output_packed,
                                        batch_first=True,
                                        padding_value=self.pad_idx,
                                        total_length=seq_lens.max())

        return output, (h, c)

class Reduce(nn.Module):
    def __init__(self, hidden_dim, dropout):
        super(Reduce, self).__init__()

        self.hidden_dim = hidden_dim

        self.reduce_h = nn.Linear(hidden_dim * 2, hidden_dim)
        self.reduce_c = nn.Linear(hidden_dim * 2, hidden_dim)

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, h, c):
        assert self.hidden_dim == h.shape[2]
        assert self.hidden_dim == c.shape[2]

        h = h.reshape(-1, self.hidden_dim * 2)
        c = c.reshape(-1, self.hidden_dim * 2)

        h_output = self.dropout(self.reduce_h(h))
        c_output = self.dropout(self.reduce_c(c))

        h_output = F.relu(h_output)
        c_output = F.relu(c_output)

        # h_output.shape = c_output.shape =
        # (batch,hidden)
        return h_output.unsqueeze(0), c_output.unsqueeze(0)

class Attention(nn.Module):
    def __init__(self, hidden_dim, dropout, use_coverage=True):

        # hidden_dim, use_coverage = False
        super(Attention, self).__init__()

        self.w_s = nn.Linear(hidden_dim * 2, hidden_dim * 2, bias=False)

        if use_coverage:
            self.w_c = nn.Linear(1, hidden_dim * 2)

        self.v = nn.Linear(hidden_dim * 2, 1, bias=False)

        self.dropout = nn.Dropout(p=dropout)

        self.use_coverage = use_coverage

    # h ：encoder hidden states h_i ,On each step t. (batch,seq_len,hidden*2)
    # mask, 0-1 encoder_mask (batch,seq_len)
    # s : decoder state s_t,one step (batch,hidden*2)
    # coverage : sum of attention score (batch,seq_len)
    def forward(self, encoder_features, mask, s, coverage):

        decoder_features = self.dropout(self.w_s(s).unsqueeze(1))  # (batch,1,hidden*2)

        # broadcast
        attention_feature = encoder_features + decoder_features  # (batch,seq_len,hidden*2)

        if self.use_coverage:
            coverage_feature = self.dropout(self.w_c(coverage.unsqueeze(2)))  # (batch,seq_len,hidden*2)
            attention_feature += coverage_feature

        e_t = self.dropout(self.v(torch.tanh(attention_feature)).squeeze(dim=2))  # (batch,seq_len)

        mask_bool = (mask == 0)  # mask pad position eq True

        e_t.masked_fill_(mask=mask_bool, value=-float('inf'))

        a_t = torch.softmax(e_t, dim=-1)  # (batch,seq_len)
        # print("a_t: ",a_t)
        if self.use_coverage:
            next_coverage = coverage + a_t
        # print("coverage: ",coverage)
        # print("next_coverage: ",next_coverage)

        return a_t, next_coverage

class GeneraProb(nn.Module):
    def __init__(self, hidden_dim, dropout):
        super(GeneraProb, self).__init__()

        self.w_h = nn.Linear(hidden_dim * 2, 1)
        self.w_s = nn.Linear(hidden_dim * 2, 1)
        self.w_x = nn.Linear(hidden_dim, 1)
        self.dropout = nn.Dropout(p=dropout)

    # h : weight sum of encoder output ,(batch,hidden*2)
    # s : decoder state                 (batch,hidden*2)
    # x : decoder input                 (batch,embed)
    def forward(self, h, s, x):
        h_feature = self.dropout(self.w_h(h))  # (batch,1)
        s_feature = self.dropout(self.w_s(s))  # (batch,1)
        x_feature = self.dropout(self.w_x(x))  # (batch,1)

        gen_feature = h_feature + s_feature + x_feature  # (batch,1)

        gen_p = torch.sigmoid(gen_feature)

        return gen_p

class Decoder(nn.Module):
    def __init__(self, n_heads, decoder_lstm_num_layer, hidden_dim, vocab_size,
                 intermediate_dim, dropout, eps, use_pointer, use_coverage):
        super(Decoder, self).__init__()

        self.bertLayer = BertLayer(hidden_dim, intermediate_dim, n_heads, dropout, eps)

        self.get_lstm_input = nn.Linear(in_features=hidden_dim * 2 + hidden_dim,
                                        out_features=hidden_dim)

        self.lstm = nn.LSTM(input_size=hidden_dim,
                            hidden_size=hidden_dim,
                            num_layers=decoder_lstm_num_layer,
                            dropout=dropout,
                            batch_first=True,
                            bidirectional=False)

        self.attention = Attention(hidden_dim, dropout, use_coverage)

        if use_pointer:
            self.genera_prob = GeneraProb(hidden_dim, dropout)

        self.dropout = nn.Dropout(p=dropout)

        # self.out = nn.Sequential(nn.Linear(in_features=hidden_dim * 3,out_features=embed_dim),
        #                          self.dropout(),
        #                          nn.ReLU(),
        #                          nn.Linear(in_features=hidden_dim,out_features=vob_size),
        #                          self.dropout)
        self.out = nn.Sequential(nn.Linear(in_features=hidden_dim * 3, out_features=hidden_dim),
                                 nn.ReLU(),
                                 nn.Linear(in_features=hidden_dim, out_features=vocab_size))

        self.use_pointer = use_pointer
        self.use_coverage = use_coverage

    # decoder_input_one_step (batch,t)   (t = step +1)
    # decoder_status = (h_t,c_t)  h_t (1,batch,hidden)
    # encoder_output (batch,seq_len,hidden*2)
    # encoder_mask (batch,seq_len)
    # context_vec (bach,hidden*2)  encoder weight sum about attention score
    # oovs_zero (batch,max_oov_size)  all-zero tensor
    # encoder_with_oov (batch,seq_len)  Index of words in encoder_with_oov can be greater than vob_size
    # coverage : Sum of attention at each step
    # step...
    def forward(self, decoder_embed_one_step, decoder_mask_one_step, decoder_status, encoder_output, encoder_features,
                encoder_mask, context_vec, oovs_zero, encoder_with_oov, coverage, step):

        bert_output = self.bertLayer(hidden_states=decoder_embed_one_step, attention_mask=decoder_mask_one_step)
        bool_mask = (decoder_mask_one_step == 0)
        bert_output_masked = bert_output.masked_fill(mask=bool_mask.unsqueeze(dim=2), value=0)
        bert_output_masked_last_step = bert_output_masked[:, -1, :]
        # print("context vector: ",context_vec)
        # print("bert output mask: ",bert_output_masked_last_step.size())
        x = self.get_lstm_input(torch.cat([context_vec, bert_output_masked_last_step], dim=-1)).unsqueeze(
            dim=1)  # (batch,1,hidden*2+embed_dim)

        decoder_output, next_decoder_status = self.lstm(x, decoder_status)

        h_t, c_t = next_decoder_status

        batch_size = c_t.shape[1]

        h_t_reshape = h_t.reshape(batch_size, -1)
        c_t_reshape = c_t.reshape(batch_size, -1)

        status = torch.cat([h_t_reshape, c_t_reshape], dim=-1)  # (batch,hidden_dim*2)

        # attention_score (batch,seq_len)  Weight of each word vector
        # next_coverage (batch,seq_len)  sum of attention_score

        attention_score, next_coverage = self.attention(encoder_features=encoder_features,
                                                        mask=encoder_mask,
                                                        s=status,
                                                        coverage=coverage)

        # (batch,hidden_dim*2)  encoder_output weight sum about attention_score
        # current_context_vec = torch.bmm(attention_score.unsqueeze(1),encoder_output).squeeze()
        # same as above
        current_context_vec = torch.einsum("ab,abc->ac", attention_score, encoder_output)

        # (batch,1)
        genera_p = None
        if self.use_pointer:
            genera_p = self.genera_prob(h=current_context_vec,
                                        s=status,
                                        x=x.squeeze())

        # (batch,hidden_dim*3)
        out_feature = torch.cat([decoder_output.squeeze(dim=1), current_context_vec], dim=-1)

        # (batch,vob_size)
        output = self.out(out_feature)

        vocab_dist = torch.softmax(output, dim=-1)

        if self.use_pointer:
            vocab_dist_p = vocab_dist * genera_p
            context_dist_p = attention_score * (1 - genera_p)
            if oovs_zero is not None:
                vocab_dist_p = torch.cat([vocab_dist_p, oovs_zero], dim=-1)
            final_dist = vocab_dist_p.scatter_add(dim=-1, index=encoder_with_oov, src=context_dist_p)
        else:
            final_dist = vocab_dist

        return final_dist, next_decoder_status, current_context_vec, attention_score, genera_p, next_coverage


# =====define model with pytorch_lightning======

class PointerGeneratorNetworks(pl.LightningModule):
    def __init__(self, opt):
        super(PointerGeneratorNetworks, self).__init__()
        self.all_mode = ["train", "valid", "decode"]
        self.encoder = Encoder(opt.n_heads, opt.encoder_lstm_num_layer, opt.vocab_size, opt.hidden_dim, opt.intermediate_dim,
                                   opt.pad_idx, opt.max_article_len, opt.dropout, opt.eps)
        self.decoder = Decoder(opt.n_heads, opt.decoder_lstm_num_layer, opt.hidden_dim, opt.vocab_size,
                                   opt.intermediate_dim, opt.dropout, opt.eps, opt.use_pointer, opt.use_coverage)
        self.w_h = nn.Linear(opt.hidden_dim * 2, opt.hidden_dim * 2, bias=False)  #
        self.reduce = Reduce(opt.hidden_dim, opt.dropout)

        self.decoder_embed = BertEmbeddings(opt.vocab_size, opt.hidden_dim, opt.pad_idx, opt.max_article_len, opt.dropout, opt.eps)
        self.hidden_dim = opt.hidden_dim
        self.max_title_len = opt.max_title_len
        self.min_title_len = opt.min_title_len
        self.eps = opt.eps
        self.use_pointer = opt.use_pointer
        self.use_coverage = opt.use_coverage
        self.coverage_loss_weight = opt.coverage_loss_weight
        self.pad_idx = opt.pad_idx
        self.unk_idx = opt.unk_idx
        self.start_idx = opt.start_idx
        self.stop_idx = opt.stop_idx
        self.vocab_path = opt.vocab_path
        self.vocab_size = opt.vocab_size
        self.beam_size = opt.beam_size

        self.pred_path = opt.result_dir
        self.vocab = Vocab(self.vocab_path, self.vocab_size)
    # ===================forward====================
    def forward(self, encoder_input, encoder_mask, encoder_with_oov,
                oovs_zero, context_vec, coverage,
                decoder_input=None, decoder_mask=None, decoder_target=None,
                mode="train", start_tensor=None, beam_size=4):

        assert mode in self.all_mode
        # loss_list = []
        if mode in ["train", "valid"]:

            model_loss = self._forward(encoder_input, encoder_mask, encoder_with_oov, oovs_zero, context_vec,
                                       coverage,
                                       decoder_input, decoder_mask, decoder_target)
            # print(model_loss)

            return model_loss
        elif mode in ["decode"]:
            return self._decoder(encoder_input=encoder_input, encoder_mask=encoder_mask,
                                 encoder_with_oov=encoder_with_oov, oovs_zero=oovs_zero, context_vec=context_vec,
                                 coverage=coverage, beam_size=beam_size)

    def _forward(self, encoder_input, encoder_mask, encoder_with_oov, oovs_zero, context_vec, coverage,
                 decoder_input, decoder_mask, decoder_target):

        assert isinstance(self.encoder, Encoder)
        encoder_outputs, encoder_hidden = self.encoder(encoder_input, encoder_mask)

        decoder_status = self.reduce(*encoder_hidden)

        encoder_features = self.w_h(encoder_outputs)

        batch_max_decoder_len = decoder_mask.size(1)
        assert batch_max_decoder_len <= self.max_title_len
        # print(batch_max_decoder_len)

        decoder_embed = self.decoder_embed(decoder_input)

        all_step_loss = []
        for step in range(1, batch_max_decoder_len + 1):
            # print("step: ",step)
            decoder_embed_one_step = decoder_embed[:, :step, :]
            decoder_mask_one_step = decoder_mask[:, :step]
            if decoder_mask_one_step.size() == torch.Size([1, 1]):
                decoder_mask_one_step = torch.ones([16, 1]).cuda().int()
            # print("decoder_embed: ",decoder_embed_one_step)
            # print("decoder_mask: ", decoder_mask_one_step)

            final_dist, decoder_status, context_vec, attention_score, genera_p, next_coverage = \
                self.decoder(
                    decoder_embed_one_step=decoder_embed_one_step,
                    decoder_mask_one_step=decoder_mask_one_step,
                    decoder_status=decoder_status,
                    encoder_output=encoder_outputs,
                    encoder_features=encoder_features,
                    encoder_mask=encoder_mask,
                    context_vec=context_vec,
                    oovs_zero=oovs_zero,
                    encoder_with_oov=encoder_with_oov,
                    coverage=coverage,
                    step=step)
            # print("atttention: ",attention_score)
            # print("coverage: ",coverage)
            target = decoder_target[:, step - 1].unsqueeze(1)
            probs = torch.gather(final_dist, dim=-1, index=target).squeeze()
            step_loss = -torch.log(probs + self.eps)

            if self.use_coverage:
                # print("sum: ",torch.sum(torch.min(attention_score, coverage), dim=-1))
                coverage_loss = self.coverage_loss_weight * torch.sum(torch.min(attention_score, coverage),
                                                                      dim=-1)

                step_loss += coverage_loss
                coverage = next_coverage

            all_step_loss.append(step_loss)
            # print("all_step_loss: ",all_step_loss)
            # if all_step_loss == []:
            #    continue

        # if all_step_loss == []:
        #    return
        token_loss = torch.stack(all_step_loss, dim=1)

        decoder_mask_cut = decoder_mask[:, :batch_max_decoder_len].float()
        assert decoder_mask_cut.shape == token_loss.shape

        decoder_lens = decoder_mask.sum(dim=-1)

        token_loss_with_mask = token_loss * decoder_mask_cut
        batch_loss_sum_token = token_loss_with_mask.sum(dim=-1)
        batch_loss_mean_token = batch_loss_sum_token / decoder_lens.float()
        result_loss = batch_loss_mean_token.mean()

        return result_loss

    def _decoder(self, encoder_input, encoder_mask, encoder_with_oov, oovs_zero, context_vec, coverage, beam_size=4):

        encoder_outputs, encoder_hidden = self.encoder(encoder_input, encoder_mask)
        decoder_status = self.reduce(*encoder_hidden)
        # print("decoder_status: ",decoder_status)

        encoder_features = self.w_h(encoder_outputs)

        beams = [Beam(tokens=[self.start_idx], log_probs=[1.0], status=decoder_status, context_vec=context_vec,
                      coverage=coverage)]

        # print("           beams:                ",beams)

        step = 0
        result = []
        last_beam_size = 0
        current_beam_size = 1

        # print("   result:   ",result)
        while step < self.max_title_len and len(result) < 4:
            # print("step: ",step)

            assert len(beams) != 0

            # print(len(beams))
            current_tokens_idx = [
                [token if token < self.vocab_size else self.unk_idx for token in b.tokens]
                for b in beams]
            # print(current_tokens_idx)
            decoder_input_one_step = torch.tensor(current_tokens_idx, dtype=torch.long, device=encoder_outputs.device)

            decoder_embed_one_step = self.decoder_embed(decoder_input_one_step)

            status_h_list = [b.status[0] for b in beams]
            status_c_list = [b.status[1] for b in beams]

            # print("step: ",step)
            # if status_h_list == []:
            #    step += 1
            #    continue
            # print("status_h_list: ",status_h_list)
            decoder_h = torch.cat(status_h_list, dim=1)  # status_h  (num_layers * num_directions, batch, hidden_size)
            decoder_c = torch.cat(status_c_list, dim=1)  # status_c  (num_layers * num_directions, batch, hidden_size)
            decoder_status = (decoder_h, decoder_c)

            context_vec_list = [b.context_vec for b in beams]
            context_vec = torch.cat(context_vec_list, dim=0)  # context_vec (batch,hidden*2)

            if self.use_coverage:
                coverage_list = [b.coverage for b in beams]
                coverage = torch.cat(coverage_list, dim=0)  # coverage (batch,seq_len)
            else:
                coverage = None

            current_beam_size = len(beams)
            if current_beam_size != last_beam_size:
                last_beam_size = current_beam_size

                encoder_outputs_expand = encoder_outputs.expand(current_beam_size, encoder_outputs.size(1),
                                                                encoder_outputs.size(2))
                encoder_mask_expand = encoder_mask.expand(current_beam_size, encoder_mask.shape[1])
                encoder_features_expand = encoder_features.expand(current_beam_size, encoder_features.size(1),
                                                                  encoder_features.size(2))
                if oovs_zero is not None:
                    oovs_zero_expand = oovs_zero.expand(current_beam_size, oovs_zero.shape[1])
                else:
                    oovs_zero_expand = None
                encoder_with_oov_expand = encoder_with_oov.expand(current_beam_size, encoder_with_oov.shape[1])

            decoder_mask_one_step = torch.ones_like(decoder_input_one_step, device=decoder_input_one_step.device)

            final_dist, decoder_status, context_vec, attention_score, genera_p, next_coverage = \
                self.decoder(
                    decoder_embed_one_step=decoder_embed_one_step,
                    decoder_mask_one_step=decoder_mask_one_step,
                    decoder_status=decoder_status,
                    encoder_output=encoder_outputs_expand,
                    encoder_features=encoder_features_expand,
                    encoder_mask=encoder_mask_expand,
                    context_vec=context_vec,
                    oovs_zero=oovs_zero_expand,
                    encoder_with_oov=encoder_with_oov_expand,
                    coverage=coverage,
                    step=step)
            # rint("context_vec:",context_vec)
            # print("next_coverage: ",next_coverage)
            # (B, vob_size)


            log_probs = torch.log(final_dist)
            topk_log_probs, topk_ids = torch.topk(log_probs, beam_size * 2, dim=-1)

            all_beams = []
            for i in range(len(beams)):
                beam = beams[i]
                h_i = decoder_status[0][:, i, :].unsqueeze(1)  # keep dim (num_layers*num_directions,batch,hidden_size)
                c_i = decoder_status[1][:, i, :].unsqueeze(1)  # keep dim (num_layers*num_directions,batch,hidden_size)
                status_i = (h_i, c_i)
                context_vec_i = context_vec[i, :].unsqueeze(0)  # keep dim (batch,hidden*2)
                if self.use_coverage:
                    coverage_i = next_coverage[i, :].unsqueeze(0)  # keep dim (batch,seq_len)
                else:
                    coverage_i = None

                for j in range(beam_size * 2):
                    if topk_ids[i, j] in [self.pad_idx, self.unk_idx]:
                        continue
                    # print(topk_ids[i,j])
                    new_beam = beam.update(token=topk_ids[i, j].item(),
                                           log_prob=topk_log_probs[i, j].item(),
                                           status=status_i,
                                           context_vec=context_vec_i,
                                           coverage=coverage_i)
                    all_beams.append(new_beam)

            beams = []
            for beam in sort_beams(all_beams):

                # if beam.tokens[-1] != self.stop_idx:
                #    beam.tokens.append(self.stop_idx)
                # beam.tokens = [idx for idx in beam.tokens if idx != 3]
                # if len(beam.tokens) != step+2:
                #    beam.tokens += [beam.tokens[-1]]
                # beam.tokens += [3]
                # print(beam.tokens)
                if beam.tokens[-1] == self.stop_idx:
                    # print(len(beam.tokens))
                    if len(beam.tokens) > self.min_title_len:
                        result.append(beam)
                    else:
                        pass
                else:
                    beams.append(beam)
                if beam_size == len(beams) or len(result) == beam_size:
                    break
                # print(len(result))

            # print("beams: ",beams)
            step += 1

        if 0 == len(result):
            result = beams
        # print(len(result))

        sorted_result = sort_beams(result)
        if sorted_result == []:
            return

        return sorted_result[0]


    # ===============setup optimizer================

    def configure_optimizers(self):
        optimizer = Optim.Adam(self.parameters(), lr=1e-3, betas=(0.9, 0.98))
        return optimizer


    # ===============training loop=================

    def training_step(self, batch, batch_idx):
        batch = from_batch_get_model_input(batch, self.hidden_dim,
                                           use_pointer=self.use_pointer, use_coverage=self.use_coverage)
        inputs = {'encoder_input': batch[0],
                'encoder_mask': batch[1],
                'encoder_with_oov': batch[2],
                'oovs_zero': batch[3],
                'context_vec': batch[4],
                'coverage': batch[5],
                'decoder_input': batch[6],
                'decoder_mask': batch[7],
                'decoder_target': batch[8],
                'mode': 'train'}

        loss = self(**inputs)
        result = pl.TrainResult(loss)
        result.log('train_loss', loss, prog_bar=True)
        return result

    # def training_epoch_end(self, training_step_outputs):
    #     avg_train_losses = torch.tensor([x["train_loss"] for x in training_step_outputs]).mean()
    #     return {'train_loss': avg_train_losses}


    # =============validation loop================

    def validation_step(self, batch, batch_idx):
        batch = from_batch_get_model_input(batch, self.hidden_dim,
                                           use_pointer=self.use_pointer, use_coverage=self.use_coverage)
        inputs = {'encoder_input': batch[0],
                  'encoder_mask': batch[1],
                  'encoder_with_oov': batch[2],
                  'oovs_zero': batch[3],
                  'context_vec': batch[4],
                  'coverage': batch[5],
                  'decoder_input': batch[6],
                  'decoder_mask': batch[7],
                  'decoder_target': batch[8],
                  'mode': 'train'}

        loss = self(**inputs)
        result = pl.EvalResult(checkpoint_on=loss)
        result.log('val_loss', loss, prog_bar=True)
        return result

    # def validation_epoch_end(self, validation_step_outputs):
    #     avg_valid_loss = validation_step_outputs['val_loss'].cpu().numpy()
    #     result = pl.EvalResult()
    #     result.log_dict({'val_loss': avg_valid_loss})
    #     return result


    # ===============testing loop=================

    def test_step(self, batch, batch_idx):
        batch_, title, oov = from_test_batch_get_model_input(batch, self.hidden_dim,
                                           use_pointer=self.use_pointer, use_coverage=self.use_coverage)
        beam = self(encoder_input=batch_[0],
                     encoder_mask=batch_[1],
                     encoder_with_oov=batch_[2],
                     oovs_zero=batch_[3],
                     context_vec=batch_[4],
                     coverage=batch_[5],
                     mode="decode",
                     beam_size=self.beam_size
                     )

        current_oovs = [token[0] for token in oov]
        current_title = [word[0] for word in title if word[0] != '<eos>' and word[0] != '<bos>']

        hypothesis_idx_list = beam.tokens[1:]
        if self.stop_idx == hypothesis_idx_list[-1]:
            hypothesis_idx_list = hypothesis_idx_list[:-1]

        hypothesis_token_list = [idx_to_token(index, oov_word=current_oovs, vocab=self.vocab)
                                 for index in hypothesis_idx_list]

        hypothesis_str = " ".join(hypothesis_token_list)
        reference_str = " ".join(current_title)

        result_str = "{}\t|\t{}\n".format(hypothesis_str, reference_str)

        with open(file=self.pred_path, mode='a', encoding='utf-8') as f:
            f.write(result_str)
            f.close()
        rouge = Rouge()
        rouge_score = rouge.get_scores(hypothesis_str, reference_str)
        result = pl.EvalResult()
        result.log('test_rouge_score', rouge_score)
        return result

    def test_epoch_end(self, test_step_outputs):
        test_rouge_score = test_step_outputs.test_rouge_score
        rouge_1 = {"f": 0., 'p': 0., 'r': 0.}
        rouge_2 = {"f": 0., 'p': 0., 'r': 0.}
        rouge_l = {"f": 0., 'p': 0., 'r': 0.}
        for score in test_rouge_score:
            rouge_1["f"] += score[0]["rouge-1"]["f"]
            rouge_1["p"] += score[0]["rouge-1"]["p"]
            rouge_1["r"] += score[0]["rouge-1"]["r"]
            rouge_2["f"] += score[0]["rouge-2"]["f"]
            rouge_2["p"] += score[0]["rouge-2"]["p"]
            rouge_2["r"] += score[0]["rouge-2"]["r"]
            rouge_l["f"] += score[0]["rouge-l"]["f"]
            rouge_l["p"] += score[0]["rouge-l"]["p"]
            rouge_l["r"] += score[0]["rouge-l"]["r"]
        rouge_1["f"] = torch.tensor(rouge_1["f"]/len(test_rouge_score))
        rouge_1["p"] = torch.tensor(rouge_1["p"]/len(test_rouge_score))
        rouge_1["r"] = torch.tensor(rouge_1["r"]/len(test_rouge_score))
        rouge_2["f"] = torch.tensor(rouge_2["f"]/len(test_rouge_score))
        rouge_2["p"] = torch.tensor(rouge_2["p"]/len(test_rouge_score))
        rouge_2["r"] = torch.tensor(rouge_2["r"]/len(test_rouge_score))
        rouge_l["f"] = torch.tensor(rouge_l["f"]/len(test_rouge_score))
        rouge_l["p"] = torch.tensor(rouge_l["p"]/len(test_rouge_score))
        rouge_l["r"] = torch.tensor(rouge_l["r"]/len(test_rouge_score))
        result = pl.EvalResult()
        result.log_dict({"avg-rouge-1-f":rouge_1["f"], "avg-rouge-1-p":rouge_1["p"], "avg-rouge-1-r":rouge_1["r"],
                         "avg-rouge-2-f":rouge_2["f"], "avg-rouge-2-p":rouge_2["p"], "avg-rouge-2-r":rouge_2["r"],
                         "avg-rouge-L-f":rouge_l["f"], "avg-rouge-L-p":rouge_l["p"], "avg-rouge-L-r":rouge_l["r"]})
        return result



