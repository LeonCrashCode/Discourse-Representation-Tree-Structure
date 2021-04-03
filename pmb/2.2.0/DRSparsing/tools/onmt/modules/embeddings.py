""" Embeddings module """
import math
import warnings

import torch
import torch.nn as nn
from torch.nn import ParameterList, Parameter
from onmt.utils.logging import logger
from onmt.modules.util_class import Elementwise

from allennlp.modules.elmo import Elmo, batch_to_ids
from pytorch_pretrained_bert import BertTokenizer, BertModel
class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for non-recurrent neural networks.

    Implementation based on "Attention Is All You Need"
    :cite:`DBLP:journals/corr/VaswaniSPUJGKP17`

    Args:
       dropout (float): dropout parameter
       dim (int): embedding size
    """

    def __init__(self, dropout, dim, max_len=5000):
        if dim % 2 != 0:
            raise ValueError("Cannot use sin/cos positional encoding with "
                             "odd dim (got dim={:d})".format(dim))
        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp((torch.arange(0, dim, 2, dtype=torch.float) *
                             -(math.log(10000.0) / dim)))
        pe[:, 0::2] = torch.sin(position.float() * div_term)
        pe[:, 1::2] = torch.cos(position.float() * div_term)
        pe = pe.unsqueeze(1)
        super(PositionalEncoding, self).__init__()
        self.register_buffer('pe', pe)
        self.dropout = nn.Dropout(p=dropout)
        self.dim = dim

    def forward(self, emb, step=None):
        """Embed inputs.

        Args:
            emb (FloatTensor): Sequence of word vectors
                ``(seq_len, batch_size, self.dim)``
            step (int or NoneType): If stepwise (``seq_len = 1``), use
                the encoding for this position.
        """

        emb = emb * math.sqrt(self.dim)
        if step is None:
            emb = emb + self.pe[:emb.size(0)]
        else:
            emb = emb + self.pe[step]
        emb = self.dropout(emb)
        return emb


class Embeddings(nn.Module):
    """Words embeddings for encoder/decoder.

    Additionally includes ability to add sparse input features
    based on "Linguistic Input Features Improve Neural Machine Translation"
    :cite:`sennrich2016linguistic`.


    .. mermaid::

       graph LR
          A[Input]
          C[Feature 1 Lookup]
          A-->B[Word Lookup]
          A-->C
          A-->D[Feature N Lookup]
          B-->E[MLP/Concat]
          C-->E
          D-->E
          E-->F[Output]

    Args:
        word_vec_size (int): size of the dictionary of embeddings.
        word_padding_idx (int): padding index for words in the embeddings.
        feat_padding_idx (List[int]): padding index for a list of features
                                   in the embeddings.
        word_vocab_size (int): size of dictionary of embeddings for words.
        feat_vocab_sizes (List[int], optional): list of size of dictionary
            of embeddings for each feature.
        position_encoding (bool): see :class:`~onmt.modules.PositionalEncoding`
        feat_merge (string): merge action for the features embeddings:
            concat, sum or mlp.
        feat_vec_exponent (float): when using `-feat_merge concat`, feature
            embedding size is N^feat_dim_exponent, where N is the
            number of values the feature takes.
        feat_vec_size (int): embedding dimension for features when using
            `-feat_merge mlp`
        dropout (float): dropout probability.
    """

    def __init__(self, word_vec_size,
                 word_vocab_size,
                 word_padding_idx,
                 position_encoding=False,
                 feat_merge="concat",
                 feat_vec_exponent=0.7,
                 feat_vec_size=-1,
                 feat_padding_idx=[],
                 feat_vocab_sizes=[],
                 dropout=0,
                 sparse=False,
                 fix_word_vecs=False
                 ):
        self._validate_args(feat_merge, feat_vocab_sizes, feat_vec_exponent,
                            feat_vec_size, feat_padding_idx)

        if feat_padding_idx is None:
            feat_padding_idx = []
        self.word_padding_idx = word_padding_idx

        self.word_vec_size = word_vec_size

        # Dimensions and padding for constructing the word embedding matrix
        vocab_sizes = [word_vocab_size]
        emb_dims = [word_vec_size]
        pad_indices = [word_padding_idx]

        # Dimensions and padding for feature embedding matrices
        # (these have no effect if feat_vocab_sizes is empty)
        if feat_merge == 'sum':
            feat_dims = [word_vec_size] * len(feat_vocab_sizes)
        elif feat_vec_size > 0:
            feat_dims = [feat_vec_size] * len(feat_vocab_sizes)
        else:
            feat_dims = [int(vocab ** feat_vec_exponent)
                         for vocab in feat_vocab_sizes]
        vocab_sizes.extend(feat_vocab_sizes)
        emb_dims.extend(feat_dims)
        pad_indices.extend(feat_padding_idx)

        # The embedding matrix look-up tables. The first look-up table
        # is for words. Subsequent ones are for features, if any exist.
        emb_params = zip(vocab_sizes, emb_dims, pad_indices)
        embeddings = [nn.Embedding(vocab, dim, padding_idx=pad, sparse=sparse)
                      for vocab, dim, pad in emb_params]
        emb_luts = Elementwise(feat_merge, embeddings)

        # The final output size of word + feature vectors. This can vary
        # from the word vector size if and only if features are defined.
        # This is the attribute you should access if you need to know
        # how big your embeddings are going to be.
        self.embedding_size = (sum(emb_dims) if feat_merge == 'concat'
                               else word_vec_size)

        # The sequence of operations that converts the input sequence
        # into a sequence of embeddings. At minimum this consists of
        # looking up the embeddings for each word and feature in the
        # input. Model parameters may require the sequence to contain
        # additional operations as well.
        super(Embeddings, self).__init__()
        self.make_embedding = nn.Sequential()
        self.make_embedding.add_module('emb_luts', emb_luts)

        if feat_merge == 'mlp' and len(feat_vocab_sizes) > 0:
            in_dim = sum(emb_dims)
            mlp = nn.Sequential(nn.Linear(in_dim, word_vec_size), nn.ReLU())
            self.make_embedding.add_module('mlp', mlp)

        self.position_encoding = position_encoding

        if self.position_encoding:
            pe = PositionalEncoding(dropout, self.embedding_size)
            self.make_embedding.add_module('pe', pe)

        if fix_word_vecs:
            self.word_lut.weight.requires_grad = False

    def _validate_args(self, feat_merge, feat_vocab_sizes, feat_vec_exponent,
                       feat_vec_size, feat_padding_idx):
        if feat_merge == "sum":
            # features must use word_vec_size
            if feat_vec_exponent != 0.7:
                warnings.warn("Merging with sum, but got non-default "
                              "feat_vec_exponent. It will be unused.")
            if feat_vec_size != -1:
                warnings.warn("Merging with sum, but got non-default "
                              "feat_vec_size. It will be unused.")
        elif feat_vec_size > 0:
            # features will use feat_vec_size
            if feat_vec_exponent != -1:
                warnings.warn("Not merging with sum and positive "
                              "feat_vec_size, but got non-default "
                              "feat_vec_exponent. It will be unused.")
        else:
            if feat_vec_exponent <= 0:
                raise ValueError("Using feat_vec_exponent to determine "
                                 "feature vec size, but got feat_vec_exponent "
                                 "less than or equal to 0.")
        n_feats = len(feat_vocab_sizes)
        if n_feats != len(feat_padding_idx):
            raise ValueError("Got unequal number of feat_vocab_sizes and "
                             "feat_padding_idx ({:d} != {:d})".format(
                                n_feats, len(feat_padding_idx)))

    @property
    def word_lut(self):
        """Word look-up table."""
        return self.make_embedding[0][0]

    @property
    def emb_luts(self):
        """Embedding look-up table."""
        return self.make_embedding[0]

    def load_pretrained_vectors(self, emb_file):
        """Load in pretrained embeddings.

        Args:
          emb_file (str) : path to torch serialized embeddings
        """

        logger.info("start loading pretrained vector")
        if emb_file:
            logger.info("pretrain file found "+ emb_file)
            pretrained = torch.load(emb_file)
            pretrained_vec_size = pretrained.size(1)
            if self.word_vec_size > pretrained_vec_size:
                self.word_lut.weight.data[:, :pretrained_vec_size] = pretrained
            elif self.word_vec_size < pretrained_vec_size:
                self.word_lut.weight.data \
                    .copy_(pretrained[:, :self.word_vec_size])
            else:
                self.word_lut.weight.data.copy_(pretrained)
        else:
            logger.info("pretrain file not found")
    def forward(self, source, step=None):
        """Computes the embeddings for words and features.

        Args:
            source (LongTensor): index tensor ``(len, batch, nfeat)``

        Returns:
            FloatTensor: Word embeddings ``(len, batch, embedding_size)``
        """

        if self.position_encoding:
            for i, module in enumerate(self.make_embedding._modules.values()):
                if i == len(self.make_embedding._modules.values()) - 1:
                    source = module(source, step=step)
                else:
                    source = module(source)
        else:
            source = self.make_embedding(source)

        return source

class ElmoEmbeddings(nn.Module):
    

    def __init__(self,
                itos,
                word_vec_size,
                word_padding_idx,
                position_encoding=False,
                dropout=0,
                sparse=False,
                fix_word_vecs=True,
                elmo_path=""):

        super(ElmoEmbeddings, self).__init__()
        
        self.itos = itos
        self.word_padding_idx = word_padding_idx
        self.word_vec_size = word_vec_size
        # The sequence of operations that converts the input sequence
        # into a sequence of embeddings. At minimum this consists of
        # looking up the embeddings for each word and feature in the
        # input. Model parameters may require the sequence to contain
        # additional operations as well.

        self.elmo_path = elmo_path
        self.emb_luts = None

        self.position_encoding = position_encoding
        if self.position_encoding:
            self.pe = PositionalEncoding(dropout, word_vec_size)

        #if fix_word_vecs:
        #    self.word_lut.weight.requires_grad = False



    def load_pretrained_vectors(self):
        """Load in pretrained embeddings.

        Args:
          emb_file (str) : path to torch serialized embeddings
        """
        self.emb_luts = Elmo(
                self.elmo_path+"_options.json",
                self.elmo_path+"_weights.hdf5",
                1, requires_grad=False, dropout=0)

    def forward(self, source, step=None):
        """Computes the embeddings for words and features.

        Args:
            source (LongTensor): index tensor ``(len, batch, nfeat)``

        Returns:
            FloatTensor: Word embeddings ``(len, batch, embedding_size)``
        """
        sentences = []
        if torch.cuda.is_available():
            device = source.get_device()
        source = source.squeeze(2).transpose(0,1).data.tolist()

        for i in range(len(source)):
            sentences.append([])
            for j in range(len(source[0])):
                if source[i][j] == self.word_padding_idx:
                    break
                sentences[-1].append(self.itos[source[i][j]])
        #print(sentences)

        character_ids = batch_to_ids(sentences)

        if torch.cuda.is_available():
            character_ids = character_ids.to(device)
        #print(character_ids)
        embeddings = self.emb_luts(character_ids)

        source = embeddings["elmo_representations"][0]
        #print(source[0])
        source = source.transpose(0,1)

        #exit()
        source = self.pe(source)

        return source


class BertEmbeddings(nn.Module):
    

    def __init__(self,
                itos,
                word_vec_size,
                word_padding_idx,
                position_encoding=False,
                dropout=0,
                sparse=False,
                fix_word_vecs=True,
                bert_type="",
                bert_cache_path=""):

        super(BertEmbeddings, self).__init__()
        
        self.itos = itos
        self.word_padding_idx = word_padding_idx
        self.word_vec_size = word_vec_size
        # The sequence of operations that converts the input sequence
        # into a sequence of embeddings. At minimum this consists of
        # looking up the embeddings for each word and feature in the
        # input. Model parameters may require the sequence to contain
        # additional operations as well.

        self.bert_type = bert_type
        self.bert_cache_path = bert_cache_path

        self.tokenizer = None
        self.emb_luts = None

        self.position_encoding = position_encoding
        if self.position_encoding:
            self.pe = PositionalEncoding(dropout, word_vec_size)

        self.scalar_parameters = ParameterList(
                [Parameter(torch.FloatTensor([0.0]),
                           requires_grad=True) for i in range(12)])
        self.gamma = Parameter(torch.FloatTensor([1.0]), requires_grad=True)
        #if fix_word_vecs:
        #    self.word_lut.weight.requires_grad = False



    def load_pretrained_vectors(self):
        """Load in pretrained embeddings.

        Args:
          emb_file (str) : path to torch serialized embeddings
        """
        self.tokenizer = BertTokenizer.from_pretrained(self.bert_type, cache_dir=self.bert_cache_path)
        self.emb_luts = BertModel.from_pretrained(self.bert_type, cache_dir=self.bert_cache_path)
        self.emb_luts.eval()

    def forward(self, source, step=None):
        """Computes the embeddings for words and features.

        Args:
            source (LongTensor): index tensor ``(len, batch, nfeat)``

        Returns:
            FloatTensor: Word embeddings ``(len, batch, embedding_size)``
        """
        sentences = []
        masks = []
        max_length = source.size(0)

        if torch.cuda.is_available():
            device = source.get_device()
        source = source.squeeze(2).transpose(0,1).data.tolist()

        
        for i in range(len(source)):
            sentences.append([])
            masks.append([])
            for j in range(len(source[0])):
                if source[i][j] == self.word_padding_idx:
                    break
                else:
                    masks[-1].append(1)
                    if self.itos[source[i][j]] == "<unk>":
                        sentences[-1].append("[UNK]")
                    else:
                        sentences[-1].append(self.itos[source[i][j]])
        
        for i in range(len(sentences)):
            j = len(sentences[i])
            while j < max_length:
                sentences[i].append("[PAD]")
                masks[i].append(0)
                j += 1
        #print(sentences)
        indexed_tokens = [self.tokenizer.convert_tokens_to_ids(item) for item in sentences]
        
        indexed_tokens = torch.tensor(indexed_tokens)
        masks = torch.tensor(masks)
        if torch.cuda.is_available():
            indexed_tokens = indexed_tokens.to(device)
            masks = masks.to(device)
        with torch.no_grad():
            encoded_layers, _ = self.emb_luts(indexed_tokens, attention_mask=masks)

        normed_weights = torch.nn.functional.softmax(torch.cat([parameter for parameter
                                                                in self.scalar_parameters]), dim=0)
        normed_weights = torch.split(normed_weights, split_size_or_sections=1)

        pieces = []
        for weight, encoded_layer in zip(normed_weights, encoded_layers):
            pieces.append(weight * encoded_layer)
  
        source = self.gamma * sum(pieces)

        source = source.transpose(0,1)

        source = self.pe(source)

        return source
        

