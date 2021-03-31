"""Define RNN-based encoders."""
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.utils.rnn import pack_padded_sequence as pack
from torch.nn.utils.rnn import pad_packed_sequence as unpack

from onmt.encoders.encoder import EncoderBase
from onmt.utils.rnn_factory import rnn_factory

import torch
from onmt.modules.treelstm import StandardTree, BatchedTree, TreeLSTM, NeighborTree, MixTree
# import networkx as nx
# import matplotlib.pyplot as plt
class TreeEncoder(EncoderBase):
    """ A generic recurrent neural network encoder.

    Args:
       rnn_type (str):
          style of recurrent unit to use, one of [RNN, LSTM, GRU, SRU]
       bidirectional (bool) : use a bidirectional RNN
       num_layers (int) : number of stacked layers
       hidden_size (int) : hidden size of each layer
       dropout (float) : dropout value for :class:`torch.nn.Dropout`
       embeddings (onmt.modules.Embeddings): embedding module to use
    """

    def __init__(self, rnn_type, tree_type, nary, bidirectional, num_layers,
                 hidden_size, dropout=0.0, embeddings=None,
                 use_bridge=False):

        super(TreeEncoder, self).__init__()
        assert embeddings is not None

        num_directions = 2 if bidirectional else 1
        assert hidden_size % num_directions == 0
        hidden_size = hidden_size // num_directions
        self.embeddings = embeddings
        self.hidden_size = hidden_size

        # self.rnn, self.no_pack_padded_seq = \
        #     rnn_factory(rnn_type,
        #                 input_size=embeddings.embedding_size,
        #                 hidden_size=hidden_size,
        #                 num_layers=num_layers,
        #                 dropout=dropout,
        #                 bidirectional=bidirectional)
        self.layers = num_layers
        self.bidir_treelstm = bidirectional

        print("nary",nary)
        self.treelstm = TreeLSTM(
            x_size=embeddings.embedding_size,
            h_size=hidden_size,
            dropout=dropout,
            cell_type=tree_type,
            n_ary=nary,
            num_stacks=num_layers,
            bidirectional=bidirectional
            )
        self.tree_type = tree_type
        self.tree_builder = None
        if tree_type in ["childsum", "n_ary"]:
            self.tree_builder = StandardTree
        elif tree_type == "neighbor":
            self.tree_builder = NeighborTree
        elif tree_type == "mix":
            self.tree_builder = MixTree
        # if bidirectional:
        #     self.childsumtreelstm = [ChildSumTreeLSTM(
        #         embeddings.embedding_size,
        #         hidden_size, dropout)]
        #         # embeddings.embedding_size//2)
        #     for layer in range(self.layers-1):
        #         self.childsumtreelstm.append(ChildSumTreeLSTM(
        #         hidden_size * 2,
        #         hidden_size, dropout))

        #     self.childsumtreelstm = nn.ModuleList(self.childsumtreelstm)

        #     self.topdown = nn.ModuleList([TopDownTreeLSTM(
        #         hidden_size,
        #         hidden_size, dropout) for layer in range(self.layers)])
        #         # embeddings.embedding_size//2,
        #         # embeddings.embedding_size//2)

        # else:
        #     self.childsumtreelstm = ChildSumTreeLSTM(
        #         embeddings.embedding_size,
        #         hidden_size, dropout) 
        #     for layer in range(self.layers-1):
        #         self.childsumtreelstm.append(ChildSumTreeLSTM(
        #         hidden_size,
        #         hidden_size, dropout)) 
        #     self.childsumtreelstm = nn.ModuleList(self.childsumtreelstm)

        # Initialize the bridge layer
        # self.use_bridge = use_bridge
        # if self.use_bridge:
        #     self._initialize_bridge(rnn_type,
        #                             hidden_size,
        #                             num_layers)

    @classmethod
    def from_opt(cls, opt, embeddings):
        """Alternate constructor."""
        return cls(
            opt.rnn_type,
            opt.tree_type,
            opt.nary,
            opt.bidirectional or opt.brnn,
            opt.enc_layers,
            opt.enc_rnn_size,
            opt.dropout[0] if type(opt.dropout) is list else opt.dropout,
            embeddings,
            opt.bridge)

    def bid(self, tree):
        if self.tree_type == "neighbor":
            return (tree[0], tree[1]), (tree[2], tree[3])
        elif self.tree_type in ["n_ary", "childsum"]:
            return tree[0], tree[1]
        else:
            assert False, "unrecognized tree type"
    def forward(self, src, lengths=None, tree=None):
        """See :func:`EncoderBase.forward()`"""
        self._check_args(src, lengths)

        emb = self.embeddings(src)
        s_len, batch, emb_dim = emb.size()

        # packed_emb = emb
        # if lengths is not None and not self.no_pack_padded_seq:
        #     # Lengths data is wrapped inside a Tensor.
        #     lengths_list = lengths.view(-1).tolist()
        #     packed_emb = pack(emb, lengths_list)

        # memory_bank, encoder_final = self.rnn(packed_emb)

        memory_bank = emb
        print(emb.size())

        tree_list = []
        back_tree_list = []
        for i in range(batch):
            # trees = self.List2Tree(tree[i])
            inputs = memory_bank[:, i, :]

            l2r, r2l = self.bid(tree[i])
            # tree_list.append(self.build_bottom_up_tree(tree[i], inputs))
            t = self.tree_builder.from_opt(self.hidden_size, l2r, inputs, lengths[i])
            tree_list.append(t)
            # nx_G = t.dgl_graph.to_networkx().to_directed()
            # pos = nx.kamada_kawai_layout(nx_G)
            # nx.draw(nx_G, pos, with_labels=True, node_color=[[.7, .7, .7]])
            # plt.savefig("path.png")
            # exit(-1)
            

            t = self.tree_builder.from_opt(self.hidden_size, r2l, inputs, lengths[i])
            back_tree_list.append(t)

        tree_batch = BatchedTree(tree_list)
        back_tree_batch = BatchedTree(back_tree_list)
        layer_hidden_states = self.treelstm([tree_batch,back_tree_batch])

        # print(layer_hidden_states[-1].size())
        # print(layer_hidden_states[-1])
        # exit(-1)

        return (layer_hidden_states[-1].size(0), None), layer_hidden_states[-1].transpose(0,1), lengths
        # if self.tree_type in ["neighbor", "mix"]:
        #     bu_hiddens, bu_states = tree_batch.get_hidden_state(dummy_node_num=1)
        # else:
        #     bu_hiddens, bu_states = tree_batch.get_hidden_state()
        # print(bu_hiddens.size())
        # tree_list = []
        # for i in range(batch):
        #     inputs = bu_hiddens[i]
        #     # tree_list.append(self.build_top_down_tree(tree[i], inputs, (bu_states['h'][i], bu_states['c'][i])))
        #     t = self.tree_builder.from_opt(self.hidden_size, False, tree[i], inputs, (bu_states['h'][i], bu_states['c'][i]))
        #     # nx_G = t.dgl_graph.to_networkx().to_directed()
        #     # pos = nx.kamada_kawai_layout(nx_G)
        #     # nx.draw(nx_G, pos, with_labels=True, node_color=[[.7, .7, .7]])
        #     # plt.savefig("path.png")
        #     # exit(-1)
        #     tree_list.append(t)

        # tree_batch = BatchedTree(tree_list)
        # tree_batch = self.td(tree_batch)
        # td_hiddens, td_states= tree_batch.get_hidden_state()
        # print(td_hiddens.size())
        # print(td_hiddens)
        # print(td_hiddens.size())


        # print(hiddens.size())

        # return (batch, None), hiddens.transpose(0,1), lengths
        # n = contexts[0][0].size(2) // 2
        # for layer in range(self.layers):
        #     state_one = torch.cat(states[layer], 0).unsqueeze(0)
        #     if self.bidir_treelstm:
        #         state_one = torch.cat([state_one[:,:,0:n], state_one[:,:,n:n*2]], 0)
        #     state_batch.append(state_one)

        #     hidden_one = torch.cat(hiddens[layer], 0).unsqueeze(0)
        #     if self.bidir_treelstm:
        #         hidden_one = torch.cat([hidden_one[:,:,0:n], hidden_one[:,:,n:n*2]], 0)
        #     hidden_batch.append(state_one)
        #     context_batch.append(torch.cat(contexts[layer], 1))

        # encoder_final = (torch.cat(hidden_batch,0), torch.cat(state_batch,0))
        # memory_bank = context_batch[-1]
        # if self.rnn is not None:
        #     n = context_batch.size(2) // 2
        #     hidden_batch = hidden_batch.unsqueeze(0)
        #     h = Variable(hidden_batch.data.new(2, batch, n).fill_(0.))
        #     h[0] = hidden_batch[:,:,0:n]
        #     h[1] = hidden_batch[:,:,n:n * 2]      

        #     state_batch = state_batch.unsqueeze(0)
        #     s = Variable(hidden_batch.data.new(2, batch, n).fill_(0.))
        #     s[0] = state_batch[:,:,0:n]
        #     s[1] = state_batch[:,:,n:n * 2]
        #     encoder_final = (h, s)
        #     memory_bank = context_batch

        # else:
        #     encoder_final = (hidden_batch.unsqueeze(0), state_batch.unsqueeze(0)) 
        #     memory_bank = context_batch            

        # if self.use_bridge:
        #     encoder_final = self._bridge(encoder_final)
        # print(encoder_final[0].size(), encoder_final[1].size())
        # print(memory_bank.size())
        # print(lengths.size())
        # exit(-1)
        # return encoder_final, memory_bank, lengths


    def _initialize_bridge(self, rnn_type,
                           hidden_size,
                           num_layers):

        # LSTM has hidden and cell state, other only one
        number_of_states = 2 if rnn_type == "LSTM" else 1
        # Total number of states
        self.total_hidden_dim = hidden_size * num_layers

        # Build a linear layer for each
        self.bridge = nn.ModuleList([nn.Linear(self.total_hidden_dim,
                                               self.total_hidden_dim,
                                               bias=True)
                                     for _ in range(number_of_states)])

    # def _bridge(self, hidden):
    #     """Forward hidden state through bridge."""
    #     def bottle_hidden(linear, states):
    #         """
    #         Transform from 3D to 2D, apply linear and return initial size
    #         """
    #         size = states.size()
    #         result = linear(states.view(-1, self.total_hidden_dim))
    #         return F.relu(result).view(size)

    #     if isinstance(hidden, tuple):  # LSTM
    #         outs = tuple([bottle_hidden(layer, hidden[ix])
    #                       for ix, layer in enumerate(self.bridge)])
    #     else:
    #         outs = bottle_hidden(self.bridge[0], hidden)
    #     return outs

    def update_dropout(self, dropout):
        self.rnn.dropout = dropout
