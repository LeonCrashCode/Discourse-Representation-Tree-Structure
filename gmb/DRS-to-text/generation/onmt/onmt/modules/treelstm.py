from __future__ import division
import torch
import dgl
from copy import deepcopy


class StandardTree:
    def __init__(self, h_size, List, inputs, n_nodes):
        self.dgl_graph = dgl.DGLGraph()
        self.h_size = h_size
        self.build_tree(List, inputs, n_nodes)

    @classmethod
    def from_opt(cls, h_size, List, inputs, n_nodes):
        """Alternate constructor."""
        return cls(
            h_size,
            List,
            inputs,
            n_nodes)

    def build_tree(self, List, inputs, n_nodes):

        x_size = inputs[0].size(0)

        # simutaniously compute forward and backward
        self.dgl_graph.add_nodes(n_nodes, data={'x': inputs[:n_nodes],
                'h': inputs.new_zeros(size=(n_nodes, self.h_size)),
                'c': inputs.new_zeros(size=(n_nodes, self.h_size))})

        self.dgl_graph.add_edges(List[0], List[1])

class NeighborTree:
    def __init__(self, h_size, List, inputs, n_nodes):
        self.dgl_graph = dgl.DGLGraph()
        self.h_size = h_size
        self.build_tree(List, inputs, n_nodes)
        
    @classmethod
    def from_opt(cls, h_size, List, inputs, n_nodes):
        """Alternate constructor."""
        return cls(
            h_size,
            List,
            inputs,
            n_nodes)

    def build_tree(self, List, inputs, n_nodes):

        x_size = inputs[0].size(0)

        # simutaniously compute forward and backward
        self.dgl_graph.add_nodes(n_nodes, data={'x': inputs[:n_nodes],
                'h': inputs.new_zeros(size=(n_nodes, self.h_size)),
                'c': inputs.new_zeros(size=(n_nodes, self.h_size))})

        device = inputs.get_device()
        if device < 0:
            device = torch.device('cpu')

        neighbors = List[0]
        self.dgl_graph.add_edges(neighbors[0], neighbors[1], data={'t': torch.tensor([[1,0]]).expand(len(neighbors[0]), 2).to(device)})

        parents = List[1]
        self.dgl_graph.add_edges(parents[0], parents[1], data={'t': torch.tensor([[0,1]]).expand(len(parents[0]), 2).to(device)})

class MixTree:
    def __init__(self, h_size, bottom_up, List, inputs, states):
        self.dgl_graph = dgl.DGLGraph()
        self.h_size = h_size
        if bottom_up:
            self.build_bottom_up_tree(List, inputs)
        else:
            self.build_top_down_tree(List, inputs, states)
    @classmethod
    def from_opt(cls, h_size, bottom_up, List, inputs, states=None):
        """Alternate constructor."""
        return cls(
            h_size,
            bottom_up,
            List,
            inputs,
            states)

    def build_bottom_up_tree(self, List, inputs):

        x_size = inputs[0].size(0)

        def init_nodes(max_N, DRS_types):
            for idx in range(max_N):
                if idx in DRS_types:
                    self.dgl_graph.add_nodes(1, data={'x': inputs[idx].unsqueeze(0),
                            'h': inputs[idx].new_zeros(size=(1, self.h_size)),
                            'c': inputs[idx].new_zeros(size=(1, self.h_size)),
                            't': inputs[idx].new_ones(size=(1, 1))})
                else:
                    self.dgl_graph.add_nodes(1, data={'x': inputs[idx].unsqueeze(0),
                            'h': inputs[idx].new_zeros(size=(1, self.h_size)),
                            'c': inputs[idx].new_zeros(size=(1, self.h_size)),
                            't': inputs[idx].new_zeros(size=(1, 1))})
                

        init_nodes(List[-1], List[-2])
        self.dgl_graph.add_nodes(1, data={'x': inputs[0].new_zeros(size=(1, x_size)),
            'h': inputs[0].new_zeros(size=(1, self.h_size)),
            'c': inputs[0].new_zeros(size=(1, self.h_size))})
        self.dummy_node_id = self.dgl_graph.number_of_nodes() - 1
        
        for i in range(3): # childsum edge, neighbor edge and parent edge
            self.dgl_graph.add_edges(List[i][0], List[i][1])

    def build_top_down_tree(self, List, inputs, states):

        x_size = inputs[0].size(0)

        def init_nodes(max_N):
            for idx in range(max_N):
                if idx == 0:
                    self.dgl_graph.add_nodes(1, data={'x': inputs[idx].unsqueeze(0),
                            'h': states[0].unsqueeze(0),
                            'c': states[1].unsqueeze(0)})
                else:
                    self.dgl_graph.add_nodes(1, data={'x': inputs[idx].unsqueeze(0),
                            'h': inputs[idx].new_zeros(size=(1, self.h_size)),
                            'c': inputs[idx].new_zeros(size=(1, self.h_size))})
                

        init_nodes(List[-1]) # no dummy node
        for i in range(3): # childsum edge, neighbor edge and parent edge
            self.dgl_graph.add_edges(List[3+i][0], List[3+i][1])

class BatchedTree:
    def __init__(self, tree_list):
        graph_list = []
        for tree in tree_list:
            graph_list.append(tree.dgl_graph)
        self.batch_dgl_graph = dgl.batch(graph_list)
    def get_hidden_state(self, dummy_node_num=0):
        graph_list = dgl.unbatch(self.batch_dgl_graph)
        hidden_states = []
        hs = []
        cs = []
        max_nodes_num = max([len(graph.nodes) - dummy_node_num for graph in graph_list])
        for graph in graph_list:
            hiddens = graph.ndata['h']
            hs.append(graph.ndata['h'][0])
            cs.append(graph.ndata['c'][0])

            hiddens = hiddens[:hiddens.size(0)-dummy_node_num]
            node_num, hidden_num = hiddens.size()
            if len(hiddens) < max_nodes_num:
                padding = hiddens.new_zeros(size=(max_nodes_num - node_num, hidden_num))
                hiddens = torch.cat((hiddens, padding), dim=0)
            hidden_states.append(hiddens)
        return torch.stack(hidden_states), {'h': torch.stack(hs), 'c':torch.stack(cs)}


class TreeLSTM(torch.nn.Module):
    def __init__(self,
                 x_size,
                 h_size,
                 dropout,
                 cell_type='n_ary',
                 n_ary=None,
                 num_stacks=1,
                 bidirectional=True):
        super(TreeLSTM, self).__init__()
        self.x_size = x_size
        self.dropout = torch.nn.Dropout(dropout)
        self.bidirectional = bidirectional
        self.cell_type = cell_type
        if cell_type == 'n_ary':
            cell_list = [NaryTreeLSTMCell(n_ary, x_size, h_size), NaryTreeLSTMCell(n_ary, x_size, h_size)]
            for i in range(num_stacks-1):
                cell_list += [NaryTreeLSTMCell(n_ary, 2*h_size, h_size), NaryTreeLSTMCell(n_ary, 2*h_size, h_size)]
            self.cell = torch.nn.ModuleList(cell_list)
        elif cell_type == 'childsum':
            cell_list = [ChildSumTreeLSTMCell(x_size, h_size), ChildSumTreeLSTMCell(x_size, h_size)]
            for i in range(num_stacks-1):
                cell_list += [ChildSumTreeLSTMCell(2*h_size, h_size), ChildSumTreeLSTMCell(2*h_size, h_size)]
            self.cell = torch.nn.ModuleList(cell_list)
        elif cell_type == 'neighbor':
            cell_list = [NeighborTreeLSTMCell(x_size, h_size), NeighborTreeLSTMCell(x_size, h_size)]
            for i in range(num_stacks-1):
                cell_list += [NeighborTreeLSTMCell(2*h_size, h_size), NeighborTreeLSTMCell(2*h_size, h_size)]
            self.cell = torch.nn.ModuleList(cell_list)
        elif cell_type == 'mix':
            self.cell = MixTreeLSTMCell(x_size, h_size)
        else:
            assert False, "cell type is not defined"
        self.num_stacks = num_stacks

    def forward(self, batch):
        # if self.num_stacks == 1:
        #     batch.batch_dgl_graph.register_message_func(self.cell.message_func)
        #     batch.batch_dgl_graph.register_reduce_func(self.cell.reduce_func)
        #     batch.batch_dgl_graph.register_apply_node_func(self.cell.apply_node_func)
        #     # batch.batch_dgl_graph.ndata['iou'] = self.cell.W_iou(self.dropout(batch.batch_dgl_graph.ndata['x']))
        #     batch.batch_dgl_graph.ndata['iou'] = self.cell.W_iou(batch.batch_dgl_graph.ndata['x'])
        #     # if self.cell_type == "mix":
        #     #     batch.batch_dgl_graph.ndata['iou_s'] = self.cell.W_iou_s(self.dropout(batch.batch_dgl_graph.ndata['x']))
        #     dgl.prop_nodes_topo(batch.batch_dgl_graph)
        #     return batch


        # batches = [deepcopy(batch) for _ in range(self.num_stacks)]
        # for stack in range(self.num_stacks):
        #     cur_batch = batches[stack]
        #     if stack > 0:
        #         prev_batch = batches[stack - 1]
        #         cur_batch.batch_dgl_graph.ndata['x'] = prev_batch.batch_dgl_graph.ndata['h']
        #     cur_batch.batch_dgl_graph.register_message_func(self.cell.message_func)
        #     cur_batch.batch_dgl_graph.register_reduce_func(self.cell.reduce_func)
        #     cur_batch.batch_dgl_graph.register_apply_node_func(self.cell.apply_node_func)
        #     if stack == self.num_stacks - 1:
        #         cur_batch.batch_dgl_graph.ndata['iou'] = self.cell.W_iou(batch.batch_dgl_graph.ndata['x'])
        #     else:
        #         cur_batch.batch_dgl_graph.ndata['iou'] = self.cell.W_iou(self.dropout(batch.batch_dgl_graph.ndata['x']))
        #     # if self.cell_type == "mix":
        #     #     cur_batch.batch_dgl_graph.ndata['iou_s'] = self.cell.W_iou_s(self.dropout(batch.batch_dgl_graph.ndata['x']))
        #     dgl.prop_nodes_topo(cur_batch.batch_dgl_graph)
        # return batches

        if self.bidirectional:
            batch, back_batch = batch

        layer_caches = []
        layer_batches = []
        layer_batches_back = []
        for stack in range(self.num_stacks):
            # print(stack)
            if stack == 0:
                pass
            else:
                if self.bidirectional:
                    batch.batch_dgl_graph.ndata['x'] = torch.cat((layer_caches[-2]['h'], layer_caches[-1]['h']), dim=-1)
                else:
                    batch.batch_dgl_graph.ndata['x'] = layer_caches[-1]['h']
            batch.batch_dgl_graph.register_message_func(self.cell[stack*2].message_func)
            batch.batch_dgl_graph.register_reduce_func(self.cell[stack*2].reduce_func)
            batch.batch_dgl_graph.register_apply_node_func(self.cell[stack*2].apply_node_func)
            if stack == self.num_stacks - 1:
                batch.batch_dgl_graph.ndata['iou'] = self.cell[stack*2].W_iou(batch.batch_dgl_graph.ndata['x'])
            else:
                batch.batch_dgl_graph.ndata['iou'] = self.cell[stack*2].W_iou(self.dropout(batch.batch_dgl_graph.ndata['x']))
            dgl.prop_nodes_topo(batch.batch_dgl_graph)

            layer_caches.append({'h': batch.batch_dgl_graph.ndata['h'], 'c': batch.batch_dgl_graph.ndata['c']})

            layer_batches.append(dgl.unbatch(batch.batch_dgl_graph))

            if self.bidirectional:
                if stack == 0:
                    pass
                else:
                    if self.bidirectional:
                        back_batch.batch_dgl_graph.ndata['x'] = torch.cat((layer_caches[-3]['h'], layer_caches[-2]['h']), dim=-1)
                    else:
                        back_batch.batch_dgl_graph.ndata['x'] = layer_caches[-1]['h']
                back_batch.batch_dgl_graph.register_message_func(self.cell[stack*2+1].message_func)
                back_batch.batch_dgl_graph.register_reduce_func(self.cell[stack*2+1].reduce_func)
                back_batch.batch_dgl_graph.register_apply_node_func(self.cell[stack*2+1].apply_node_func)
                if stack == self.num_stacks - 1:
                    back_batch.batch_dgl_graph.ndata['iou'] = self.cell[stack*2+1].W_iou(back_batch.batch_dgl_graph.ndata['x'])
                else:
                    back_batch.batch_dgl_graph.ndata['iou'] = self.cell[stack*2+1].W_iou(self.dropout(back_batch.batch_dgl_graph.ndata['x']))
                dgl.prop_nodes_topo(back_batch.batch_dgl_graph)

                layer_caches.append({'h': back_batch.batch_dgl_graph.ndata['h'], 'c': back_batch.batch_dgl_graph.ndata['c']})

                layer_batches_back.append(dgl.unbatch(back_batch.batch_dgl_graph))

        if self.bidirectional:
            layer_hidden_states = []
            for stack in range(self.num_stacks):

                graph_list = layer_batches[stack]
                graph_list_back = layer_batches_back[stack]

                hidden_states = []
                # hs = []
                # cs = []
                max_nodes_num = max([len(graph.nodes) for graph in graph_list])
                for graph, graph_back in zip(graph_list, graph_list_back):
                    hiddens = torch.cat((graph.ndata['h'],graph_back.ndata['h']), dim=-1)
                    # hs.append(graph.ndata['h'][0])
                    # cs.append(graph.ndata['c'][0])
                    node_num, hidden_num = hiddens.size()
                    if len(hiddens) < max_nodes_num:
                        padding = hiddens.new_zeros(size=(max_nodes_num - node_num, hidden_num))
                        hiddens = torch.cat((hiddens, padding), dim=0)
                    hidden_states.append(hiddens)

                layer_hidden_states.append(torch.stack(hidden_states))
            return layer_hidden_states

        else:
            layer_hidden_states = []
            for stack in range(self.num_stacks):

                graph_list = layer_batches[stack]

                hidden_states = []
                # hs = []
                # cs = []
                max_nodes_num = max([len(graph.nodes) for graph in graph_list])
                for graph in graph_list:
                    hiddens = graph.ndata['h']
                    # hs.append(graph.ndata['h'][0])
                    # cs.append(graph.ndata['c'][0])
                    node_num, hidden_num = hiddens.size()
                    if len(hiddens) < max_nodes_num:
                        padding = hiddens.new_zeros(size=(max_nodes_num - node_num, hidden_num))
                        hiddens = torch.cat((hiddens, padding), dim=0)
                    hidden_states.append(hiddens)
                layer_hidden_states.append(torch.stack(hidden_states))
            return layer_hidden_states


class NaryTreeLSTMCell(torch.nn.Module):
    def __init__(self, n_ary, x_size, h_size):
        super(NaryTreeLSTMCell, self).__init__()
        self.n_ary = n_ary
        self.h_size = h_size
        self.W_iou = torch.nn.Linear(x_size, 3 * h_size, bias=False)
        self.U_iou = torch.nn.Linear(n_ary * h_size, 3 * h_size, bias=False)
        self.b_iou = torch.nn.Parameter(torch.zeros(1, 3 * h_size), requires_grad=True)
        self.U_f = torch.nn.Linear(n_ary * h_size, n_ary * h_size)

    def message_func(self, edges):
        return {'h': edges.src['h'], 'c': edges.src['c']}

    def reduce_func(self, nodes):
        h_cat = nodes.mailbox['h'].view(nodes.mailbox['h'].size(0), -1)
        padding_hs = self.n_ary - nodes.mailbox['h'].size(1)
        assert self.n_ary - nodes.mailbox['h'].size(1) >= 0
        padding = h_cat.new_zeros(size=(nodes.mailbox['h'].size(0), padding_hs * self.h_size))
        h_cat = torch.cat((h_cat, padding), dim=1)
        f = torch.sigmoid(self.U_f(h_cat)).view(nodes.mailbox['h'].size(0), self.n_ary, self.h_size)
        padding_cs = self.n_ary - nodes.mailbox['c'].size(1)
        padding = h_cat.new_zeros(size=(nodes.mailbox['c'].size(0), padding_cs, self.h_size))
        c = torch.cat((nodes.mailbox['c'], padding), dim=1)
        c = torch.sum(f * c, 1)
        return {'iou': nodes.data['iou'] + self.U_iou(h_cat), 'c': c}

    def apply_node_func(self, nodes):
        iou = nodes.data['iou'] + self.b_iou
        i, o, u = torch.chunk(iou, 3, 1)
        i, o, u = torch.sigmoid(i), torch.sigmoid(o), torch.tanh(u)
        c = i * u + nodes.data['c']
        h = o * torch.tanh(c)
        return {'h': h, 'c': c}


class ChildSumTreeLSTMCell(torch.nn.Module):
    def __init__(self, x_size, h_size):
        super(ChildSumTreeLSTMCell, self).__init__()
        self.W_iou = torch.nn.Linear(x_size, 3 * h_size, bias=False)
        self.U_iou = torch.nn.Linear(h_size, 3 * h_size, bias=False)
        self.b_iou = torch.nn.Parameter(torch.zeros(1, 3 * h_size), requires_grad=True)
        self.U_f = torch.nn.Linear(h_size, h_size)

    def message_func(self, edges):
        return {'h': edges.src['h'], 'c': edges.src['c']}

    def reduce_func(self, nodes):
        h_tild = torch.sum(nodes.mailbox['h'], 1)
        f = torch.sigmoid(self.U_f(nodes.mailbox['h']))
        c = torch.sum(f * nodes.mailbox['c'], 1)
        return {'iou': nodes.data['iou'] + self.U_iou(h_tild), 'c': c}

    def apply_node_func(self, nodes):
        iou = nodes.data['iou'] + self.b_iou
        i, o, u = torch.chunk(iou, 3, 1)
        i, o, u = torch.sigmoid(i), torch.sigmoid(o), torch.tanh(u)
        c = i * u + nodes.data['c']
        h = o * torch.tanh(c)
        return {'h': h, 'c': c}

class NeighborTreeLSTMCell(torch.nn.Module):
    def __init__(self, x_size, h_size):
        super(NeighborTreeLSTMCell, self).__init__()
        self.h_size = h_size
        self.W_iou = torch.nn.Linear(x_size, 3 * h_size, bias=False)
        self.U_iou = torch.nn.Linear(2 * h_size, 3 * h_size, bias=False)
        self.b_iou = torch.nn.Parameter(torch.zeros(1, 3 * h_size), requires_grad=True)
        self.U_f = torch.nn.Linear(2 * h_size, 2 * h_size)

    def message_func(self, edges):
        return {'h': edges.src['h'], 'c': edges.src['c'], 't': edges.data['t']}

    def reduce_func(self, nodes):
        h_cat = nodes.mailbox['h'].view(nodes.mailbox['h'].size(0), -1)
        if 2 - nodes.mailbox['h'].size(1) == 0:
            f = torch.sigmoid(self.U_f(h_cat)).view(nodes.mailbox['h'].size(0), 2, self.h_size)
            c = nodes.mailbox['c']
            c = torch.sum(f*c, 1)
        else:
            mask = nodes.mailbox['t'].view(nodes.mailbox['t'].size(0), 2)
            mask = mask.unsqueeze(-1).expand(mask.size(0), mask.size(1), self.h_size)
            h_cat = torch.cat((h_cat, h_cat), dim=1)
            f = torch.sigmoid(self.U_f(h_cat)).view(nodes.mailbox['h'].size(0), 2, self.h_size)
            c_cat = torch.cat((nodes.mailbox['c'], nodes.mailbox['c']), dim=1)
            c = torch.sum(f*c_cat*mask, 1)

        # padding_hs = 2 - nodes.mailbox['h'].size(1)
        # padding = h_cat.new_zeros(size=(nodes.mailbox['h'].size(0), padding_hs * self.h_size))
        # h_cat = torch.cat((h_cat, padding), dim=1)
        # f = torch.sigmoid(self.U_f(h_cat)).view(nodes.mailbox['h'].size(0), 2, self.h_size)
        # padding_cs = 2 - nodes.mailbox['c'].size(1)
        # padding = h_cat.new_zeros(size=(nodes.mailbox['c'].size(0), padding_cs, self.h_size))
        # c = torch.cat((nodes.mailbox['c'], padding), dim=1)
        # c = torch.sum(f * c, 1)
        return {'iou': nodes.data['iou'] + self.U_iou(h_cat), 'c': c}

    def apply_node_func(self, nodes):
        iou = nodes.data['iou'] + self.b_iou
        i, o, u = torch.chunk(iou, 3, 1)
        i, o, u = torch.sigmoid(i), torch.sigmoid(o), torch.tanh(u)
        c = i * u + nodes.data['c']
        h = o * torch.tanh(c)
        return {'h': h, 'c': c}

class MixTreeLSTMCell(torch.nn.Module):
    def __init__(self, x_size, h_size):
        super(MixTreeLSTMCell, self).__init__()
        self.h_size = h_size

        self.W_iou = torch.nn.Linear(x_size, 3 * h_size, bias=False)
        self.U_iou = torch.nn.Linear(2 * h_size, 3 * h_size, bias=False)
        self.b_iou = torch.nn.Parameter(torch.zeros(1, 3 * h_size), requires_grad=True)
        self.U_f = torch.nn.Linear(2 * h_size, 2 * h_size)

        self.W_iou_s = torch.nn.Linear(x_size, 3 * h_size, bias=False)
        self.U_iou_s = torch.nn.Linear(h_size, 3 * h_size, bias=False)
        self.b_iou_s = torch.nn.Parameter(torch.zeros(1, 3 * h_size), requires_grad=True)
        self.U_f_s = torch.nn.Linear(h_size, h_size)

    def message_func(self, edges):
        return {'h': edges.src['h'], 'c': edges.src['c']}

    def reduce_func(self, nodes):
        types = nodes.data['t'].view(-1).tolist()

        if all([t == 1 for t in types]):
            iou, c = self.sum_reduce_func(nodes.data['iou_s'], nodes.mailbox['h'], nodes.mailbox['c'])
        elif all([t == 0 for t in types]):
            iou, c = self.nary_reduce_func(nodes.data['iou'], nodes.mailbox['h'], nodes.mailbox['c'])
        else:
            data_iou_1 = []
            data_h_1 = []
            data_c_1 = []

            data_iou_0 = []
            data_h_0 = []
            data_c_0 = []
            
            for i, t in enumerate(types):
                if t > 0:
                    data_iou_1.append(nodes.data['iou_s'][i])
                    data_h_1.append(nodes.mailbox['h'][i])
                    data_c_1.append(nodes.mailbox['c'][i])
                else:
                    data_iou_0.append(nodes.data['iou'][i])
                    data_h_0.append(nodes.mailbox['h'][i])
                    data_c_0.append(nodes.mailbox['c'][i])

            data_iou_1 = torch.stack(data_iou_1)
            data_h_1 = torch.stack(data_h_1)
            data_c_1 = torch.stack(data_c_1)

            data_iou_0 = torch.stack(data_iou_0)
            data_h_0 = torch.stack(data_h_0)
            data_c_0 = torch.stack(data_c_0)

            iou_1, c_1 = self.sum_reduce_func(data_iou_1, data_h_1, data_c_1)
            iou_0, c_0 = self.nary_reduce_func(data_iou_0, data_h_0, data_c_0)


            iou = []
            c = []
            idx_1 = 0
            idx_0 = 0
            for i, t in enumerate(types):
                if t > 0:
                    iou.append(iou_1[idx_1])
                    c.append(c_1[idx_1])
                    idx_1 += 1
                else:
                    iou.append(iou_0[idx_0])
                    c.append(c_0[idx_0])
                    idx_0 += 1
            iou = torch.stack(iou)
            c = torch.stack(c)

        return {'iou': iou, 'c': c}

    def nary_reduce_func(self, iou, h, c):
        # print("nary_reduce_func")
        assert h.size(1) <= 2

        h_cat = h.view(h.size(0), -1)
        padding_hs = 2 - h.size(1)
        padding = h_cat.new_zeros(size=(h.size(0), padding_hs * self.h_size))
        h_cat = torch.cat((h_cat, padding), dim=1)
        f = torch.sigmoid(self.U_f(h_cat)).view(h.size(0), 2, self.h_size)

        padding_cs = 2 - c.size(1)
        padding = h_cat.new_zeros(size=(c.size(0), padding_cs, self.h_size))
        c = torch.cat((c, padding), dim=1)
        c = torch.sum(f*c, 1)

        return iou + self.U_iou(h_cat), c

    def sum_reduce_func(self, iou, h, c):
        # print("sum_reduce_func")
        h_tild = torch.sum(h, 1)
        f = torch.sigmoid(self.U_f_s(h))
        c = torch.sum(f * c, 1)

        return iou + self.U_iou_s(h_tild), c

    def apply_node_func(self, nodes):
        types = nodes.data['t'].view(-1).tolist()

        if all([t == 1 for t in types]):
            h, c = self.apply_sum_node_func(nodes.data['iou_s'], nodes.data['c'])
        elif all([t == 0 for t in types]):
            h, c = self.apply_nary_node_func(nodes.data['iou'], nodes.data['c'])
        else:
            data_iou_1 = []
            data_c_1 = []

            data_iou_0 = []
            data_c_0 = []

            for i, t in enumerate(types):
                if t > 0:
                    data_iou_1.append(nodes.data['iou_s'][i])
                    data_c_1.append(nodes.data['c'][i])
                else:
                    data_iou_0.append(nodes.data['iou'][i])
                    data_c_0.append(nodes.data['c'][i])

            data_iou_1 = torch.stack(data_iou_1)
            data_c_1 = torch.stack(data_c_1)

            data_iou_0 = torch.stack(data_iou_0)
            data_c_0 = torch.stack(data_c_0)        

            h_1, c_1 = self.apply_sum_node_func(data_iou_1, data_c_1)
            h_0, c_0 = self.apply_nary_node_func(data_iou_0, data_c_0)

            h = []
            c = []
            idx_1 = 0
            idx_0 = 0
            for i, t in enumerate(types):
                if t > 0:
                    h.append(h_1[idx_1])
                    c.append(c_1[idx_1])
                    idx_1 += 1
                else:
                    h.append(h_0[idx_0])
                    c.append(c_0[idx_0])
                    idx_0 += 1
            h = torch.stack(h)
            c = torch.stack(c)
        return {'h': h, 'c': c}

    def apply_sum_node_func(self, iou, c):
        # print("apply_sum_node_func")
        iou = iou + self.b_iou_s
        i, o, u = torch.chunk(iou, 3, 1)
        i, o, u = torch.sigmoid(i), torch.sigmoid(o), torch.tanh(u)
        c = i * u + c
        h = o * torch.tanh(c)

        return h, c

    def apply_nary_node_func(self, iou, c):
        # print("apply_nary_node_func")
        iou = iou + self.b_iou
        i, o, u = torch.chunk(iou, 3, 1)
        i, o, u = torch.sigmoid(i), torch.sigmoid(o), torch.tanh(u)
        c = i * u + c
        h = o * torch.tanh(c)

        return h, c

