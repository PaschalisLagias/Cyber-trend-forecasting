# import sys
# import time
# from util import DataLoaderS

# torch imports
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple, Union
from pathlib import Path
from matplotlib import pyplot
import seaborn as sns

# custom script imports
from layer import dilated_inception, graph_constructor, mixprop, LayerNorm

SEED = 123


class GTNet(nn.Module):
    def __init__(
        self,
        gcn_true: bool,
        build_adp: bool,
        gcn_depth: int,
        num_nodes: int,
        device: torch.device,
        predefined_adp: Optional[torch.Tensor] = None,
        static_feat: Optional[torch.Tensor] = None,
        dropout: float = 0.3,
        subgraph_size: int = 20,
        node_dim: int = 40,
        dilation_exponential: int = 1,
        conv_channels: int = 32,
        residual_channels: int = 32,
        skip_channels: int = 64,
        end_channels: int = 128,
        seq_length: int = 12,
        in_dim: int = 2,
        out_dim: int = 12,
        layers: int = 3,
        propalpha: float = 0.05,
        tanhalpha: float = 3,
        layer_norm_affline: bool = True
    ):
        """
        Init the GTNet model.

        :param gcn_true: Whether to use GCN layers.
        :param build_adp: Whether to build an adaptive adjacency matrix.
        :param gcn_depth: Depth of the GCN layers.
        :param num_nodes: Number of nodes in the graph.
        :param device: The device (CPU or GPU) to run the model on.
        :param predefined_adp: Default adjacency matrix if build_adp is False.
        :param static_feat: Static node features. Defaults to None.
        :param dropout: Dropout rate. Defaults to 0.3.
        :param subgraph_size: Size of the subgraph for graph construction.
        :param node_dim: Dimension of node embeddings.
        :param dilation_exponential: Dilation factor for dilated convolutions.
        :param conv_channels: Number of channels for convolutional layers.
        :param residual_channels: Number of residual channels.
        :param skip_channels: Number of skip connection channels.
        :param end_channels: Number of channels for the output layers.
        :param seq_length: Length of the input sequence.
        :param in_dim: Input feature dimension.
        :param out_dim: Output dimension.
        :param layers: Number of GTNet layers.
        :param propalpha: Alpha parameter for mixprop layer.
        :param tanhalpha: Alpha parameter for tanh activation in
        graph constructor.
        layer_norm_affline: Whether to use affine transformation in LayerNorm.
        """
        super(GTNet, self).__init__()

        # Init attention scores and adaptive adjacency matrix placeholders
        self.attention_scores = None
        self.adp = None

        self.gcn_true = gcn_true
        self.build_adp = build_adp
        self.num_nodes = num_nodes
        self.dropout = dropout
        self.predefined_adp = predefined_adp
        self.filter_convs = nn.ModuleList()
        self.gate_convs = nn.ModuleList()
        self.residual_convs = nn.ModuleList()
        self.skip_convs = nn.ModuleList()
        self.gconv1 = nn.ModuleList()
        self.gconv2 = nn.ModuleList()
        self.norm = nn.ModuleList()
        self.start_conv = nn.Conv2d(in_channels=in_dim,
                                    out_channels=residual_channels,
                                    kernel_size=(1, 1))
        self.gc = graph_constructor(
            num_nodes, subgraph_size, node_dim, device,
            alpha=tanhalpha, static_feat=static_feat
        )

        self.seq_length = seq_length
        kernel_size = 7

        if dilation_exponential > 1:
            self.receptive_field = int(
                1 + (kernel_size - 1) *
                (dilation_exponential**layers - 1) / (dilation_exponential - 1)
            )
        else:
            self.receptive_field = layers * (kernel_size - 1) + 1

        for i in range(1):
            if dilation_exponential > 1:
                rf_size_i = int(
                    1 + i * (kernel_size - 1) *
                    (dilation_exponential**layers - 1) /
                    (dilation_exponential - 1)
                )
            else:
                rf_size_i = i * layers * (kernel_size - 1) + 1

            new_dilation = 1
            for j in range(1, layers + 1):
                if dilation_exponential > 1:
                    rf_size_j = int(
                        rf_size_i + (kernel_size-1) *
                        (dilation_exponential**j - 1) /
                        (dilation_exponential - 1)
                    )
                else:
                    rf_size_j = rf_size_i+j * (kernel_size - 1)

                self.filter_convs.append(
                    dilated_inception(
                        residual_channels, conv_channels,
                        dilation_factor=new_dilation
                    )
                )

                self.gate_convs.append(
                    dilated_inception(
                        residual_channels, conv_channels,
                        dilation_factor=new_dilation
                    )
                )

                self.residual_convs.append(
                    nn.Conv2d(
                        in_channels=conv_channels,
                        out_channels=residual_channels, kernel_size=(1, 1)
                    )
                )

                if self.seq_length > self.receptive_field:
                    self.skip_convs.append(
                        nn.Conv2d(
                            in_channels=conv_channels,
                            out_channels=skip_channels,
                            kernel_size=(1, self.seq_length - rf_size_j + 1)
                        )
                    )

                else:
                    self.skip_convs.append(
                        nn.Conv2d(
                            in_channels=conv_channels,
                            out_channels=skip_channels,
                            kernel_size=(
                                1, self.receptive_field - rf_size_j + 1
                            )
                        )
                    )

                if self.gcn_true:
                    self.gconv1.append(
                        mixprop(
                            conv_channels, residual_channels,
                            gcn_depth, dropout, propalpha
                        )
                    )

                    self.gconv2.append(
                        mixprop(
                            conv_channels, residual_channels,
                            gcn_depth, dropout, propalpha
                        )
                    )

                if self.seq_length > self.receptive_field:
                    self.norm.append(
                        LayerNorm(
                            (residual_channels, num_nodes,
                             self.seq_length - rf_size_j + 1),
                            elementwise_affine=layer_norm_affline))
                else:
                    self.norm.append(
                        LayerNorm(
                            (residual_channels, num_nodes,
                             self.receptive_field - rf_size_j + 1),
                            elementwise_affine=layer_norm_affline))

                new_dilation *= dilation_exponential

        self.layers = layers
        self.end_conv_1 = nn.Conv2d(
            in_channels=skip_channels, out_channels=end_channels,
            kernel_size=(1, 1), bias=True
        )

        self.end_conv_2 = nn.Conv2d(
            in_channels=end_channels, out_channels=out_dim,
            kernel_size=(1, 1), bias=True
        )

        if self.seq_length > self.receptive_field:
            self.skip0 = nn.Conv2d(
                in_channels=in_dim, out_channels=skip_channels,
                kernel_size=(1, self.seq_length), bias=True
            )

            self.skipE = nn.Conv2d(
                in_channels=residual_channels, out_channels=skip_channels,
                kernel_size=(1, self.seq_length-self.receptive_field + 1),
                bias=True
            )

        else:
            self.skip0 = nn.Conv2d(
                in_channels=in_dim, out_channels=skip_channels,
                kernel_size=(1, self.receptive_field), bias=True
            )

            self.skipE = nn.Conv2d(
                in_channels=residual_channels, out_channels=skip_channels,
                kernel_size=(1, 1), bias=True
            )

        self.idx = torch.arange(self.num_nodes).to(device)

    def forward(self, input_, idx=None, return_attention=False) -> Union[
        torch.Tensor, 
        Tuple[torch.Tensor, Optional[torch.Tensor]]
    ]:
        self.attention_scores = None
        seq_len = input_.size(3)
        msg = 'input sequence length not equal to preset sequence length'
        assert seq_len == self.seq_length, msg

        if self.seq_length < self.receptive_field:
            input_ = nn.functional.pad(
                input_,
                (self.receptive_field - self.seq_length, 0, 0, 0)
            )

        if self.gcn_true:
            if self.build_adp:
                if idx is None:

                    # this line computes the adjacency matrix adaptively
                    # by calling the function forward in the gc
                    adp = self.gc(self.idx)
                else:
                    adp = self.gc(idx)
            else:
                adp = self.predefined_adp
        
        # print('Forward...')
        # time.sleep(1)
        # print(adp[4])
        # time.sleep(3)
        # #sys.exit()

        # col=DataLoaderS.col
        # for i in range(adp.shape[0]):
        #     print('connections to node '+col[i]+': [',end='')
        #     counter=0
        #     for j in range(adp.shape[1]):
        #         if adp[i,j].item()>0:
        #             print(col[j],end='')
        #             if j<adp.shape[1]-1:
        #                 print(', ', end='')
        #             counter+=1
        #         if j==adp.shape[1]-1:
        #             print('] total=',counter)
        # sys.exit()

        x = self.start_conv(input_)
        skip = self.skip0(
            F.dropout(input_, self.dropout, training=self.training)
        )

        for i in range(self.layers):
            residual = x
            filter_ = self.filter_convs[i](x)
            filter_ = torch.tanh(filter_)
            gate = self.gate_convs[i](x)
            gate = torch.sigmoid(gate)
            x = filter_ * gate
            x = F.dropout(x, self.dropout, training=self.training) 
                    
            s = x
            s = self.skip_convs[i](s)
            skip = s + skip
            if self.gcn_true:
                x1, attention_matrix1 = self.gconv1[i](x, adp)
                x2, attention_matrix2 = self.gconv2[i](
                    x, adp.transpose(1, 0)
                )
                x = x1 + x2
                self.attention_scores = (
                    attention_matrix1 + attention_matrix2
                ) / 2
            else:
                x = self.residual_convs[i](x)

            x = x + residual[:, :, :, -x.size(3):]
            if idx is None:
                x = self.norm[i](x, self.idx)
            else:
                x = self.norm[i](x, idx)

        skip = self.skipE(x) + skip
        x = F.relu(skip)
        x = F.relu(self.end_conv_1(x))
        x = self.end_conv_2(x)
        if return_attention:
            return x, self.attention_scores
        return x

    def visualize_sample_attention_scores(self) -> None:
        """
        Plot the first ten nodes from the most recent forward pass.
        """
        if self.attention_scores is None:
            msg = 'Run forward with return_attention=True before plotting.'
            raise RuntimeError(msg)

        attention_scores = self.attention_scores[:10, :10]
        output_dir = Path(__file__).resolve().parent / 'explainability'
        output_dir.mkdir(parents=True, exist_ok=True)
        pyplot.figure(figsize=(10, 8))
        sns.heatmap(
            attention_scores.detach().cpu().numpy(),
            cmap='viridis', annot=True, fmt='.2f'
        )
        pyplot.title('Attention Scores')
        pyplot.xlabel('Nodes')
        pyplot.ylabel('Nodes')
        pyplot.savefig(
            output_dir / 'Attention_sample.pdf',
            format='pdf',
            bbox_inches='tight'
        )
        pyplot.show()

    def visualize_attention_scores(
            self,
            names: List[str],
            rows: List[int],
            columns: List[int],
            title: str,
            x_axis_label: str,
            y_axis_label: str
            ) -> None:
        """
        Plot selected rows and columns from the most recent attention matrix.
        List `names` must contain one label per graph node.

        :param names: List of graph nodes for attention scores.
        :param rows: List of row indices to get names for heatmap axes.
        :param columns: List of column indices to get names for heatmap axes.
        :param title: Plot title.
        """
        if self.attention_scores is None:
            msg = 'Run forward with return_attention=True before plotting.'
            raise RuntimeError(msg)

        if len(names) != self.num_nodes:
            msg = 'names must contain one label for every graph node.'
            raise ValueError(msg)

        if not rows or not columns:
            raise ValueError('rows and columns must not be empty.')
        if (
            min(rows) < 0 or max(rows) >= self.num_nodes or
            min(columns) < 0 or max(columns) >= self.num_nodes
        ):
            error_msg = 'rows and columns must contain valid node indices.'
            raise IndexError(error_msg)

        attention_scores = self.attention_scores[rows][:, columns].detach()
        max_attention = attention_scores.abs().max()
        if max_attention > 0:
            attention_scores /= max_attention

        figure_width = max(8, min(20, len(columns) * 0.35))
        figure_height = max(6, min(20, len(rows) * 0.35))
        pyplot.figure(figsize=(figure_width, figure_height))

        scores_array = attention_scores.cpu().numpy()
        x_tick_labels = [names[index] for index in columns]
        y_tick_labels = [names[index] for index in rows]

        heatmap = sns.heatmap(
            scores_array, cmap='OrRd', annot=False,
            xticklabels=x_tick_labels, yticklabels=y_tick_labels
            )

        pyplot.title(f'Attention Scores: {x_axis_label} - {y_axis_label}')
        heatmap.set_xticklabels(
            heatmap.get_xticklabels(), rotation=90, fontsize=9
        )
        heatmap.set_yticklabels(
            heatmap.get_yticklabels(), rotation=0, fontsize=9
        )

        heatmap.set_xlabel(x_axis_label, labelpad=12)
        heatmap.set_ylabel(y_axis_label, labelpad=12)

        # Explainability output folder
        fig_path = f'Attention_{title}.pdf'
        output_dir = Path(__file__).resolve().parent / 'explainability'
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save plot
        pyplot.savefig(output_dir / fig_path, format='pdf', bbox_inches='tight')
        pyplot.show()
