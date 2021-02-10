import logging

from allennlp.modules.matrix_attention import BilinearMatrixAttention
from allennlp.nn.util import masked_softmax, masked_max
import torch
from torch import nn
import torch.nn.functional as F

from .semantic_encoder import SemanticEncoder
from .gnn import RGCN


logger = logging.getLogger(__name__)


class RGCNSemanticEncoderBase(SemanticEncoder):
    def __init__(self, args):
        super().__init__(args)

        self.n_graph_attn_composition_layers = args.n_graph_attn_composition_layers
        self.output_size = self.transformer.config.hidden_size
        self.graph_dim = args.graph_dim

        if self.use_semantic_graph:
            self.emb_proj = nn.Linear(self.transformer.config.hidden_size, self.graph_dim)

            def get_gnn_instance(n_layers):
                return RGCN(
                    num_bases=args.graph_n_bases,
                    h_dim=self.graph_dim,
                    num_relations=self.num_relations,
                    num_hidden_layers=n_layers,
                    dropout=args.graph_dropout,
                    activation=self.activation,
                )

            self.rgcn = get_gnn_instance(args.n_graph_layers)
            if self.n_graph_attn_composition_layers > 0:
                self.composition_rgcn = get_gnn_instance(self.n_graph_attn_composition_layers)

            self.attn_biaffine = BilinearMatrixAttention(self.graph_dim, self.graph_dim, use_input_biases=True)
            self.attn_proj = nn.Linear(4 * self.graph_dim, self.graph_dim)

            self.graph_output_proj = nn.Linear(self.graph_dim, self.graph_dim)
            self.output_size += (2 if self.is_sentence_pair_task else 1) * self.graph_dim

            if self.args.post_combination_layernorm:
                self.post_combination_layernorm = nn.LayerNorm(self.output_size, eps=self.transformer.config.layer_norm_eps)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, sent_a_masks=None, sent_b_masks=None, graphs_a=None, gdata_a=None, graphs_b=None, gdata_b=None):
        """
        graphs_*: batched DGLGraph across sentences;
            bsz * DGLGraph
        g_data_*: dictinoary with values having shape:
            (bsz, ...)
        """
        self._check_input(input_ids, token_type_ids, attention_mask, sent_a_masks, sent_b_masks)
        if sent_a_masks is not None and attention_mask is not None:
            assert (sent_a_masks & attention_mask == sent_a_masks).all()
        if sent_b_masks is not None and attention_mask is not None:
            assert (sent_b_masks & attention_mask == sent_b_masks).all()

        last_layers, pooled_output = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask.long() if attention_mask is not None else None,
            token_type_ids=token_type_ids,
        )
        transformer_output = self.dropout(pooled_output)  # (bsz, transformer_dim)

        # rgcn
        rgcn_output = None
        if self.use_semantic_graph:
            batch_size = input_ids.shape[0]

            graphs_a_empty = len(graphs_a.nodes) == 0
            if not graphs_a_empty:
                node_embs_a, node_emb_mask_a = self.pool_node_embeddings(last_layers, sent_a_masks, gdata_a, graphs_a.batch_num_nodes)
                node_embs_a = self.propagate_graph(graphs_a, node_embs_a, node_emb_mask_a)

            if self.is_sentence_pair_task:
                graphs_b_empty = len(graphs_b.nodes) == 0
                if not graphs_b_empty:
                    node_embs_b, node_emb_mask_b = self.pool_node_embeddings(last_layers, sent_b_masks, gdata_b, graphs_b.batch_num_nodes)
                    node_embs_b = self.propagate_graph(graphs_b, node_embs_b, node_emb_mask_b)
                    if not graphs_a_empty:
                        node_embs_a, node_embs_b = self.interact_graphs(graphs_a, graphs_b, node_embs_a, node_embs_b, node_emb_mask_a, node_emb_mask_b)

            if graphs_a_empty:
                rgcn_output_a = torch.zeros(batch_size, self.graph_dim, dtype=torch.float, device=last_layers.device)
            else:
                rgcn_output_a = self.pool_graph(node_embs_a, node_emb_mask_a)

            if self.is_sentence_pair_task:
                if graphs_b_empty:
                    rgcn_output_b = torch.zeros(batch_size, self.graph_dim, dtype=torch.float, device=last_layers.device)
                else:
                    rgcn_output_b = self.pool_graph(node_embs_b, node_emb_mask_b)

            rgcn_output_a = self.activation(self.graph_output_proj(rgcn_output_a))
            if self.args.final_dropout:
                rgcn_output_a = self.dropout(rgcn_output_a)
            if self.is_sentence_pair_task:
                rgcn_output_b = self.activation(self.graph_output_proj(rgcn_output_b))
                if self.args.final_dropout:
                    rgcn_output_b = self.dropout(rgcn_output_b)

            rgcn_output = [rgcn_output_a]
            if self.is_sentence_pair_task:
                rgcn_output.append(rgcn_output_b)
            rgcn_output = torch.cat(tuple(rgcn_output), dim=-1)

        return transformer_output, rgcn_output

    def propagate_graph(self, graph, node_embeddings, node_embeddings_mask):
        """
        Parameters:
            node_embs: (bsz, max_num_nodes, emb_dim)
            node_embeddings_mask: (bsz, max_num_nodes)

        Returns:
            node_embs: (bsz, max_num_nodes, emb_dim)
        """
        node_embeddings = self.flatten_node_embeddings(node_embeddings, node_embeddings_mask)
        node_embeddings = self.activation(self.emb_proj(node_embeddings))

        node_embeddings = self.rgcn(graph, node_embeddings)

        return self.unflatten_node_embeddings(node_embeddings, node_embeddings_mask)

    def interact_graphs(self, graph_a, graph_b, node_embs_a, node_embs_b, node_emb_mask_a, node_emb_mask_b):
        """
        Parameters:
            node_embs_{a,b}: (bsz, n_nodes_{a,b}, graph_dim)
            node_emb_mask_{a,b}: (bsz, n_nodes_{a,b})
        """
        orig_node_embs_a, orig_node_embs_b = node_embs_a, node_embs_b

        # attn: (bsz, n_nodes_a, n_nodes_b)
        attn = self.attn_biaffine(node_embs_a, node_embs_b)

        normalized_attn_a = masked_softmax(attn, node_emb_mask_a.unsqueeze(2), dim=1)  # (bsz, n_nodes_a, n_nodes_b)
        attended_a = normalized_attn_a.transpose(1, 2).bmm(node_embs_a)  # (bsz, n_nodes_b, graph_dim)
        new_node_embs_b = torch.cat([node_embs_b, attended_a, node_embs_b - attended_a, node_embs_b * attended_a], dim=-1)  # (bsz, n_nodes_b, graph_dim * 4)
        new_node_embs_b = self.activation(self.attn_proj(new_node_embs_b))  # (bsz, n_nodes_b, graph_dim)

        normalized_attn_b = masked_softmax(attn, node_emb_mask_b.unsqueeze(1), dim=2)  # (bsz, n_nodes_a, n_nodes_b)
        attended_b = normalized_attn_b.bmm(node_embs_b)  # (bsz, n_nodes_a, graph_dim)
        new_node_embs_a = torch.cat([node_embs_a, attended_b, node_embs_a - attended_b, node_embs_a * attended_b], dim=-1)  # (bsz, n_nodes_a, graph_dim * 4)
        new_node_embs_a = self.activation(self.attn_proj(new_node_embs_a))  # (bsz, n_nodes_b, graph_dim)

        node_embs_a = self.flatten_node_embeddings(new_node_embs_a, node_emb_mask_a)
        node_embs_b = self.flatten_node_embeddings(new_node_embs_b, node_emb_mask_b)

        if self.n_graph_attn_composition_layers:
            node_embs_a = self.composition_rgcn(graph_a, node_embs_a)
            node_embs_b = self.composition_rgcn(graph_b, node_embs_b)

        node_embs_a = self.unflatten_node_embeddings(node_embs_a, node_emb_mask_a)
        node_embs_b = self.unflatten_node_embeddings(node_embs_b, node_emb_mask_b)

        # If the other graph is empty, we don't do any attention at all and use the original embedding
        node_embs_a = torch.where(node_emb_mask_b.any(1, keepdim=True).unsqueeze(-1), node_embs_a, orig_node_embs_a)
        node_embs_b = torch.where(node_emb_mask_a.any(1, keepdim=True).unsqueeze(-1), node_embs_b, orig_node_embs_b)

        return node_embs_a, node_embs_b

    def pool_graph(self, node_embs, node_emb_mask):
        """
        Parameters:
            node_embs: (bsz, n_nodes, graph_dim)
            node_emb_mask: (bsz, n_nodes)

        Returns:
            (bsz, graph_dim (*2))
        """
        node_emb_mask = node_emb_mask.unsqueeze(-1)
        output = masked_max(node_embs, node_emb_mask, 1)
        output = torch.where(node_emb_mask.any(1), output, torch.zeros_like(output))
        return output

    def flatten_node_embeddings(self, node_embeddings, node_embeddings_mask):
        return node_embeddings[node_embeddings_mask]

    def unflatten_node_embeddings(self, node_embeddings, node_embeddings_mask):
        output_node_embeddings = node_embeddings.new_zeros(
            node_embeddings_mask.shape[0], node_embeddings_mask.shape[1], node_embeddings.shape[-1]
        )
        output_node_embeddings[node_embeddings_mask] = node_embeddings
        return output_node_embeddings

    ###############################################################

    @staticmethod
    def add_model_specific_args(parser, root_dir):
        SemanticEncoder.add_model_specific_args(parser, root_dir)

        parser.add_argument("--graph_dim",
                            default=16,
                            type=int,
                            help="The dimension for graph nodes.")
        parser.add_argument("--graph_n_bases",
                            default=-1,
                            type=int,
                            help="The number of bases for RGCN weight decomposition.")
        parser.add_argument("--n_graph_layers",
                            default=2,
                            type=int,
                            help="The number of RGCN updates.")
        parser.add_argument("--graph_dropout",
                            default=0.0,
                            type=float,
                            help="The RGCN dropout.")
        parser.add_argument("--n_graph_attn_composition_layers",
                            default=0,
                            type=int,
                            help="Whether to use an additional composition layer after the graph attenion.")
        parser.add_argument("--post_combination_layernorm",
                            action='store_true',
                            help="Whether to add a layer norm to the combined transformer & graph encoder output.")
        parser.add_argument("--final_dropout",
                            action='store_true',
                            help="Whether to add a final dropout to the graph encoder output before combining it with the transformer output.")

        return parser


class RGCNSemanticEncoder(RGCNSemanticEncoderBase):
    def __init__(self, args):
        super().__init__(args)

        if not hasattr(self.args, 'separate_losses_ratio'):
            self.args.separate_losses_ratio = None  # for backward compatibility

        if self.args.separate_losses_ratio is not None:
            assert self.use_semantic_graph
            self.transformer_classifier = nn.Linear(self.transformer.config.hidden_size, self.num_labels)
            self.gnn_classifier = nn.Linear(self.output_size - self.transformer.config.hidden_size, self.num_labels)
        else:
            self.classifier = nn.Linear(self.output_size, self.num_labels)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, sent_a_masks=None, sent_b_masks=None, graphs_a=None, gdata_a=None, graphs_b=None, gdata_b=None, label_gdata=None, metadata=None):
        transformer_output, rgcn_output = super().forward(input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask, sent_a_masks=sent_a_masks, sent_b_masks=sent_b_masks, graphs_a=graphs_a, gdata_a=gdata_a, graphs_b=graphs_b, gdata_b=gdata_b)

        if self.args.separate_losses_ratio is not None:
            transformer_logits = self.transformer_classifier(transformer_output)
            gnn_logits = self.gnn_classifier(rgcn_output)
            return {'logits': transformer_logits, 'gnn_logits': gnn_logits}
        else:
            output = [transformer_output]
            output.extend([rgcn_output] if self.use_semantic_graph else [])
            output = torch.cat(tuple(output), dim=-1)
            if self.use_semantic_graph and self.args.post_combination_layernorm:
                output = self.post_combination_layernorm(output)
            logits = self.classifier(output)
            return {'logits': logits}

    def compute_loss(self, output_dict, labels):
        logits = output_dict["logits"]
        if self.output_mode == "classification":
            loss = F.cross_entropy(logits.view(-1, logits.shape[-1]), labels.view(-1))
        elif self.output_mode == "regression":
            loss = F.mse_loss(logits.view(-1), labels.view(-1))

        if self.args.separate_losses_ratio is not None:
            logits = output_dict["gnn_logits"]
            if self.output_mode == "classification":
                gnn_loss = F.cross_entropy(logits.view(-1, logits.shape[-1]), labels.view(-1))
            elif self.output_mode == "regression":
                gnn_loss = F.mse_loss(logits.view(-1), labels.view(-1))
            loss = (1 - self.args.separate_losses_ratio) * loss + self.args.separate_losses_ratio * gnn_loss

        return loss

    ################################################################

    @staticmethod
    def add_model_specific_args(parser, root_dir):
        RGCNSemanticEncoderBase.add_model_specific_args(parser, root_dir)
        parser.add_argument("--separate_losses_ratio",
                            default=None,
                            type=float,
                            help="If specificed, RoBERTa and RGCN heads as treated as separate output and the 2 losses are combined with this ratio.")

        return parser
