import torch
from typing import Optional, Literal
from dgl.readout import mean_nodes

from etseed.model.se3_transformer.se3_transformer.model.basis import get_basis, update_basis_with_fused
from etseed.model.se3_transformer.se3_transformer.model.layers.attention import AttentionBlockSE3
from etseed.model.se3_transformer.se3_transformer.model.layers.convolution import ConvSE3, ConvSE3FuseLevel
from etseed.model.se3_transformer.se3_transformer.model.layers.norm import NormSE3
from etseed.model.se3_transformer.se3_transformer.model.fiber import Fiber
from etseed.utils.utils import build_graph
from etseed.model.diffusion.positional_embedding import SinusoidalPosEmb

class ExtendedModule(torch.nn.Module):
    @property
    def device(self):
        return next(self.parameters()).device

class ExtendedSequential(torch.nn.Sequential):
    def append(self, module):
        index = len(self)
        self.add_module(str(index), module)

    def append_list(self, modules):
        assert isinstance(modules, torch.nn.Sequential) or isinstance(modules, (list, tuple))
        for module in modules:
            self.append(module)

class Sequential(ExtendedSequential):
    """ Sequential module with arbitrary forward args and kwargs. Used to pass graph, basis and edge features. """

    def forward(self, input, *args, **kwargs):
        for module in self:
            input = module(input, *args, **kwargs)
        return input

class EquivariantNet(ExtendedModule):
    def __init__(self,
                 num_layers: int,
                 num_degrees: int,
                 num_channels: int,
                 num_heads: int,
                 channels_div: int,
                 fiber_in: Fiber,
                 fiber_out: Fiber,
                 fiber_edge: Fiber = Fiber({0: 0}),
                 pooling: Optional[Literal['avg', 'max']] = None,
                 norm: bool = True,
                 use_layer_norm: bool = True,
                 tensor_cores: bool = False,
                 low_memory: bool = False,
                 voxelize = True,
                 voxel_size = 0.01,
                 radius_threshold = 0.02,
                 final_layer_norm = False,
                 **kwargs):
        
        super().__init__()
        self.num_layers = num_layers
        self.fiber_edge = fiber_edge
        self.num_heads = num_heads
        self.channels_div = channels_div
        self.pooling = pooling

        self.voxelize = voxelize
        self.voxel_size = voxel_size
        self.radius_threshold = radius_threshold

        fiber_hidden = Fiber.create(num_degrees, num_channels)
        # we set fiber out here, the output channel can be changed

        self.fiber_in = fiber_in
        self.fiber_out = fiber_out
        self.max_degree = max(*fiber_in.degrees, *fiber_hidden.degrees, *fiber_out.degrees)
        self.tensor_cores = tensor_cores
        self.low_memory = low_memory

        if low_memory:
            self.fuse_level = ConvSE3FuseLevel.NONE
        else:
            # Fully fused convolutions when using Tensor Cores (and not low memory mode)
            self.fuse_level = ConvSE3FuseLevel.FULL if tensor_cores else ConvSE3FuseLevel.PARTIAL

        graph_modules = []
        for i in range(num_layers):
            graph_modules.append(AttentionBlockSE3(fiber_in=fiber_in,
                                                   fiber_out=fiber_hidden,
                                                   fiber_edge=fiber_edge,
                                                   num_heads=num_heads,
                                                   channels_div=channels_div,
                                                   use_layer_norm=use_layer_norm,
                                                   max_degree=self.max_degree,
                                                   fuse_level=self.fuse_level,
                                                   low_memory=low_memory))
            if norm:
                graph_modules.append(NormSE3(fiber_hidden))
            fiber_in = fiber_hidden

        graph_modules.append(ConvSE3(fiber_in=fiber_in,
                                     fiber_out=fiber_out,
                                     fiber_edge=fiber_edge,
                                     self_interaction=True,
                                     use_layer_norm=use_layer_norm,
                                     max_degree=self.max_degree,
                                     fuse_level=self.fuse_level,
                                     low_memory=low_memory))
        if final_layer_norm:
            graph_modules.append(NormSE3(fiber_out))
        self.graph_modules = Sequential(*graph_modules)

    def forward(self, inputs, given_graph=None, given_basis=None, **kwargs):
        xyz = inputs["xyz"]
        feature = inputs["feature"]
        if isinstance(xyz, torch.Tensor):
            batch_size = xyz.shape[0]
        else:
            batch_size = len(xyz)

        if not given_graph:
            batch_graph, node_feats, edge_feats, pcds, raw_node_feats = build_graph(xyz, feature, dist_threshold=self.radius_threshold, voxelize=self.voxelize, voxel_size=self.voxel_size, fiber_in=self.fiber_in)
        else:
            batch_graph = given_graph["batch_graph"] 
            node_feats = given_graph["node_feats"]
            edge_feats = given_graph["edge_feats"]
            pcds = given_graph["pcds"]
            raw_node_feats = given_graph["raw_node_feats"]  # raw node feats are flattened node_feats
 
        if not given_basis:
            # Compute bases in case they weren't precomputed as part of the data loading
            basis = get_basis(batch_graph.edata['rel_pos'], max_degree=self.max_degree, compute_gradients=False,
                                    use_pad_trick=self.tensor_cores and not self.low_memory,
                                    amp=torch.is_autocast_enabled())

            # Add fused bases (per output degree, per input degree, and fully fused) to the dict
            basis = update_basis_with_fused(basis, self.max_degree, use_pad_trick=self.tensor_cores and not self.low_memory,
                                            fully_fused=self.fuse_level == ConvSE3FuseLevel.FULL)
        else:
            basis = given_basis
        
        node_feats = self.graph_modules(node_feats, edge_feats, graph=batch_graph, basis=basis)
        
        output = node_feats[list(node_feats.keys())[0]].reshape(node_feats[list(node_feats.keys())[0]].shape[0], -1)
        for type_l in list(node_feats.keys())[1:]:
            output = torch.cat(
                [output, node_feats[type_l].reshape(node_feats[type_l].shape[0], -1)], dim=1
            )
        assert output.shape[-1] == self.fiber_out.num_features    # n, output_dim (no bs!)
        
        if self.pooling:
            with batch_graph.local_scope():
                batch_graph.ndata["h"] = output
                output = mean_nodes(batch_graph, "h")
            return {
                "batch_graph": batch_graph,
                "node_feats": node_feats,
                "edge_feats": edge_feats,
                "pcds": pcds,
                "raw_node_feats": raw_node_feats,
            }, basis, output # list, list, basis, tensor

        # reshape raw_features to batch_sizes
        reshaped_output = []
        idx = 0
        for i in range(batch_size):
            reshaped_output.append(output[idx:(idx+pcds[i].shape[0])])
            idx += pcds[i].shape[0]

        return {
            "batch_graph": batch_graph,
            "node_feats": node_feats,
            "edge_feats": edge_feats,
            "pcds": pcds,
            "raw_node_feats": raw_node_feats,
        }, basis, reshaped_output # list, list, basis, list


class SE3Backbone(ExtendedModule):
    def __init__(
        self,
        fiber_in: Fiber = Fiber({
                "0": 3,
            }),
        fiber_out: Fiber = Fiber({
                "0": 4,
                "1": 4,
                "2": 4,
                "3": 4,
            }),
        num_layers: int = 2,
        num_degrees: int = 4,
        num_channels: int = 8,
        num_heads: int = 2,
        channels_div: int = 2,
        voxelize: bool = True,
        voxel_size: float = 0.02,
        radius_threshold: float = 0.04,
        pooling: bool = False,
    ):
        super().__init__()
        self.net = EquivariantNet(
            num_layers=num_layers, 
            num_degrees=num_degrees, 
            num_channels=num_channels, 
            num_heads=num_heads, 
            fiber_in=fiber_in,
            fiber_out=fiber_out,  # 1 heatmap, 3 axises
            channels_div=channels_div, 
            voxelize=voxelize,
            voxel_size=voxel_size,
            radius_threshold=radius_threshold,
            pooling=pooling,
        )

    
    # ['xyz', 'rgb', 'seg_center', 'axes']
    # ADD: ['action']
    def forward(self, inputs):
        if "feature" not in inputs.keys():
            inputs["feature"] = inputs["rgb"]
        given_graph, basis, feature = self.net(inputs)  
        # node_feats are the (voxelized) feature vector, and feature are feed-forward output
        return {
            "xyz": given_graph["pcds"],
            "feature": feature,
            "raw_inputs": inputs,
            "frame_related_states": inputs["frame_related_states"] if "frame_related_states" in inputs.keys() else None,
            "basis": basis,
            "given_graph": given_graph,
        }