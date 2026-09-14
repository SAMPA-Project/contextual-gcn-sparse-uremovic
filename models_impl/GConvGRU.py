
import torch.nn
from torch_geometric_temporal.nn.recurrent import GConvGRU as GConvGRU_impl
import torch.nn.functional as F
from torch_geometric.nn import ChebConv

# base model: GCRNN encoder-decoder conditioned on a contextual raster/vector,
# with a residual (skip) connection from the context embedding into the MLP head.
class GCRNNBase(torch.nn.Module):
    def __init__(self, filters, layers, cnn_filters, node_features, forecast_features, known_features, ctx_features, history, horizon, khops=2):
        super(GCRNNBase, self).__init__()
        self.encoders = torch.nn.ModuleList([GConvGRU_impl(filters, filters, K=khops) if l > 0 else GConvGRU_impl(node_features, filters, K=khops) for l in range(layers)])
        self.decoders = torch.nn.ModuleList([GConvGRU_impl(filters, filters, K=khops) if l > 0 else GConvGRU_impl(known_features, filters, K=khops) for l in range(layers)])
        self.cnni = torch.nn.Conv2d(ctx_features, filters, kernel_size=5)
        self.cnnh = torch.nn.Conv2d(filters, filters, kernel_size=5)
        self.cnno = torch.nn.Conv2d(filters, filters, kernel_size=5)
        self.gcn = ChebConv(filters, filters, K=2)
        self.lineari = torch.nn.Linear(filters + filters, filters)
        self.linearh = torch.nn.Linear(filters, filters)
        self.linearo = torch.nn.Linear(filters, forecast_features)
        self.clineari = torch.nn.Linear(ctx_features, filters)
        self.clinearh = torch.nn.Linear(filters, filters)
        self.clinearo = torch.nn.Linear(filters, filters)
        self.node_features = node_features
        self.ctx_features = ctx_features
        self.known_features = known_features
        self.forecast_features = forecast_features
        self.history = history
        self.horizon = horizon
        self.L = layers
        self.filters = filters

    def __mlp_regressor(self, h):
        h = self.lineari(h)
        h = F.relu(h)
        h = self.linearh(h)
        h = F.relu(h)
        h = self.linearo(h)
        return h

    def __cnn(self, ctx):
        if ctx.ndim == 2:
            c = self.clineari(ctx)
            c = F.relu(c)
            c = self.clinearh(c)
            c = F.relu(c)
            c = self.clinearo(c)
            c = F.relu(c)
            return c
        else:
            c = self.cnni(ctx)
            c = F.relu(c)
            c = self.cnnh(c)
            c = F.relu(c)
            c = self.cnno(c)
            c = F.relu(c)
            c = F.softmax(c, dim=1)
            return c.reshape(shape=(c.shape[0], c.shape[1]))

    def forward(self, x, ctx, edge_index, edge_weight):
        i = 0
        hs = []
        cnn_ = self.__cnn(ctx) # extract features from raster data
        aux = self.gcn(cnn_, edge_index, edge_weight) # pass aux features over graph
        H = [aux for l in range(self.L)] # condition rnn with aux
        for sample in torch.swapaxes(x, 0, 1):
            if i >= self.history:
                H[0] = self.decoders[0](sample[:,-self.known_features:], edge_index, edge_weight, H[0])
                for l in range(1, self.L):
                    H[l] = self.decoders[l](H[l-1], edge_index, edge_weight, H[l])
            else:
                H[0] = self.encoders[0](sample, edge_index, edge_weight, H[0])
                for l in range(1, self.L):
                    H[l] = self.encoders[l](H[l-1], edge_index, edge_weight, H[l])
            h = F.relu(H[self.L-1])
            h = self.__mlp_regressor(torch.cat([h, aux],axis=-1))
            hs.append(torch.clone(h))
            i += 1
        return torch.stack(hs, dim=1)

# virtual-node model: real-node GCRNN passes its hidden states, plus raster/vector
# context, into per-layer graph convs over the full (real+virtual) graph, whose
# outputs condition a second GCRNN decoder that forecasts the virtual nodes.
class GCRNNVirtual(torch.nn.Module):
    def __init__(self, filters, layers, khops, node_features, forecast_features, known_features, ctx_features, history, horizon):
        super(GCRNNVirtual, self).__init__()
        self.encoders = torch.nn.ModuleList([GConvGRU_impl(filters, filters, khops) if l > 0 else GConvGRU_impl(node_features, filters, khops) for l in range(layers)])
        self.decoders = torch.nn.ModuleList([GConvGRU_impl(filters, filters, khops) if l > 0 else GConvGRU_impl(known_features, filters, khops) for l in range(layers)])
        self.cnni = torch.nn.Conv2d(ctx_features, filters, kernel_size=5)
        self.cnnh = torch.nn.Conv2d(filters, filters, kernel_size=5)
        self.cnno = torch.nn.Conv2d(filters, filters, kernel_size=5)
        self.gcn = ChebConv(filters, filters, 2)
        self.linear = torch.nn.Linear(filters, filters)
        self.virtual_gcns = torch.nn.ModuleList([ChebConv(filters + filters, filters, 2) for l in range(layers)])
        self.virtual_linears = torch.nn.ModuleList([torch.nn.Linear(filters + filters, filters) for l in range(layers)])
        self.lineari = torch.nn.Linear(filters + filters, filters)
        self.linearh = torch.nn.Linear(filters, filters)
        self.linearo = torch.nn.Linear(filters, forecast_features)
        self.clineari = torch.nn.Linear(ctx_features, filters)
        self.clinearh = torch.nn.Linear(filters, filters)
        self.clinearo = torch.nn.Linear(filters, filters)
        self.node_features = node_features
        self.ctx_features = ctx_features
        self.known_features = known_features
        self.history = history
        self.horizon = horizon
        self.forecast_features = forecast_features
        self.L = layers
        self.filters = filters

    def __cnn(self, ctx):
        if ctx.ndim == 2:
            c = self.clineari(ctx)
            c = F.relu(c)
            c = self.clinearh(c)
            c = F.relu(c)
            c = self.clinearo(c)
            c = F.relu(c)
            return c
        else:
            c = self.cnni(ctx)
            c = F.relu(c)
            c = self.cnnh(c)
            c = F.relu(c)
            c = self.cnno(c)
            c = F.relu(c)
            #c = F.softmax(c, dim=1)
            return c.reshape(shape=(c.shape[0], c.shape[1]))

    def __mlp_regressor(self, h):
        h = self.lineari(h)
        h = F.relu(h)
        h = self.linearh(h)
        h = F.relu(h)
        h = self.linearo(h)
        return h

    def forward(self, x, x_virtual, ctx, ctx_virtual, edge_index, edge_index_virtual, edge_weight, edge_weight_virtual, mask_real_diagonal_batched, mask_virtual_diagonal_batched, h_real_):
        i = 0
        hs = []
        cnn_ = self.__cnn(ctx_virtual) # extract features from raster data (ALL nodes)
        cnn_real_ = self.__cnn(ctx) # extract features from raster data (real nodes)
        aux_real = self.gcn(cnn_real_, edge_index, edge_weight) # pass aux features over REAL graph
        H = [aux_real.clone() for l in range(self.L)] # condition rnn with aux
        h_real = None
        for sample in torch.swapaxes(x, 0, 1):
            if i >= self.history:
                if h_real is None:
                    h_real = [torch.clone(H[l]) for l in range(self.L)] # here begins forecast (history window ended). save hidden states from (real node) gcrnn, to condition virtual decoder gcrnns with
                H[0] = self.decoders[0](sample[:,-self.known_features:], edge_index, edge_weight, H[0])
                for l in range(1, self.L):
                    H[l] = self.decoders[l](H[l-1], edge_index, edge_weight, H[l])
                h = F.relu(H[self.L-1])
                h = self.__mlp_regressor(torch.cat([h,aux_real],axis=-1))
                hs.append(torch.clone(h))
            else:
                H[0] = self.encoders[0](sample, edge_index, edge_weight, H[0])
                for l in range(1, self.L):
                    H[l] = self.encoders[l](H[l-1], edge_index, edge_weight, H[l])
            i += 1
        real_idx = torch.tensor(mask_real_diagonal_batched, dtype=torch.bool, device=x.device)
        for l in range(self.L):
            h_real_[l][real_idx] = h_real[l]

        aux_virtual = [self.virtual_gcns[l](torch.cat([cnn_, h_real_[l]], axis=-1), edge_index_virtual, edge_weight_virtual) for l in range(self.L)] # pass aux features over VIRTUAL graph, each layer has own gcn
        regressor_aux = self.linear(cnn_)
        H = aux_virtual # condition virtual gcrnn with cnn and hidden state from real nodes data passed around graph
        hs_virtual = []
        for sample in torch.swapaxes(x_virtual, 0, 1)[self.history:]: # only go through forecast
            H[0] = self.decoders[0](sample[:,-self.known_features:], edge_index_virtual, edge_weight_virtual, H[0])
            for l in range(1, self.L):
                H[l] = self.decoders[l](H[l-1], edge_index_virtual, edge_weight_virtual, H[l])
            h = F.relu(H[self.L-1])
            h = self.__mlp_regressor(torch.cat([h,regressor_aux],axis=-1))
            hs_virtual.append(torch.clone(h))
        return torch.cat([torch.stack(hs, dim=1), torch.stack(hs_virtual, dim=1)[mask_virtual_diagonal_batched]]) # just stack here virtual forecasts after real node forecasts, prepare suitable y instead of interleaving y forecast here
