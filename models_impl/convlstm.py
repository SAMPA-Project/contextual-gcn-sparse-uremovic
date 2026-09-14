import torch.nn as nn
import torch
import torch.nn.functional as F


class ConvLSTMCell(nn.Module):

    def __init__(self, input_dim, hidden_dim, kernel_size, bias):
        """
        Initialize ConvLSTM cell.

        Parameters
        ----------
        input_dim: int
            Number of channels of input tensor.
        hidden_dim: int
            Number of channels of hidden state.
        kernel_size: (int, int)
            Size of the convolutional kernel.
        bias: bool
            Whether or not to add the bias.
        """

        super(ConvLSTMCell, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.kernel_size = kernel_size
        self.padding = kernel_size[0] // 2, kernel_size[1] // 2
        self.bias = bias

        self.conv = nn.Conv2d(in_channels=self.input_dim + self.hidden_dim,
                              out_channels=4 * self.hidden_dim,
                              kernel_size=self.kernel_size,
                              padding=self.padding,
                              bias=self.bias)

    def forward(self, input_tensor, cur_state):
        h_cur, c_cur = cur_state

        combined = torch.cat([input_tensor, h_cur], dim=1)  # concatenate along channel axis

        combined_conv = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)

        return h_next, c_next

    def init_hidden(self, batch_size, image_size):
        height, width = image_size
        return (torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device),
                torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device))


class ConvLSTM_ctxCNN_encoderdecoder(nn.Module):
    """
    ConvLSTM encoder-decoder with CNN context initialization.

    Mirrors GCRNNBase:
      - CNN extracts spatial features from context raster
      - Encoder runs over history frames, hidden state initialized from CNN context
      - Decoder runs over horizon, fed known features (e.g. NWP forecasts)
      - MLP regressor maps final hidden state to output channels

    Input shapes:
        x   : (B, T_total, C_in, H, W)   T_total = history + horizon
        ctx : (B, C_ctx, H, W)   context raster (same spatial res or downsampled)

    Output shape:
        (B, horizon, C_out, H, W)
    """

    def __init__(
        self,
        filters,
        layers,
        node_features,       # input channels per frame  (C_in)
        forecast_features,   # output channels            (C_out)
        known_features,      # channels available during decoding (last N channels of x)
        ctx_features,        # channels in context raster (C_ctx)
        history,
        horizon,
        kernel_size=(3, 3),
    ):
        super().__init__()

        self.node_features     = node_features
        self.forecast_features = forecast_features
        self.known_features    = known_features
        self.ctx_features      = ctx_features
        self.history           = history
        self.horizon           = horizon
        self.L                 = layers
        self.filters           = filters

        # ── encoder cells (history) ───────────────────────────────────────────
        self.encoders = nn.ModuleList([
            ConvLSTMCell(
                input_dim  = node_features if l == 0 else filters,
                hidden_dim = filters,
                kernel_size= kernel_size,
                bias       = True,
            )
            for l in range(layers)
        ])

        # ── decoder cells (horizon) ───────────────────────────────────────────
        # first decoder layer takes known_features channels as input
        self.decoders = nn.ModuleList([
            ConvLSTMCell(
                input_dim  = known_features if l == 0 else filters,
                hidden_dim = filters,
                kernel_size= kernel_size,
                bias       = True,
            )
            for l in range(layers)
        ])

        # ── context CNN ───────────────────────────────────────────────────────
        self.cnni = nn.Conv2d(ctx_features,       filters,       kernel_size=5, padding=2)
        self.cnnh = nn.Conv2d(filters,         filters,  kernel_size=5, padding=2)
        self.cnno = nn.Conv2d(filters,    filters,  kernel_size=5, padding=2)

        self.ctx_proj = nn.Linear(filters, filters)

        # ── output projection ─────────────────────────────────────────────────
        self.output_conv = nn.Conv2d(filters, forecast_features, kernel_size=1)

    # ── private helpers ───────────────────────────────────────────────────────

    def __cnn(self, ctx):
        """
        ctx : (B, C_ctx, H, W)
        out : (B, filters, H, W)
        """
        c = F.relu(self.cnni(ctx))
        c = F.relu(self.cnnh(c))
        c = F.relu(self.cnno(c))
        c = F.softmax(c, dim=1)
        return c   # (B, filters, H, W)

    def __init_hidden_from_ctx(self, ctx_map):
        """
        ctx_map : (B, filters, H, W)
        returns : list of (h, c) each (B, filters, H, W)
        """
        c0 = torch.zeros_like(ctx_map)
        return [(ctx_map.clone(), c0.clone()) for _ in range(self.L)]

    # ── forward ───────────────────────────────────────────────────────────────

    def forward(self, x, ctx):
        """
        x   : (B, T_total, C_in, H, W)   T_total = history + horizon
        ctx : (B, C_ctx, H, W)

        returns : (B, horizon, C_out, H, W)
        """
        B, T_total, C, H, W = x.shape

        # ── context → initial hidden state ───────────────────────────────────
        ctx_vec   = self.__cnn(ctx)                        # (B, cnn_filters//4)
        states    = self.__init_hidden_from_ctx(ctx_vec)
        # states: list of L x (h, c) each (B, filters, H, W)

        hs = []

        for t in range(T_total):
            frame = x[:, t]   # (B, C, H, W)

            if t < self.history:
                # ── encoder ──────────────────────────────────────────────────
                inp = frame
                for l in range(self.L):
                    h, c = self.encoders[l](inp, states[l])
                    states[l] = (h, c)
                    inp = h

            else:
                # ── decoder ──────────────────────────────────────────────────
                # feed only known features (last known_features channels)
                inp = frame[:, -self.known_features:]
                for l in range(self.L):
                    h, c = self.decoders[l](inp, states[l])
                    states[l] = (h, c)
                    inp = h

                # project top layer hidden state to output
                out = F.relu(states[self.L - 1][0])        # (B, filters, H, W)
                out = self.output_conv(out)                 # (B, C_out, H, W)
                hs.append(out)

        return torch.stack(hs, dim=1)   # (B, hist+horizon, C_out, H, W)
