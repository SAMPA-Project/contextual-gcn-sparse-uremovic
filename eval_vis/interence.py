

import torch
import numpy as np

from training.training import get_device

def restore_original_order(tensor, mask_real, mask_virtual):
    num_real = mask_real.sum()

    tensor_real = tensor[:num_real]
    tensor_virtual = tensor[num_real:]

    restored = torch.empty(
        mask_real.shape[0],
        *tensor.shape[1:],
        device=tensor.device,
        dtype=tensor.dtype
    )

    restored[mask_real] = tensor_real
    restored[mask_virtual] = tensor_virtual

    return restored

def inference(model, test_dataset, station_context, BATCH_SIZE, ids_virtual=[], reconnect_paths=False):
    if len(ids_virtual) == 0 or ids_virtual is None:
        virtual_node_training=False
    else:
        virtual_node_training=True
    device = get_device()
    model.to(device)
    ctx_single = station_context

    nnodes = ctx_single.shape[0]
    nedges = test_dataset.edge_index.shape[1]
    edge_index_single = test_dataset[0].edge_index
    edge_attr_single = test_dataset[0].edge_attr

    if ctx_single.ndim == 2:
        ctx_s = ctx_single
        station_context = torch.tensor(np.concatenate([ctx_s for i in range(BATCH_SIZE)], axis=0))
        station_context_test = torch.tensor(np.concatenate([ctx_s for i in range(BATCH_SIZE)], axis=0))
    else:
        x_ = int(np.sqrt(ctx_single.shape[1]))
        ctx_s = ctx_single.reshape(nnodes, x_, x_, ctx_single.shape[-1])
        station_context = torch.tensor(np.concatenate([ctx_s for i in range(BATCH_SIZE)], axis=0).swapaxes(3,1))
        station_context_test = torch.tensor(np.concatenate([ctx_s for i in range(BATCH_SIZE)], axis=0).swapaxes(3,1))
    station_context = station_context.to(device)
    station_context_test = station_context_test.to(device)

    y_hat_ = []
    y_ = []
    with torch.no_grad():
        shuffled_array = np.arange(0, test_dataset.snapshot_count-1, 1) # no shuffling in inference
        for i in range(0, len(shuffled_array)-BATCH_SIZE-1, BATCH_SIZE):
            x = test_dataset[shuffled_array[i:i+BATCH_SIZE]].x
            x = x.reshape(-1, x.shape[-2], x.shape[-1])
            y = test_dataset[shuffled_array[i:i+BATCH_SIZE]].y
            y = y.reshape(-1, y.shape[-2], y.shape[-1])
            ei = np.tile(edge_index_single, BATCH_SIZE)  # (2, B*E)
            offset = np.arange(BATCH_SIZE) * nnodes
            offset = np.repeat(offset, nedges)
            ei = ei + offset[None, :]
            ei = torch.tensor(ei, dtype=torch.int64)
            ew = edge_attr_single.repeat(BATCH_SIZE)

            if virtual_node_training:
                from data_handling.graph_virtualization import virtualize_graph
                h_real_, x_real, y, y_real, ei_real, ew_real, station_context_real, mask_real_diagonal_batched, mask_virtual_diagonal_batched = virtualize_graph(x, y, ei, ew, station_context_test, ids_virtual, nnodes, BATCH_SIZE, device, model.L, model.filters, reconnect=reconnect_paths)

            x = x.to(device)
            y = y.to(device)
            ei = ei.to(device)
            ew = ew.to(device)

            if station_context is not None:
                if virtual_node_training:
                    y_hat = model(x_real, x, station_context_real, station_context_test, ei_real, ei, ew_real, ew, mask_real_diagonal_batched, mask_virtual_diagonal_batched, h_real_)
                    # restore order
                    y = restore_original_order(
                        y,
                        np.array(mask_real_diagonal_batched),
                        mask_virtual_diagonal_batched
                    )
                    y_hat = restore_original_order(
                        y_hat,
                        np.array(mask_real_diagonal_batched),
                        mask_virtual_diagonal_batched
                    )
                else:
                    y_hat = model(x, station_context_test, ei, ew)
            else:
                y_hat = model(x, ei, ew)
            y_hat_.append(y_hat.reshape(y.shape))
            y_.append(y)

    return y_, y_hat_
