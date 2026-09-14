
from timeit import default_timer as timer

import torch
import numpy as np

import random

from enum import Enum
import os
import numpy as np
import torch

import matplotlib.pyplot as plt

class TrainingMode(Enum):
    UNIVARIATE_1HORIZON = 1
    UNIVARIATE_NHORIZON = 2
    MULTIVARIATE_1HORIZON = 3

def save(name, model):
    os.makedirs('results/experiments/{}'.format(name), exist_ok=True)
    torch.save(model.state_dict(), 'results/experiments/{}/model.pt'.format(name))

def log_loss(name, losses, time):
    losses = np.mean(np.array(([losses[1][i].cpu().numpy() for i in range(len(losses[1]))])), axis=1)
    os.makedirs('results/experiments/{}'.format(name), exist_ok=True)
    with open('results/experiments/{}/losses.txt'.format(name), 'w') as fp:
        fp.write(str(losses.tolist()))
        fp.write('time elapsed={}s'.format(time))
    nfeat = losses.shape[-1]
    plt.clf()
    for i in range(nfeat):
        plt.plot(losses[:,i], label='feat_{}'.format(i))
    plt.legend()
    plt.yscale('log')
    plt.savefig('results/experiments/{}/losses.png'.format(name))
    plt.clf()

def load(name, model):
    model.load_state_dict(torch.load('results/experiments/{}/model.pt'.format(name)))
    model.eval()

def train_raster(
    experiment_name,
    model,
    train_dataset,      # (S_train, grid_h, grid_w, T, F)  numpy or tensor
    test_dataset,       # (S_test,  grid_h, grid_w, T, F)
    train_targets,      # (S_train, grid_h, grid_w, H, F_out)
    test_targets,       # (S_test,  grid_h, grid_w, H, F_out)
    context,            # (grid_h, grid_w, C_ctx)  static raster — same for all samples
    loss_fn,
    optimizer,
    EPOCHS,
    BATCH_SIZE,
    BATCH_SIZE_TEST=None,
    dynamic_lr=True,
):
    """
    Train ConvLSTM_ctxCNN_encoderdecoder on interpolated raster sequences.
 
    train/test_dataset : (S, T_total, C_in, H, W)   — already in channel-first format
    train/test_targets : (S, H, C_out, H, W)
    context            : (C_ctx, H, W)               — tiled to batch inside loop
    """
    if BATCH_SIZE_TEST is None:
        BATCH_SIZE_TEST = BATCH_SIZE
 
    device    = torch.device(0)
    model.to(device)
 
    # pre-process context: tile to (B, C_ctx, H, W) for train and test
    ctx       = torch.tensor(context, dtype=torch.float32)  # (C_ctx, H, W)
    ctx_train = ctx.unsqueeze(0).expand(BATCH_SIZE,      -1, -1, -1).to(device)
    ctx_test  = ctx.unsqueeze(0).expand(BATCH_SIZE_TEST, -1, -1, -1).to(device)
 
    S_train = train_dataset.shape[0]
    S_test  = test_dataset.shape[0]
 
    losses   = [[], []]
    best     = None
    early_stop = 0
 
    start = timer()
    try:
        for epoch in range(EPOCHS):
 
            # ── train ─────────────────────────────────────────────────────────
            model.train()
            idx = np.random.permutation(S_train)
            cost = 0
 
            for i in range(0, S_train - BATCH_SIZE, BATCH_SIZE):
                bi = idx[i:i + BATCH_SIZE]
                x  = torch.tensor(train_dataset[bi], dtype=torch.float32).to(device)  # (B, T, C, H, W)
                y  = torch.tensor(train_targets[bi],  dtype=torch.float32).to(device)  # (B, H, C_out, H, W)
 
                y_hat = model(x, ctx_train)   # (B, H, C_out, H, W)
                cost  = loss_fn(y_hat, y)
                cost.backward()
                optimizer.step()
                optimizer.zero_grad()
 
                print(f'epoch {epoch+1}/{EPOCHS}  batch {i//BATCH_SIZE+1}/{S_train//BATCH_SIZE}  loss={cost.item():.5f}')
 
            # ── eval every 10 epochs ──────────────────────────────────────────
            if epoch % 10 != 0:
                continue
 
            model.eval()
            val_cost = 0.0
            with torch.no_grad():
                for i in range(0, S_test - BATCH_SIZE_TEST, BATCH_SIZE_TEST):
                    x = torch.tensor(test_dataset[i:i + BATCH_SIZE_TEST], dtype=torch.float32).to(device)
                    y = torch.tensor(test_targets[i:i + BATCH_SIZE_TEST],  dtype=torch.float32).to(device)
                    y_hat   = model(x, ctx_test)
                    val_cost += loss_fn(y_hat, y).item()
 
            val_cost /= max(S_test // BATCH_SIZE_TEST, 1)
            losses[1].append(val_cost)
            print(f'epoch {epoch+1}/{EPOCHS}  val_loss={val_cost:.5f}  best={best}  early_stop={early_stop}')
 
            if best is None or val_cost < best:
                save(experiment_name, model)
                best = val_cost
                early_stop = 0
            else:
                early_stop += 1
                if early_stop > EPOCHS // 4:
                    load(experiment_name, model)
                    if not dynamic_lr:
                        break
                    for g in optimizer.param_groups:
                        g['lr'] /= 1.5
                    early_stop = 0
                    if optimizer.param_groups[0]['lr'] < 0.0005:
                        break
 
    except KeyboardInterrupt:
        if best is None:
            save(experiment_name, model)
        print('training interrupted.')
 
    end = timer()
    print(f'time elapsed: {end - start:.1f}s')
    load(experiment_name, model)
    model.to(torch.device('cpu'))
    return losses

def train(experiment_name, model, train_dataset, test_dataset, loss, optimizer, EPOCHS, BATCH_SIZE, REPORT_TRAIN_LOSS_EPOCHS, mode: TrainingMode, mean=0, std=1, station_context=None, dynamic_lr=True, BATCH_SIZE_TEST=None, virtual_node_training=False, virtual_node_ratio=0.25, reconnect_paths=False):
    losses = [[], []]
    best = None
    early_stop = 0
    if BATCH_SIZE_TEST is None:
        BATCH_SIZE_TEST = BATCH_SIZE

    device = torch.device(0)

    if not virtual_node_training:
        virtual_node_ratio = 1

    ctx_single = station_context

    start=timer()
    try:
        for epoch in range(EPOCHS):
            nnodes = train_dataset[0].x.shape[0]
            nedges = train_dataset.edge_index.shape[1]
            edge_index_single = train_dataset[0].edge_index
            edge_attr_single = train_dataset[0].edge_attr

            if ctx_single is not None:
                if ctx_single.ndim == 2:
                    ctx_s = ctx_single
                    station_context = torch.tensor(np.concatenate([ctx_s for i in range(BATCH_SIZE)], axis=0))
                    station_context_test = torch.tensor(np.concatenate([ctx_s for i in range(BATCH_SIZE_TEST)], axis=0))
                else:
                    x_ = int(np.sqrt(ctx_single.shape[1]))
                    ctx_s = ctx_single.reshape(nnodes, x_, x_, ctx_single.shape[-1])
                    station_context = torch.tensor(np.concatenate([ctx_s for i in range(BATCH_SIZE)], axis=0).swapaxes(3,1))
                    station_context_test = torch.tensor(np.concatenate([ctx_s for i in range(BATCH_SIZE_TEST)], axis=0).swapaxes(3,1))
                station_context = station_context.to(device)
                station_context_test = station_context_test.to(device)
            
            sample_len = train_dataset[0].x.shape[1]
            random_offset = np.random.randint(0, sample_len)
            if optimizer is not None:
                shuffled_array = np.arange(random_offset, train_dataset.snapshot_count, sample_len)
                np.random.shuffle(shuffled_array)
                model.train()
                model.to(device)
                cost = 0
                for i in range(0, len(shuffled_array)-BATCH_SIZE-1, BATCH_SIZE):
                    x = train_dataset[shuffled_array[i:i+BATCH_SIZE]].x
                    x = x.reshape(-1, x.shape[-2], x.shape[-1])
                    y = train_dataset[shuffled_array[i:i+BATCH_SIZE]].y
                    y = y.reshape(-1, y.shape[-2], y.shape[-1])
                    ei = np.tile(edge_index_single, BATCH_SIZE)  # (2, B*E)
                    offset = np.arange(BATCH_SIZE) * nnodes
                    offset = np.repeat(offset, nedges)
                    ei = ei + offset[None, :]
                    ei = torch.tensor(ei, dtype=torch.int64)
                    ew = edge_attr_single.repeat(BATCH_SIZE)
                    
                    if virtual_node_training:
                        ids_virtual = list(np.sort(np.array(random.sample(range(0, nnodes), int(nnodes * virtual_node_ratio)))))
                        from data_handling.graph_virtualization import virtualize_graph
                        h_real_, x_real, y, y_real, ei_real, ew_real, station_context_real, mask_real_diagonal_batched, mask_virtual_diagonal_batched = virtualize_graph(x, y, ei, ew, station_context, ids_virtual, nnodes, BATCH_SIZE, device, model.L, model.filters, reconnect=reconnect_paths)

                    x = x.to(device)
                    y = y.to(device)
                    ei = ei.to(device)
                    ew = ew.to(device)

                    #print(torch.cuda.mem_get_info())
                    if station_context is not None:
                        if virtual_node_training:
                            y_hat = model(x_real, x, station_context_real, station_context, ei_real, ei, ew_real, ew, mask_real_diagonal_batched, mask_virtual_diagonal_batched, h_real_)
                        else:
                            y_hat = model(x, station_context, ei, ew)
                    else:
                        y_hat = model(x, ei, ew)
                    cost += loss(y_hat.reshape(y.shape), y)
                    
                    print('epoch: {}/{}, batch: {}/{}'.format(epoch+1, EPOCHS, i//BATCH_SIZE+1, int(len(shuffled_array)//BATCH_SIZE)))
                    torch.mean(cost).backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                    optimizer.zero_grad()
                    cost = 0
            model.eval()
            model.to(device)
            cost = 0

            if epoch % 10 != 0:
                continue
            with torch.no_grad():
                # eval test loss
                sample_len = test_dataset[0].x.shape[1]
                ids_ = np.arange(nnodes)
                random_offset=0
                shuffled_array = np.arange(random_offset, test_dataset.snapshot_count, sample_len)#//sample_sparseness)
                for i in range(0, len(shuffled_array)-BATCH_SIZE_TEST-1, BATCH_SIZE_TEST):
                    np.random.shuffle(ids_)
                    ids_virtual = list(ids_[:int(nnodes * virtual_node_ratio)]) # each batch different mask
                    x = test_dataset[shuffled_array[i:i+BATCH_SIZE_TEST]].x
                    x = x.reshape(-1, x.shape[-2], x.shape[-1])
                    y = test_dataset[shuffled_array[i:i+BATCH_SIZE_TEST]].y
                    y = y.reshape(-1, y.shape[-2], y.shape[-1])
                    ei = np.tile(edge_index_single, BATCH_SIZE_TEST)  # (2, B*E)
                    offset = np.arange(BATCH_SIZE_TEST) * nnodes
                    offset = np.repeat(offset, nedges)
                    ei = ei + offset[None, :]
                    ei = torch.tensor(ei, dtype=torch.int64)
                    ew = edge_attr_single.repeat(BATCH_SIZE_TEST)

                    if virtual_node_training:
                        from data_handling.graph_virtualization import virtualize_graph
                        h_real_, x_real, y, y_real, ei_real, ew_real, station_context_real, mask_real_diagonal_batched, mask_virtual_diagonal_batched = virtualize_graph(x, y, ei, ew, station_context_test, ids_virtual, nnodes, BATCH_SIZE_TEST, device, model.L, model.filters, reconnect=reconnect_paths)

                    x = x.to(device)
                    y = y.to(device)
                    ei = ei.to(device)
                    ew = ew.to(device)

                    if station_context is not None:
                        if virtual_node_training:
                            y_hat = model(x_real, x, station_context_real, station_context_test, ei_real, ei, ew_real, ew, mask_real_diagonal_batched, mask_virtual_diagonal_batched, h_real_)
                        else:
                            y_hat = model(x, station_context_test, ei, ew)
                    else:
                        y_hat = model(x, ei, ew)
                    cost += loss(y_hat.reshape(y.shape), y)
                    print('epoch: {}/{}, val batch: {}/{}'.format(epoch+1, EPOCHS, i//BATCH_SIZE_TEST+1, int(len(shuffled_array)//BATCH_SIZE_TEST)))
                # log test loss
                fktr = int(1/virtual_node_ratio)
                cost /= fktr
                fun = torch.sqrt if isinstance(loss, torch.nn.MSELoss) else (lambda x: x)
                print("loss_test: {}".format(str(torch.mean(cost, axis=0))))
                if optimizer:
                    print('lr={}, batch size={}, last_best={}, best={}, curr={}'.format(optimizer.param_groups[0]['lr'], BATCH_SIZE, early_stop, best, fun(torch.mean(cost, axis=(0, 1)))))
                losses[1].append(torch.mean(cost, axis=0))
                if best is None:
                    save(experiment_name, model)
                    best = fun(torch.mean(torch.abs(cost), axis=(0, 1)))
                else:
                    ratioimprov = torch.mean(best / fun(torch.mean(torch.abs(cost), axis=(0, 1))))
                    if ratioimprov > 1.0:
                        save(experiment_name, model)
                        best = fun(torch.mean(torch.abs(cost), axis=(0, 1)))
                        early_stop = 0
                    else:
                        early_stop += 1
                    if early_stop > EPOCHS//4:
                        load(experiment_name, model)
                        if not dynamic_lr:
                            break
                        if dynamic_lr:
                            for g in optimizer.param_groups:
                                g['lr'] /= 1.5
                        early_stop=0
                        if optimizer.param_groups[0]['lr'] < 0.0005 or BATCH_SIZE > 8192:
                            break
    except KeyboardInterrupt:
        if best is None:
            save(experiment_name, model)
        print('finish training.')
    end=timer()
    print("time_elapsed: {}".format(end-start))

    load(experiment_name, model)
    model.to(torch.device('cpu'))
    try:
        log_loss(experiment_name, losses, end-start)
    except:
        pass
    return