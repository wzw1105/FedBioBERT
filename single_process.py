import os
import torch.distributed as dist
import torch.multiprocessing as mp

# -*- coding: utf-8 -*-
import numpy as np


import time
import torch

from parameters import get_args
from pcode.new_master import Master
from pcode.new_worker import Worker
import pcode.utils.topology as topology
import pcode.utils.checkpoint as checkpoint
import pcode.utils.logging as logging
import pcode.utils.param_parser as param_parser


def init_config(conf):
    # define the graph for the computation.
    conf.graph = topology.define_graph_topology(
        world=conf.world,
        world_conf=conf.world_conf,
        n_participated=conf.n_participated,
        on_cuda=conf.on_cuda,
    )
    conf.graph.rank = conf.comm_rank

    conf.num_encoder = conf.num_client_encoder + conf.num_server_encoder
    if conf.num_encoder != 4 and conf.num_encoder != 12:
        raise Exception("num client encoders plus num server encoders is not 4(bert_small) or 12(bert_base)")

    # small和base的hidden_size有区别
    conf.hidden_size = 512 if conf.num_encoder == 4 else 768
    conf.attention_heads = 8 if conf.num_encoder == 4 else 12

    # init related to randomness on cpu.
    if not conf.same_seed_process:
        conf.manual_seed = 1000 * conf.manual_seed + conf.graph.rank
    np.random.seed(conf.manual_seed)
    conf.random_state = np.random.RandomState(conf.manual_seed)
    torch.manual_seed(conf.manual_seed)

    # configure cuda related.
    if conf.graph.on_cuda:
        assert torch.cuda.is_available()
        torch.cuda.manual_seed(conf.manual_seed)
        torch.cuda.set_device(conf.graph.primary_device)
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = True if conf.train_fast else False

    # init the model arch info.
    conf.arch_info = (
        param_parser.dict_parser(conf.complex_arch)
        if conf.complex_arch is not None
        else {"master": conf.arch, "worker": conf.arch}
    )
    conf.arch_info["worker"] = conf.arch_info["worker"].split(":")

    # parse the fl_aggregate scheme.
    conf._fl_aggregate = conf.fl_aggregate
    conf.fl_aggregate = (
        param_parser.dict_parser(conf.fl_aggregate)
        if conf.fl_aggregate is not None
        else conf.fl_aggregate
    )
    [setattr(conf, f"fl_aggregate_{k}", v) for k, v in conf.fl_aggregate.items()]

    # define checkpoint for logging (for federated learning server).
    checkpoint.init_checkpoint(conf, rank=str(conf.graph.rank))

    # configure logger.
    conf.logger = logging.Logger(conf.checkpoint_dir)

    # display the arguments' info.
    if conf.graph.rank == 0:
        logging.display_args(conf)

    # sync the processes.
    dist.barrier()

def main(conf):
    # init the distributed world.
    try:
        dist.init_process_group(backend="gloo", init_method=conf.init_method, rank=conf.comm_rank, world_size=int(conf.n_clients * conf.participation_ratio + 0.5) + 1)
    except AttributeError as e:
        print(f"failed to init the distributed world: {e}.")
        conf.distributed = False

    # init the config.
    init_config(conf)
    # start federated learning.
    process = Master(conf) if conf.graph.rank == 0 else Worker(conf)
    process.run()


if __name__ == "__main__":
    conf = get_args()
    conf.timestamp = str(int(time.time()))
    main(conf)
