# -*- coding: utf-8 -*-
import torch.distributed as dist
from transformers import RobertaConfig

import pcode.models as models
from pcode.models.roberta_client.roberta_client import RobertaClient
from pcode.models.roberta_server.roberta_server import RobertaServer
from pcode.models.roberta_client_head.roberta_client_head import RobertaClientHead
from pcode.models.roberta_whole.roberta_whole import RobertaForMaskedLM

def define_model(
        conf,
        show_stat=True,
        to_consistent_model=True,
        use_complex_arch=True,  # 即使用复杂的结构
        client_id=None,
        arch=None,
):
    _id = client_id if client_id is not None else 0

    client_config = RobertaConfig(
        vocab_size=30522,  # we align this to the tokenizer vocab_size
        max_position_embeddings=514,
        hidden_size=conf.hidden_size,  # 隐藏单元
        num_attention_heads=conf.attention_heads,
        num_hidden_layers=conf.num_client_encoder  # 表示多少个encoder
    )

    client_head_config = RobertaConfig(
        vocab_size=30522,  # we align this to the tokenizer vocab_size
        max_position_embeddings=514,
        hidden_size=conf.hidden_size,  # 隐藏单元
        num_attention_heads=conf.attention_heads,
        num_hidden_layers=0  # 表示多少个encoder
    )

    server_config = RobertaConfig(
        vocab_size=30522,  # we align this to the tokenizer vocab_size
        max_position_embeddings=514,
        hidden_size=conf.hidden_size,  # 隐藏单元
        num_attention_heads=conf.attention_heads,
        num_hidden_layers=conf.num_server_encoder  # 表示多少个encoder
    )

    whole_model_config = RobertaConfig(
        vocab_size=30522,  # we align this to the tokenizer vocab_size
        max_position_embeddings=514,
        hidden_size=conf.hidden_size,  # 隐藏单元
        num_attention_heads=conf.attention_heads,
        num_hidden_layers=conf.num_encoder  # 表示多少个encoder
    )
    conf.logger.log(f"hidden_size: {conf.hidden_size}, attention_heads: {conf.attention_heads}, tot_encoder: {conf.num_encoder}")

    if _id == 0: # server
        # part1: client端模型的参数设置
        client_model = RobertaClient(client_config)
        # part3: client端head block的参数设置
        client_head_model = RobertaClientHead(client_head_config)
        #partt2: server端transformer block的参数设置
        server_model = RobertaServer(server_config)
        whole_model = RobertaForMaskedLM(whole_model_config)

        '''print("--------------client------------")
        for name in client_model.state_dict():
            print(name)
        print()
        print("--------------server------------")
        for name in server_model.state_dict():
            print(name)
        print()
        print("--------------client-head------------")
        for name in client_head_model.state_dict():
            print(name)
        print()
        print("--------------whole-head------------")
        for name in whole_model.state_dict():
            print(name)
        print()'''

        return client_model, client_head_model, server_model, whole_model
    else:
        client_model = RobertaClient(client_config)
        client_head_model = RobertaClientHead(client_head_config)
        return client_model, client_head_model

"""define loaders for different models."""


def determine_arch(conf, client_id, use_complex_arch):
    # the client_id starts from 1.
    _id = client_id if client_id is not None else 0
    if use_complex_arch:
        if _id == 0:
            arch = conf.arch_info["master"]
        else:
            archs = conf.arch_info["worker"]  # worker使用多个arch时，archs是一个字符串列表
            if len(conf.arch_info["worker"]) == 1:
                arch = archs[0]
            else:
                assert "num_clients_per_model" in conf.arch_info
                #
                assert (
                        conf.arch_info["num_clients_per_model"] * len(archs)
                        == conf.n_clients
                )
                arch = archs[int((_id - 1) / conf.arch_info["num_clients_per_model"])]
    else:
        arch = conf.arch
    return arch


def define_cv_classification_model(conf, client_id, use_complex_arch, arch):
    # determine the arch.
    # 获得此进程使用的模型结构
    arch = determine_arch(conf, client_id, use_complex_arch) if arch is None else arch
    # use the determined arch to init the model.
    if "wideresnet" in arch:
        model = models.__dict__["wideresnet"](conf)
    elif "resnet" in arch and "resnet_evonorm" not in arch:
        model = models.__dict__["resnet"](conf, arch=arch)
    elif "resnet_evonorm" in arch:
        model = models.__dict__["resnet_evonorm"](conf, arch=arch)
    elif "regnet" in arch.lower():
        model = models.__dict__["regnet"](conf, arch=arch)
    elif "densenet" in arch:
        model = models.__dict__["densenet"](conf)
    elif "vgg" in arch:
        model = models.__dict__["vgg"](conf)
    elif "mobilenetv2" in arch:
        model = models.__dict__["mobilenetv2"](conf)
    elif "shufflenetv2" in arch:
        model = models.__dict__["shufflenetv2"](conf, arch=arch)
    elif "efficientnet" in arch:
        model = models.__dict__["efficientnet"](conf)
    elif "federated_averaging_cnn" in arch:
        model = models.__dict__["simple_cnn"](conf)
    elif "moderate_cnn" in arch:
        model = models.__dict__["moderate_cnn"](conf)
    else:
        model = models.__dict__[arch](conf)
    return arch, model

'''elif "robertaclient" in arch:
    # init RobertaConfig
    config.vocab_size = conf.vocab_size
    config.num_hidden_layers = conf.num_client_encoder
    model = roberta_client(RobertaConfig)
elif "robertaserver" in arch:
    config.vocab_size = conf.vocab_size
    config.num_hidden_layers = conf.num_server_encoder
    model = roberta_server(RobertaConfig)'''

"""some utilities functions."""


def get_model_stat(conf, model, arch):
    conf.logger.log(
        "\t=> {} created model '{}. Total params: {}M".format(
            "Master"
            if conf.graph.rank == 0
            else f"Worker-{conf.graph.worker_id} (client-{conf.graph.client_id})",
            arch,
            sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6,
        )
    )


def consistent_model(conf, model):
    """it might because of MPI, the model for each process is not the same.

    This function is proposed to fix this issue,
    i.e., use the  model (rank=0) as the global model.
    """
    conf.logger.log("\tconsistent model for process (rank {})".format(conf.graph.rank))
    cur_rank = conf.graph.rank
    for param in model.parameters():
        param.data = param.data if cur_rank == 0 else param.data - param.data
        dist.all_reduce(param.data, op=dist.ReduceOp.SUM)
