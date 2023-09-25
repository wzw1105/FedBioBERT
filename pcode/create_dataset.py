# -*- coding: utf-8 -*-
import torch

from pcode.datasets.partition_data import DataPartitioner
from pcode.datasets.prepare_data import get_dataset
import pcode.datasets.mixup_data as mixup
from BertSingle.RobertDataset.RobertaDataset import RobertaDataset

"""create dataset and load the data_batch."""


def load_data_batch(conf, _input, _target, is_training=True):
    """Load a mini-batch and record the loading time."""
    if conf.graph.on_cuda:
        _input, _target = _input.cuda(), _target.cuda()

    # argument data.
    if conf.use_mixup and is_training:
        _input, _target_a, _target_b, mixup_lambda = mixup.mixup_data(
            _input,
            _target,
            alpha=conf.mixup_alpha,
            assist_non_iid=conf.mixup_noniid,
            use_cuda=conf.graph.on_cuda,
        )
        _data_batch = {
            "input": _input,
            "target_a": _target_a,
            "target_b": _target_b,
            "mixup_lambda": mixup_lambda,
        }
    else:
        _data_batch = {"input": _input, "target": _target}
    return _data_batch


def define_dataset(conf, data, display_log=True):
    # prepare general train/test.
    conf.partitioned_by_user = True if "femnist" == conf.data else False
    train_dataset = get_dataset(conf, data, conf.data_dir, split="train")

    return {"train": train_dataset, "val": None, "test": None}


def define_val_dataset(conf, train_dataset, test_dataset):
    assert conf.val_data_ratio >= 0

    partition_sizes = [
        (1 - conf.val_data_ratio) * conf.train_data_ratio,
        (1 - conf.val_data_ratio) * (1 - conf.train_data_ratio),
        conf.val_data_ratio,
    ]

    data_partitioner = DataPartitioner(
        conf,
        train_dataset,
        partition_sizes,
        partition_type="origin",
        consistent_indices=False,
    )
    train_dataset = data_partitioner.use(0)

    # split for val data.
    if conf.val_data_ratio > 0:
        assert conf.partitioned_by_user is False

        val_dataset = data_partitioner.use(2)
        return train_dataset, val_dataset, test_dataset
    else:
        return train_dataset, None, test_dataset


# 新增参数：start_indices：dataset中本轮client需要的数据的开始下标，batch_size：本轮需要的数据数量
def define_data_loader(
        conf, dataset, localdata_id=None, is_train=True, shuffle=True, data_partitioner=None
):
    # determine the data to load,
    # either the whole dataset, or a subset specified by partition_type.
    if is_train:
        world_size = conf.n_clients
        world_size = conf.n_clients
        partition_sizes = [1.0 / world_size for _ in range(world_size)]  # 每个client使用数据集的大小所占的比例
        assert localdata_id is not None

        # 只有femnist数据集时才为true
        if conf.partitioned_by_user:  # partitioned by "users".
            # in case our dataset is already partitioned by the client.
            # and here we need to load the dataset based on the client id.
            dataset.set_user(localdata_id)
            data_to_load = dataset
        else:  # (general) partitioned by "labels".
            # in case we have a global dataset and want to manually partition them.
            if data_partitioner is None:
                # update the data_partitioner.
                data_partitioner = DataPartitioner(  # 定义一个划分数据的方式，其中最关键的数据就是partitions，其中存储了每个client使用的全局数据的下标集合
                    conf, dataset, partition_sizes, partition_type=conf.partition_data
                )
            # note that the master node will not consume the training dataset.
            data_to_load = data_partitioner.use(localdata_id)  # 得到的是client=localdata_id的client用户dataset，在master运行的时候这个data_to_load并没有用到
        conf.logger.log(
            f"Data partition for train (client_id={localdata_id + 1}) finished."
        )
    else:  # test或者evaluate都不用划分，直接使用整个数据
        if conf.partitioned_by_user:  # partitioned by "users".
            # in case our dataset is already partitioned by the client.
            # and here we need to load the dataset based on the client id.
            dataset.set_user(localdata_id)
            data_to_load = dataset
        else:
            data_to_load = dataset
        conf.logger.log("Data partition for validation/test.")

    # use Dataloader.
    data_loader = torch.utils.data.DataLoader(  # 定义loader
        data_to_load,
        batch_size=conf.batch_size,
        shuffle=shuffle,
        num_workers=conf.num_workers,
        pin_memory=conf.pin_memory,
        drop_last=False,
        multiprocessing_context='fork'
    )

    # Some simple statistics.
    conf.logger.log(
        "\tData stat for {}: # of samples={} for {}. # of batches={}. The batch size={}".format(
            "train" if is_train else "validation/test",
            len(data_to_load),
            f"client_id={localdata_id + 1}" if localdata_id is not None else "Master",
            len(data_loader),
            conf.batch_size,
        )
    )
    conf.num_batches_per_device_per_epoch = len(data_loader)
    conf.num_whole_batches_per_worker = (
            conf.num_batches_per_device_per_epoch * conf.local_n_epochs
    )
    return data_loader, data_partitioner
