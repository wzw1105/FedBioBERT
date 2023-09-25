# -*- coding: utf-8 -*-
import time

import torch
import os
import tqdm
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.optim import AdamW

import pcode.local_training.compressor as compressor
import pcode.local_training.random_reinit as random_reinit
import pcode.datasets.mixup_data as mixup
import pcode.create_model as create_model
import pcode.create_dataset as create_dataset
import pcode.create_optimizer as create_optimizer
import pcode.create_scheduler as create_scheduler
import pcode.create_metrics as create_metrics
from pcode.utils.tensor_buffer import TensorBuffer
from pcode.utils.logging import display_training_stat
from pcode.utils.timer import Timer
from pcode.utils.stat_tracker import RuntimeTracker
import copy


class Worker(object):
    def __init__(self, conf):
        self.conf = conf

        # some initializations.
        self.rank = conf.graph.rank
        conf.graph.worker_id = conf.graph.rank
        device_id = conf.gpu #if conf.single_process is True else conf.device_id[conf.graph.worker_id] # self.conf.graph.worker_id % self.conf.num_device
        self.device = torch.device(f"cuda:{device_id}") #  if self.conf.graph.on_cuda else "cpu"
        conf.logger.log(f"Worker-{self.conf.graph.worker_id} choose device {device_id}")

        # define the timer for different operations.
        # if we choose the `train_fast` mode, then we will not track the time.
        self.timer = Timer(
            verbosity_level=1 if conf.track_time and not conf.train_fast else 0,
            log_fn=conf.logger.log_metric,
        )

        #用于接收服务器发送的transformer中间输出的TensorBuffer
        self.trans_output_buffer = TensorBuffer([torch.zeros(self.conf.batch_size, self.conf.max_len, self.conf.hidden_size)])

        # create dataset (as well as the potential data_partitioner) for training.
        #dist.barrier()
        #self.dataset = create_dataset.define_dataset(conf, data=conf.data)  # 定义数据集

        #记录当前在第几个communication round，用于learning_rate的更新
        self.cur_step = 0 if conf.from_checkpoint is False else conf.from_checkpoint_round * conf.num_batch * conf.n_local_rounds
        self.learning_rate = 0
        self.warmup_steps = int(conf.warmup_ratio * conf.n_comm_rounds * conf.num_batch * conf.n_local_rounds)
        self.tot_steps = conf.n_comm_rounds * conf.num_batch * conf.n_local_rounds
        self.decay_steps = self.tot_steps - self.warmup_steps
        self.end_learning_rate = self.conf.end_lr
        self.power = 2.0 #learning_rate decay的多项式次数
        self.update_learning_rate()
        conf.logger.log(f"Worker-{conf.graph.worker_id} has {self.warmup_steps} warmup steps, init_learning_rate {self.learning_rate}.")

        if not os.path.exists(f'{self.conf.checkpoint_save_dir}/loss'):
            os.system(f'mkdir {self.conf.checkpoint_save_dir}/loss')
        os.system(f'rm -rf {self.conf.checkpoint_save_dir}/loss/*')

        conf.logger.log(f"Worker-{conf.graph.worker_id} finished initialization !!!!")
        dist.barrier()

    def run(self):
        self.conf.tot_time = 0
        start_round = 1 if self.conf.from_checkpoint is False else self.conf.from_checkpoint_round + 1
        self.conf.logger.log(f"woker - {self.conf.graph.rank} starts from global_round {start_round}")

        for comm_round in range(1, 1 + self.conf.n_comm_rounds):
            self.conf.graph.comm_round = comm_round

            # 获得master节点给所有n_clients划分的数据下标，也就是为了获得data_partitioner
            dist.barrier()
            self._listen_to_master()

            self.dataset = create_dataset.define_dataset(self.conf, data=self.conf.data)  # 定义数据集
            dist.barrier()

            self.train_loader, self.data_partitioner = create_dataset.define_data_loader(
                # 这里define_data_loader的目的是client接收master划分的indices然后，找到自己的数据
                self.conf,
                dataset=self.dataset["train"],
                localdata_id=self.conf.graph.client_id - 1, # random id here.
                is_train=True,
                data_partitioner=None,
            )

            self.conf.logger.log(f"comm_round={comm_round} (worker_id={self.conf.graph.worker_id}) received data_partitioner from master.")
            # 从server获得client-avg模型
            self._recv_model_from_master()
            self._recv_head_model_from_master()
            self.conf.logger.log(f"comm_round={comm_round} (worker_id={self.conf.graph.worker_id}) received model and head_model from master.")

            for local_round in range(1, 1 + self.conf.n_local_rounds):
                dist.barrier()
                # 获得当前local_round的client_id, comm_round, next_batch_ids消息
                self.conf.graph.local_round = local_round
                self._train()
                dist.barrier()
                # self._send_optim_to_master()
            # 将模型发送给server
            self._send_model_to_master()
            self._send_head_model_to_master()
            dist.barrier()
    # 获得此local_round的所有batch (依据next_index获取) 由于太耗时不再使用
    def get_current_batches(self, next_batch_id):
        data_iter = iter(self.train_loader)
        data = data_iter.__next__()
        cur_ind = 0
        result = []
        while cur_ind < next_batch_id:
            data = data_iter.__next__()
            cur_ind += 1
        num_batches = self.conf.batch_size // self.conf.mini_batch_size
        for i in range(num_batches):
            result.append(data)
            if i < num_batches - 1:
                data = data_iter.__next__()
        self.conf.logger.log(
            f"comm_round={self.conf.graph.comm_round} local_round={self.conf.graph.local_round} "
            f"(client_id={self.conf.graph.client_id}) get the batch iterator for current local round.")
        return result

    def _listen_to_master(self):
        self.conf.logger.log(f"comm_round={self.conf.graph.comm_round} Worker-{self.conf.graph.worker_id} received activation from master.")
        # listen to master, related to the function `_activate_selected_clients` in `master.py`.
        msg = torch.zeros((3, self.conf.n_participated))
        dist.broadcast(tensor=msg, src=0)

        # 获得当前
        self.conf.graph.client_id, self.conf.graph.comm_round, self.data_loader_length= (
            msg[:, self.conf.graph.rank - 1].to(int).cpu().numpy().tolist()
        )
        # once we receive the signal, we init for the local training.
        self.model, self.head_model = create_model.define_model(
            self.conf, to_consistent_model=False, client_id=self.conf.graph.client_id
        )
        self.model_state_dict = self.model.state_dict()
        self.head_model_state_dict = self.head_model.state_dict()
        self.model_tb = TensorBuffer(list(self.model_state_dict.values()))
        self.head_model_tb = TensorBuffer(list(self.head_model_state_dict.values()))
        dist.barrier()

    def _recv_model_from_master(self):
        # related to the function `_send_model_to_selected_clients` in `master.py`
        dist.barrier()
        old_buffer = copy.deepcopy(self.model_tb.buffer)
        dist.recv(self.model_tb.buffer, src=0)
        new_buffer = copy.deepcopy(self.model_tb.buffer)
        self.model_tb.unpack(self.model_state_dict.values())

        self.model.load_state_dict(self.model_state_dict)
        self.optim = AdamW(self.model.parameters(), lr=self.learning_rate, betas=(0.9, 0.999), weight_decay=0.01, eps=1e-6)
        self.conf.logger.log(
            f"comm_round={self.conf.graph.comm_round} "
            f"(client-{self.conf.graph.client_id}) received the model (client_model) from Master. The model status {'is updated' if old_buffer.norm() != new_buffer.norm() else 'is not updated'}."
        )
        dist.barrier()

    def _recv_head_model_from_master(self):
        # related to the function `_send_model_to_selected_clients` in `master.py`
        dist.barrier()
        old_buffer = copy.deepcopy(self.head_model_tb.buffer)
        dist.recv(self.head_model_tb.buffer, src=0)
        new_buffer = copy.deepcopy(self.head_model_tb.buffer)
        self.head_model_tb.unpack(self.head_model_state_dict.values())

        self.head_model.load_state_dict(self.head_model_state_dict)
        self.head_optim = AdamW(self.head_model.parameters(), lr=self.learning_rate, betas=(0.9, 0.999), weight_decay=0.01, eps=1e-6)
        self.conf.logger.log(
            f"comm_round={self.conf.graph.comm_round} "
            f"(client-{self.conf.graph.client_id}) received the model (client_head_model) from Master. The model status {'is updated' if old_buffer.norm() != new_buffer.norm() else 'is not updated'}."
        )
        dist.barrier()

    def _recv_optim_from_master(self):
        dist.barrier()
        dist.recv(self.optim_tb, src=0)
        self.optim_tb.unpack(self.optim_state_dict.values())
        self.optim.load_state_dict(self.optim_state_dict)
        self.conf.logger.log(
            f"comm_round={self.conf.graph.comm_round} local_round={self.conf.graph.local_round} "
            f"(client-{self.conf.graph.client_id}) received the optim from Master."
        )
        dist.barrier()

    def _recv_output_from_master(self):
        # dist.barrier()
        server_output =  [torch.zeros(self.conf.batch_size, self.conf.max_len, self.conf.hidden_size, requires_grad=True)]
        dist.recv(self.trans_output_buffer.buffer, src=0)
        self.trans_output_buffer.unpack(server_output)
        # print(f"client - {self.rank} receive output: ",server_output[0])

        # dist.barrier()
        self.conf.logger.log(
            f"\tcomm_round={self.conf.graph.comm_round} local_round={self.conf.graph.local_round} "
            f"(client-{self.conf.graph.client_id}) received the server output from Master."
        )
        return server_output[0]

    def update_learning_rate(self):
        if self.cur_step <= self.warmup_steps:
            self.learning_rate = 1.0 * self.cur_step * self.conf.lr / self.warmup_steps
        else:
            # linear: self.learning_rate = self.conf.lr - 1.0 * (self.cur_step - self.warmup_steps) * self.conf.lr / self.decay_steps
            # polynomial decay
            if (self.cur_step - self.warmup_steps) <= self.decay_steps:
                self.learning_rate = (self.conf.lr - self.end_learning_rate) * pow((1 - ((self.cur_step - self.warmup_steps) / self.decay_steps)), self.power) + self.end_learning_rate
            else:
                self.learning_rate = self.end_learning_rate

    def learning_rate_step(self):
        self.cur_step += 1
        self.update_learning_rate()
        for param in self.optim.param_groups:
            param['lr']=self.learning_rate
        for param in self.head_optim.param_groups:
            param['lr']=self.learning_rate

    def _train(self):
        self.model.train()
        self.head_model.train()

        # init the model and dataloader.
        if self.conf.graph.on_cuda:
            self.model = self.model.to(self.device)
            self.head_model = self.head_model.to(self.device)

        self.conf.logger.log(
            f"comm_round={self.conf.graph.comm_round} local_round={self.conf.graph.local_round} "
            f"(client_id={self.conf.graph.client_id}) enters the local training phase."
        )

        #使用数据进行训练
        index = 0
        all_loss = []
        time_point = []
        for data in self.train_loader:
            batch_start = time.time()
            # 更新模型
            data = {key: value.to(self.device) for key, value in data.items()} if self.conf.graph.on_cuda else data
            self.conf.logger.log(
                f"\tcomm_round={self.conf.graph.comm_round} local_round={self.conf.graph.local_round} "
                f"(client-{self.conf.graph.client_id}) start forward propagation."
            )
            output = self.model(data['bert_input'], attention_mask=data['segment_label'], labels=data['bert_label'])

            loss = output[0]

            dist.barrier() # 放在这儿客户端就不用等服务端梯度更新完
            # 将输出发送给master
            self.client_output = output[1].cpu() #
            self.attention_mask = output[3].cpu() #
            self._send_output_to_master()

            self.optim.zero_grad()
            loss.backward() # 使用local loss更新模型
            self.optim.step()
            self.optim.zero_grad()

            # 等待server推理完并接收server的中间输出（最后一个encoder的输出）
            server_out = self._recv_output_from_master().to(self.device)
            server_out.retain_grad()

            # 完成Head层的推理、反向传播、head模型的更新
            output = self.head_model(server_output=server_out, labels=data['bert_label'])
            self.head_optim.zero_grad()
            loss = output[0]
            loss.backward()
            self.head_optim.step()

            # 发送server_out的梯度给server
            self._send_grad_to_master(server_out.grad.cpu()) #

            self.conf.logger.log(
                f"\tcomm_round={self.conf.graph.comm_round} local_round={self.conf.graph.local_round} "
                f"(client-{self.conf.graph.client_id}) loss: {loss.item()}"
            )
            all_loss.append(loss.item())
            self.conf.tot_time += time.time() - batch_start
            time_point.append(self.conf.tot_time)

            # 更新learning_rate
            self.learning_rate_step()
            self.conf.logger.log(
                f"\tcomm_round={self.conf.graph.comm_round} local_round={self.conf.graph.local_round} "
                f"(client-{self.conf.graph.client_id}) optim_lr: {self.optim.param_groups[0]['lr']}, head_optim_lr: {self.head_optim.param_groups[0]['lr']}"
            )

            index += 1
            if index == self.data_loader_length:
                break

        self.conf.logger.log(
            f"comm_round={self.conf.graph.comm_round} local_round={self.conf.graph.local_round} "
            f"(client_id={self.conf.graph.client_id}) finished one local_round of training."
        )
        self.model = self.model.cpu()
        self.head_model = self.head_model.cpu()

        with open(f"{self.conf.checkpoint_save_dir}/loss/client-{self.conf.graph.worker_id}-loss.txt", 'a') as w:
            for time_, data in zip(time_point, all_loss):
                w.write("%.4f %.4f\n" % (time_, data))
            w.close()
        self.conf.logger.log(
            f"comm_round={self.conf.graph.comm_round} local_round={self.conf.graph.local_round} "
            f"(client_id={self.conf.graph.client_id}) saved loss."
        )
        # torch.cuda.empty_cache()

    def _inference(self, data_batch):
        """Inference on the given model and get loss and accuracy."""
        # do the forward pass and get the output.
        output = self.model(data_batch["input"])

        # evaluate the output and get the loss, performance.
        if self.conf.use_mixup:
            loss = mixup.mixup_criterion(
                self.criterion,
                output,
                data_batch["target_a"],
                data_batch["target_b"],
                data_batch["mixup_lambda"],
            )

            performance_a = self.metrics.evaluate(loss, output, data_batch["target_a"])
            performance_b = self.metrics.evaluate(loss, output, data_batch["target_b"])
            performance = [
                data_batch["mixup_lambda"] * _a + (1 - data_batch["mixup_lambda"]) * _b
                for _a, _b in zip(performance_a, performance_b)
            ]
        else:
            loss = self.criterion(output, data_batch["target"])
            performance = self.metrics.evaluate(loss, output, data_batch["target"])

        # update tracker.
        if self.tracker is not None:
            self.tracker.update_metrics(
                [loss.item()] + performance, n_samples=data_batch["input"].size(0)
            )
        return loss, output

    def _add_grad_from_prox_regularized_loss(self):
        assert self.conf.local_prox_term >= 0
        if self.conf.local_prox_term != 0:
            assert self.conf.weight_decay == 0
            assert self.conf.optimizer == "sgd"
            assert self.conf.momentum_factor == 0

            for _param, _init_param in zip(
                    self.model.parameters(), self.init_model.parameters()
            ):
                if _param.grad is not None:
                    _param.grad.data.add_(
                        (_param.data - _init_param.data) * self.conf.local_prox_term
                    )

    def _local_training_with_self_distillation(self, loss, output, data_batch):
        if self.conf.self_distillation > 0:
            loss = loss * (
                    1 - self.conf.self_distillation
            ) + self.conf.self_distillation * self._divergence(
                student_logits=output / self.conf.self_distillation_temperature,
                teacher_logits=self.init_model(data_batch["input"])
                               / self.conf.self_distillation_temperature,
            )
        return loss

    def _divergence(self, student_logits, teacher_logits):
        divergence = F.kl_div(
            F.log_softmax(student_logits, dim=1),
            F.softmax(teacher_logits, dim=1),
            reduction="batchmean",
        )  # forward KL
        return divergence

    def _turn_off_grad(self, model):
        for param in model.parameters():
            param.requires_grad = False
        return model

    def _send_model_to_master(self):
        dist.barrier()
        self.conf.logger.log(
            f"comm_round={self.conf.graph.comm_round} "
            f"(client_id={self.conf.graph.client_id}) is sending the model (roberta_client_model) back to Master."
        )
        flatten_model = TensorBuffer(list(self.model.state_dict().values()))
        dist.send(tensor=flatten_model.buffer, dst=0)
        dist.barrier()

    def _send_head_model_to_master(self):
        dist.barrier()
        self.conf.logger.log(
            f"comm_round={self.conf.graph.comm_round} "
            f"(client_id={self.conf.graph.client_id}) is sending the model (roberta_client_model) back to Master."
        )
        flatten_head_model = TensorBuffer(list(self.head_model.state_dict().values()))
        dist.send(tensor=flatten_head_model.buffer, dst=0)
        dist.barrier()

    def _send_optim_to_master(self):
        dist.barrier()
        self.conf.logger.log(
            f"comm_round={self.conf.graph.comm_round} local_round={self.conf.graph.local_round} "
            f"(client_id={self.conf.graph.client_id}) is sending the client optim back to Master."
        )
        flatten_model = TensorBuffer(list(self.optim.state_dict().values()))
        dist.send(tensor=flatten_model.buffer, dst=0)
        dist.barrier()

    # send intermidiate_client_output attention_mask and labels to the master.
    def _send_output_to_master(self): #ok
        dist.barrier()
        flatten_model = TensorBuffer([self.client_output, self.attention_mask])
        dist.send(tensor=flatten_model.buffer, dst=0)

        # print(f"client - {self.rank} send: ", self.client_output)
        self.conf.logger.log(
            f"\tcomm_round={self.conf.graph.comm_round} local_round={self.conf.graph.local_round} "
            f"(client-{self.conf.graph.client_id}) send output to Master."
        )
        dist.barrier()

    def _send_grad_to_master(self, grad):
        #dist.barrier()
        flatten_model = TensorBuffer([grad])
        dist.send(tensor=flatten_model.buffer, dst=0)

        #print(f"client - {self.rank} send grad： ", grad)
        self.conf.logger.log(
            f"\tcomm_round={self.conf.graph.comm_round} local_round={self.conf.graph.local_round} "
            f"(client-{self.conf.graph.client_id}) send gradient to Master."
        )
        #dist.barrier()