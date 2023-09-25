# -*- coding: utf-8 -*-
import collections
import os
import copy
import time

import numpy as np
import torch
import torch.distributed as dist
import tqdm
from torch.optim import AdamW
from transformers import RobertaConfig

import pcode.master_utils as master_utils
import pcode.create_coordinator as create_coordinator
import pcode.create_aggregator as create_aggregator
import pcode.create_dataset as create_dataset
import pcode.create_metrics as create_metrics
import pcode.create_model as create_model
import pcode.utils.checkpoint as checkpoint
from pcode.models.roberta_client.roberta_client import RobertaClient
from pcode.models.roberta_server.roberta_server import RobertaServer
from pcode.models.roberta_client_head.roberta_client_head import RobertaClientHead
from pcode.models.roberta_whole.roberta_whole import RobertaForMaskedLM
from pcode.utils.tensor_buffer import TensorBuffer
import pcode.utils.cross_entropy as cross_entropy
from pcode.utils.early_stopping import EarlyStoppingTracker

class Master(object):
    def __init__(self, conf):
        self.conf = conf

        # some initializations.
        self.device_id = conf.gpu
        self.device = torch.device(f"cuda:{self.device_id}" if self.conf.graph.on_cuda else "cpu")

        self.parallel_device_ids = None

        self.client_ids = list(range(1, 1 + conf.n_clients)) #client_id是所有参与者的编号
        self.world_ids = list(range(1, 1 + conf.n_participated)) #每次随机选的实际参与者的编号
        self.selected_client_ids = range(1, 1 + conf.n_participated)

        # 获得master节点的模型
        self.client_model, self.client_head_model, self.server_model, self.whole_model = create_model.define_model(conf, to_consistent_model=False)
        self.client_models = [copy.deepcopy(self.client_model) for i in range(self.conf.n_participated)] # 用于接收每个参与者的client模型
        self.client_head_models = [copy.deepcopy(self.client_head_model) for i in range(self.conf.n_participated)]
        self.conf.logger.log("Master create client-model client-head-model server-model successfully.")

        # 从checkpoint加载
        if self.conf.from_checkpoint:
            self.server_model = RobertaServer.from_pretrained(f"{self.conf.checkpoint_save_dir}/server_model")
            for world_id in self.world_ids:
                self.client_models[world_id - 1] = RobertaClient.from_pretrained(f"{self.conf.checkpoint_save_dir}/client_model")
                self.client_head_models[world_id - 1] = RobertaClientHead.from_pretrained(f"{self.conf.checkpoint_save_dir}/client_head_model")
            self.conf.logger.log(f"server load checkpoint from {self.conf.checkpoint_save_dir}/round{self.conf.from_checkpoint_round} successfully.")

        # state_dict
        self.client_state_dict = self.client_model.state_dict()
        self.client_head_state_dict = self.client_head_model.state_dict()
        self.client_model_tb = TensorBuffer(list(self.client_state_dict.values()))
        self.client_head_model_tb = TensorBuffer(list(self.client_head_state_dict.values()))

        # 设置learning_rate
        self.cur_step = 0 if conf.from_checkpoint is False else conf.from_checkpoint_round * conf.num_batch * conf.n_local_rounds # 记录当前在第几个communication round，用于learning_rate的更新
        self.learning_rate = 0
        self.warmup_steps = int(conf.warmup_ratio * conf.n_comm_rounds * conf.num_batch * conf.n_local_rounds)
        self.tot_steps = conf.n_comm_rounds * conf.num_batch * conf.n_local_rounds
        self.decay_steps = self.tot_steps - self.warmup_steps
        self.end_learning_rate = self.conf.end_lr #多项式下降的最终学习率
        self.power = 2.0 #learning_rate decay的多项式次数
        self.update_learning_rate() #根据cur_step更新learning_rate
        conf.logger.log(f"Master has {self.warmup_steps} warmup steps, init_learning_rate {self.learning_rate}.")

        if not self.conf.fedavg_server_model:
            self.server_model = self.server_model.train().to(self.device)
            self.optim = AdamW(self.server_model.parameters(), lr=self.learning_rate, betas=(0.9, 0.999), weight_decay=0.01, eps=1e-6)
        else:
            self.parallel_device_ids = [torch.device(f"cuda:{i % self.conf.num_device}") for i in range(1 + conf.n_participated)]
            self.tmp_server_model = [copy.deepcopy(self.server_model) for i in range(self.conf.n_participated)] # .train().to(self.parallel_device_ids[i])
            self.optim = [AdamW(self.tmp_server_model[i].parameters(), lr=self.learning_rate, betas=(0.9, 0.999), weight_decay=0.01, eps=1e-6) for i in range(self.conf.n_participated)]
        conf.logger.log("Master created model optimizers.")

        # create dataset (as well as the potential data_partitioner) for training.
        #dist.barrier() # 每个进程进入这个函数后都会被阻塞，当所有进程都进入这个函数后，阻塞解除，继续向下执行。
        #self.dataset = create_dataset.define_dataset(conf, data=conf.data)
        #conf.logger.log(f"Master distributed datasets to {conf.n_participated} clients.")

        self.embedding_state_key = ['roberta.embeddings.word_embeddings.weight',
                               'roberta.embeddings.position_embeddings.weight',
                               'roberta.embeddings.token_type_embeddings.weight', 'roberta.embeddings.LayerNorm.weight',
                               'roberta.embeddings.LayerNorm.bias']
        self.lm_head_state_key = ['lm_head.bias', #'roberta.pooler.dense.weight', 'roberta.pooler.dense.bias',
                             'lm_head.dense.weight', 'lm_head.dense.bias', 'lm_head.layer_norm.weight',
                             'lm_head.layer_norm.bias', 'lm_head.decoder.weight', 'lm_head.decoder.bias']

        if not os.path.exists(f'{self.conf.checkpoint_save_dir}/client_model'):
            os.system(f'mkdir {self.conf.checkpoint_save_dir}/client_model')
        if not os.path.exists(f'{self.conf.checkpoint_save_dir}/client_head_model'):
            os.system(f'mkdir {self.conf.checkpoint_save_dir}/client_head_model')
        if not os.path.exists(f'{self.conf.checkpoint_save_dir}/server_model'):
            os.system(f'mkdir {self.conf.checkpoint_save_dir}/server_model')

        self.conf.logger.log("Master finished initialization !!!!!")
        dist.barrier()

    def run(self):
        start_round = 1 if self.conf.from_checkpoint is False else self.conf.from_checkpoint_round + 1
        self.conf.logger.log(f"server starts from global_round {start_round}")

        for comm_round in range(start_round, self.conf.n_comm_rounds + 2):
            self.conf.graph.comm_round = comm_round
            self.save_model(comm_round - 1)
            if comm_round == self.conf.n_comm_rounds + 1:
                break
            dist.barrier()

            # 从n_clients中随机选择n_participated个clients
            self.selected_client_ids = self._random_select_clients()

            # init the activation tensor and broadcast to all clients (either start or stop).
            self._activate_selected_clients(self.selected_client_ids, self.conf.graph.comm_round)

            self.dataset = create_dataset.define_dataset(self.conf, data=self.conf.data)
            dist.barrier()

            _, self.data_partitioner = create_dataset.define_data_loader(
                self.conf,
                dataset=self.dataset["train"],
                localdata_id=0,  # random id here.
                is_train=True,
                data_partitioner=None,
            )
            self.conf.logger.log(
                f"Master splited data, starting one round of federated learning: (comm_round={comm_round})."
            )
            # 将avg-client模型发送给所有client, 每个global_round发送一次avg模型（model和head_model）
            self._send_model_to_selected_clients(self.selected_client_ids)
            self._send_head_model_to_selected_clients(self.selected_client_ids)
            self.conf.logger.log(f"(comm_round={comm_round}) Master send model and head_model to clients successfully.")
            if self.conf.fedavg_server_model:
                self._load_avg_server_model()
            # 开始一个local_epoch
            for local_round in range(1, 1 + self.conf.n_local_rounds): # 选定本轮参与训练的client后，每个client训练的epoch数量
                dist.barrier()
                self.conf.graph.local_round = local_round
                self.update_server_model()
                dist.barrier()

                self.conf.logger.log(f"comm_round={self.conf.graph.comm_round} local_round={self.conf.graph.local_round} Master finished one local_round of federated learning.\n")
            if self.conf.fedavg_server_model:
                self._aggregate_server_model()
            # 从client那里接收模型
            flatten_local_models = self._receive_models_from_selected_clients(self.selected_client_ids)
            flatten_local_head_models = self._receive_head_models_from_selected_clients(self.selected_client_ids)

            # 做client_model的avg
            self._aggregate_model_and_evaluate(flatten_local_models)
            self._aggregate_head_model_and_evaluate(flatten_local_head_models)

            dist.barrier()

    def save_model(self,comm_round):
        self.aggregate_client_server(self.client_models[0].state_dict(), self.client_head_models[0].state_dict(), self.server_model.state_dict())
        #print(f"after ", whole_model.state_dict()['roberta.encoder.layer.0.attention.self.key.weight'])
        #print(f"after ", whole_model.state_dict()['roberta.encoder.layer.1.attention.self.key.weight'])
        if not os.path.exists(f'{self.conf.checkpoint_save_dir}/round{comm_round}'):
            os.system(f'mkdir {self.conf.checkpoint_save_dir}/round{comm_round}')
        if not os.path.exists(f'{self.conf.checkpoint_save_dir}/client_model'):
            os.system(f'mkdir {self.conf.checkpoint_save_dir}/client_model')
        if not os.path.exists(f'{self.conf.checkpoint_save_dir}/client_head_model'):
            os.system(f'mkdir {self.conf.checkpoint_save_dir}/client_head_model')

        self.whole_model.save_pretrained(f'{self.conf.checkpoint_save_dir}/round{comm_round}')
        self.client_models[0].save_pretrained(f'{self.conf.checkpoint_save_dir}/client_model')
        self.client_head_models[0].save_pretrained(f'{self.conf.checkpoint_save_dir}/client_head_model')
        self.server_model.save_pretrained(f'{self.conf.checkpoint_save_dir}/server_model')
        self.conf.logger.log(f"Master saved models in {self.conf.checkpoint_save_dir}/round{comm_round}")

    def _random_select_clients(self):
        selected_client_ids = self.conf.random_state.choice(
            self.client_ids, self.conf.n_participated, replace=False
        ).tolist()
        selected_client_ids.sort()
        self.conf.logger.log(
            f"comm_round={self.conf.graph.comm_round} Master selected {self.conf.n_participated} from {self.conf.n_clients} clients: {selected_client_ids}."
        )
        return selected_client_ids

    def _activate_selected_clients(self, selected_client_ids, comm_round):
        # Activate the selected clients:
        # the first row indicates the client id,
        # the second row indicates the current_comm_round,
        # the third row indicates the next batch indices
        self.data_loader_length = self.conf.num_batch #控制每个client训练的batch数相同
        selected_client_ids = np.array(selected_client_ids)
        activation_msg = torch.zeros((3, len(selected_client_ids)))
        activation_msg[0, :] = torch.Tensor(selected_client_ids)
        activation_msg[1, :] = comm_round
        activation_msg[2, :] = self.data_loader_length # length of data_loader
        dist.broadcast(tensor=activation_msg, src=0)
        self.conf.logger.log(f"comm_round-{self.conf.graph.comm_round} Master activated the selected clients.")
        dist.barrier()

    def _send_model_to_selected_clients(self, selected_client_ids):
        # the master_model can be large; the client_models can be small and different.
        dist.barrier()
        self.conf.logger.log(f"Master send the models to workers.")
        for world_id, selected_client_id in zip(self.world_ids, selected_client_ids):
            client_model_state_dict = self.client_models[world_id - 1].state_dict()
            flatten_model = TensorBuffer(list(client_model_state_dict.values()))

            dist.send(tensor=flatten_model.buffer, dst=world_id)
            self.conf.logger.log(
                f"\tcomm_round={self.conf.graph.comm_round} Master send the current model=client_model to process_id={world_id}."
            )
        dist.barrier()

    def _send_head_model_to_selected_clients(self, selected_client_ids):
        # the master_model can be large; the client_models can be small and different.
        dist.barrier()
        self.conf.logger.log(f"Master send the models to workers.")
        for world_id, selected_client_id in zip(self.world_ids, selected_client_ids):
            client_head_model_state_dict = self.client_head_models[world_id - 1].state_dict()
            flatten_model = TensorBuffer(list(client_head_model_state_dict.values()))

            dist.send(tensor=flatten_model.buffer, dst=world_id)
            self.conf.logger.log(
                f"\tcomm_round={self.conf.graph.comm_round} Master send the current head model=client_model to process_id={world_id}."
            )
        dist.barrier()

    def _send_optim_to_selected_clients(self, selected_client_ids):
        dist.barrier()
        self.conf.logger.log(f"Master send the optims to workers.")
        for worker_rank, selected_client_id in enumerate(selected_client_ids, start=1):
            optim_state_dict = self.client_optims[selected_client_id - 1].state_dict()
            flatten_model = TensorBuffer(list(optim_state_dict.values()))

            dist.send(tensor=flatten_model.buffer, dst=worker_rank)
            self.conf.logger.log(f"\tcomm_round={self.conf.graph.comm_round} local_round={self.conf.graph.local_round} Master send the current client-optim to process_id={worker_rank}.")
        dist.barrier()

    def _send_output_to_selected_clients(self, selected_client_ids, server_outputs):
        for world_id in self.world_ids:
            flatten_model = TensorBuffer([server_outputs[world_id - 1].cpu()]) #
            #

            dist.send(tensor=flatten_model.buffer, dst=world_id)
            self.conf.logger.log(f"\tcomm_round={self.conf.graph.comm_round} local_round={self.conf.graph.local_round} Master send the server output to process_id={world_id}.")

    def _send_output_to_a_client(self, world_id, server_output):
        data = [server_output.cpu()]
        flatten_model = TensorBuffer(data) #
        # print(f"master send to client {world_id}", server_output)

        dist.send(tensor=flatten_model.buffer, dst=world_id)
        self.conf.logger.log(f"\tcomm_round={self.conf.graph.comm_round} local_round={self.conf.graph.local_round} Master send the server output to client-{world_id}.")

    def _receive_output_from_selected_clients(self): #ok
        self.conf.logger.log(f"Master waits to receive the local outputs.")
        result = dict()

        dist.barrier()
        # define the TensorBuffer for accepting the outputs from clients
        all_client_output_tb = dict()
        for world_id in self.world_ids:
            client_output = torch.zeros(self.conf.batch_size, self.conf.max_len, self.conf.hidden_size)
            attention_mask = torch.zeros(self.conf.batch_size, 1, 1, self.conf.max_len)
            # print(f"Attention_mask: {attention_mask.shape}")
            # label = torch.zeros(self.conf.mini_batch_size, self.conf.hidden_size)
            client_output_tb = TensorBuffer([client_output, attention_mask])
            all_client_output_tb[world_id] = client_output_tb

        reqs = []
        for world_id in self.world_ids:
            req = dist.irecv(tensor=all_client_output_tb[world_id].buffer, src=world_id)
            reqs.append(req)
        for req in reqs:
            req.wait()

        for world_id in self.world_ids:
            client_output = torch.zeros(self.conf.batch_size, self.conf.max_len, self.conf.hidden_size)
            attention_mask = torch.zeros(self.conf.batch_size, 1, 1, self.conf.max_len)
            all_client_output_tb[world_id].unpack([client_output, attention_mask])
            # print(f"master receive output from {world_id}: ", client_output)
            result[world_id] = {'client_output':client_output, 'attention_mask':attention_mask}
        self.conf.logger.log(f"\tcomm_round={self.conf.graph.comm_round} local_round={self.conf.graph.local_round} Master receive outputs from clients.")
        dist.barrier()

        return result

    def _receive_grad_from_selected_clients(self):
        result = dict()
        # define the TensorBuffer for accepting the outputs from clients
        all_client_grad_tb = dict()
        for world_id in self.world_ids:
            grad = torch.zeros(self.conf.batch_size, self.conf.max_len, self.conf.hidden_size)
            # label = torch.zeros(self.conf.mini_batch_size, self.conf.hidden_size)
            grad_tb = TensorBuffer([grad])
            all_client_grad_tb[world_id] = grad_tb

        reqs = []
        for world_id in self.world_ids:
            req = dist.irecv(tensor=all_client_grad_tb[world_id].buffer, src=world_id)
            reqs.append(req)
        for req in reqs:
            req.wait()

        for world_id in self.world_ids:
            grad = torch.zeros(self.conf.batch_size, self.conf.max_len, self.conf.hidden_size)
            all_client_grad_tb[world_id].unpack([grad])
            result[world_id] = grad
        self.conf.logger.log(f"\tcomm_round={self.conf.graph.comm_round} local_round={self.conf.graph.local_round} Master receive gradients from clients.")

        return result

    def _receive_grad_from_a_client(self, world_id):
        grad = torch.zeros(self.conf.batch_size, self.conf.max_len, self.conf.hidden_size)
        # label = torch.zeros(self.conf.mini_batch_size, self.conf.hidden_size)
        grad_tb = TensorBuffer([grad])
        dist.recv(tensor=grad_tb.buffer, src=world_id)

        grad = torch.zeros(self.conf.batch_size, self.conf.max_len, self.conf.hidden_size)
        grad_tb.unpack([grad])

        #print(f"master receive client - {world_id} grad: ", grad)

        self.conf.logger.log(f"\tcomm_round={self.conf.graph.comm_round} local_round={self.conf.graph.local_round} Master receive gradient from client-{world_id}.")
        return grad



    def _receive_models_from_selected_clients(self, selected_client_ids):
        self.conf.logger.log(f"Master waits to receive the local models.")
        dist.barrier()

        # init the placeholders to recv the local models from workers.
        # 初始化存储模型参数的空间
        flatten_local_models = dict() # selected_client_id -> 扁平化存储模型参数的类TensorBuffer
        for selected_client_id in selected_client_ids:
            client_tb = TensorBuffer(
                list(self.client_model.state_dict().values()) # 每个client端的模型确定，所以待接收的模型参数列表大小确定
            )
            client_tb.buffer = torch.zeros_like(client_tb.buffer) # buffer内的数全部初始化为0
            flatten_local_models[selected_client_id] = client_tb

        # async to receive model from clients.
        reqs = []
        for client_id, world_id in zip(selected_client_ids, self.world_ids):
            req = dist.irecv( # irecv表示异步接收，表示什么？
                tensor=flatten_local_models[client_id].buffer, src=world_id
            ) # 从client那儿接受模型参数

            reqs.append(req)

        for req in reqs:
            req.wait()

        dist.barrier()
        self.conf.logger.log(f"comm_round={self.conf.graph.comm_round} Master received all local models.")
        return flatten_local_models

    def _receive_head_models_from_selected_clients(self, selected_client_ids):
        self.conf.logger.log(f"Master waits to receive the local head models.")
        dist.barrier()

        # init the placeholders to recv the local models from workers.
        # 初始化存储模型参数的空间
        flatten_local_head_models = dict() # selected_client_id -> 扁平化存储模型参数的类TensorBuffer
        for selected_client_id in selected_client_ids:
            client_head_tb = TensorBuffer(
                list(self.client_head_model.state_dict().values()) # 每个client端的模型确定，所以待接收的模型参数列表大小确定
            )
            client_head_tb.buffer = torch.zeros_like(client_head_tb.buffer) # buffer内的数全部初始化为0
            flatten_local_head_models[selected_client_id] = client_head_tb

        # async to receive model from clients.
        reqs = []
        for client_id, world_id in zip(selected_client_ids, self.world_ids):
            req = dist.irecv( # irecv表示异步接收，表示什么？
                tensor=flatten_local_head_models[client_id].buffer, src=world_id
            ) # 从client那儿接受模型参数

            reqs.append(req)

        for req in reqs:
            req.wait()

        dist.barrier()
        self.conf.logger.log(f"comm_round={self.conf.graph.comm_round} Master received all local models.")
        return flatten_local_head_models

    def _receive_optims_from_selected_clients(self, selected_client_ids):
        self.conf.logger.log(f"Master waits to receive the local optims.")
        dist.barrier()

        # init the placeholders to recv the local models from workers.
        # 初始化存储模型参数的空间
        flatten_local_optims = dict()  # selected_client_id -> 扁平化存储模型参数的类TensorBuffer
        for selected_client_id in selected_client_ids:
            client_tb = TensorBuffer(
                list(self.client_optims[selected_client_id - 1].state_dict().values())  # 每个client端的模型确定，所以待接收的模型参数列表大小确定
            )
            client_tb.buffer = torch.zeros_like(client_tb.buffer)  # buffer内的数全部初始化为0
            flatten_local_optims[selected_client_id] = client_tb

        # async to receive model from clients.
        reqs = []
        for client_id, world_id in zip(selected_client_ids, self.world_ids):
            req = dist.irecv(  # irecv表示异步接收，表示什么？
                tensor=flatten_local_optims[client_id].buffer, src=world_id
            )  # 从client那儿接受模型参数

            reqs.append(req)

        for req in reqs:
            req.wait()

        for selected_client_id in selected_client_ids:
            client_optim_state_dict = self.client_optims[selected_client_id - 1].state_dict()
            flatten_local_optims[selected_client_id].unpack(client_optim_state_dict.values())
            self.client_optims[selected_client_id - 1].load_state_dict(client_optim_state_dict)

        dist.barrier()
        self.conf.logger.log(
            f"comm_round={self.conf.graph.comm_round} local_round={self.conf.graph.local_round} Master received all local optims.")

    def _aggregate_model_and_evaluate(self, flatten_local_models):
        self.conf.logger.log(
            f"comm_round={self.conf.graph.comm_round} Master aggregate the client_model."
        )
        self.client_model_tb.buffer = torch.zeros_like(self.client_model_tb.buffer)
        for key, value in flatten_local_models.items():
            self.client_model_tb.buffer += value.buffer
        self.client_model_tb.buffer = self.client_model_tb.buffer / len(flatten_local_models)
        self.client_model_tb.unpack(self.client_state_dict.values())

        tmp_state_dict = copy.deepcopy(self.client_state_dict)
        for world_id in self.world_ids:
            if not self.conf.fedavg_embedding_head:
                flatten_local_models[self.selected_client_ids[world_id - 1]].unpack(tmp_state_dict.values())
                for k in self.embedding_state_key:
                    assert k in tmp_state_dict
                    self.client_state_dict[k] = tmp_state_dict[k]
                for k in self.lm_head_state_key:
                    assert k in tmp_state_dict
                    self.client_state_dict[k] = tmp_state_dict[k]
            self.client_models[world_id - 1].load_state_dict(self.client_state_dict)

    def _aggregate_head_model_and_evaluate(self, flatten_local_head_models):
        self.conf.logger.log(
            f"comm_round={self.conf.graph.comm_round} Master aggregate the client_head_model.\n\n"
        )
        self.client_head_model_tb.buffer = torch.zeros_like(self.client_head_model_tb.buffer)
        for key, value in flatten_local_head_models.items():
            self.client_head_model_tb.buffer += value.buffer
        self.client_head_model_tb.buffer = self.client_head_model_tb.buffer / len(flatten_local_head_models)
        self.client_head_model_tb.unpack(self.client_head_state_dict.values())

        tmp_state_dict = copy.deepcopy(self.client_head_state_dict)
        for world_id in self.world_ids:
            if not self.conf.fedavg_embedding_head:
                flatten_local_head_models[self.selected_client_ids[world_id - 1]].unpack(tmp_state_dict.values())
                for k in self.lm_head_state_key:
                    assert k in tmp_state_dict
                    self.client_head_state_dict[k] = tmp_state_dict[k]
            self.client_head_models[world_id - 1].load_state_dict(self.client_head_state_dict)

    def _aggregate_server_model(self):
        self.conf.logger.log(f"comm_round={self.conf.graph.comm_round} Master aggregate the server model.")
        worker_state_dict = [self.tmp_server_model[world_id - 1].cpu().state_dict() for world_id in self.world_ids]
        weight_keys = list(worker_state_dict[0].keys())
        fed_state_dict = collections.OrderedDict()
        for key in weight_keys:
            key_sum = 0
            for i in range(self.conf.n_participated):
                key_sum = key_sum + worker_state_dict[i][key]
            fed_state_dict[key] = key_sum / self.conf.n_participated
        self.server_model.load_state_dict(fed_state_dict)

    def _load_avg_server_model(self):
        self.conf.logger.log(f"comm_round={self.conf.graph.comm_round} Master load average server model.")
        for world_id in self.world_ids:
            self.tmp_server_model[world_id - 1].load_state_dict(self.server_model.state_dict())
            # self.tmp_server_model[world_id - 1].to(self.parallel_device_ids[world_id - 1])
            if not self.conf.server_model_to_cpu:
                self.tmp_server_model[world_id - 1].train().to(self.parallel_device_ids[world_id - 1])
            self.optim[world_id - 1] = AdamW(self.tmp_server_model[world_id - 1].parameters(), lr=self.learning_rate, betas=(0.9, 0.999), weight_decay=0.01, eps=1e-6)

    # (learning_rate - end_learning_rate) * (1 - global_step / decay_steps) ^ (power) + end_learning_rate
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
        for world_id in self.world_ids:
            for param in self.optim[world_id - 1].param_groups:
                param['lr']=self.learning_rate

    # client_outputs, attention_masks, labels
    def update_server_model(self):
        # server-side 的model不做avg，线性方式在同一个上训练
        if not self.conf.fedavg_server_model:
            batch_iter = tqdm.tqdm(range(self.data_loader_length), desc=f"Global Round-{self.conf.graph.comm_round}, Local Round-{self.conf.graph.local_round}")
            for batch_index in batch_iter: # 循环 len(data_loader) 次
                self.conf.logger.log(f"comm_round={self.conf.graph.comm_round} local_round={self.conf.graph.local_round} batch_index={batch_index} Master start one batch of training.")
                dist.barrier()
                # 收集所有client中间输出对应的server中间输出
                client_outputs = self._receive_output_from_selected_clients()
                server_outputs = []
                for world_id in self.world_ids:
                    encoder_out, attention_mask = client_outputs[world_id]['client_output'].to(self.device), client_outputs[world_id]['attention_mask'].to(self.device)
                    server_output = self.server_model(client_output=encoder_out, extended_attention_mask=attention_mask) # labels为空，解决隐私问题
                    # server_outputs.append(server_output[0])
                    self._send_output_to_a_client(world_id=world_id, server_output=server_output[0])
                    client_grad = self._receive_grad_from_a_client(world_id=world_id).to(self.device)

                    #print(f"pre - {world_id}: ", self.server_model.state_dict()['roberta.encoder.layer.0.attention.self.key.weight'])
                    self.optim.zero_grad()
                    server_output[0].backward(client_grad)
                    self.optim.step()

                    #print(f"suf - {world_id}: ", self.server_model.state_dict()['roberta.encoder.layer.0.attention.self.key.weight'])
                    self.conf.logger.log(
                        f"comm_round={self.conf.graph.comm_round}, local_round={self.conf.graph.local_round}, Master update server model with client-{world_id}!"
                    )
                dist.barrier()

            self.conf.logger.log(
                f"comm_round={self.conf.graph.comm_round}, local_round={self.conf.graph.local_round}, Master finished one local_round of training!"
            )
        # server-side model针对每个client分开训练，然后做avg
        else:
            batch_iter = tqdm.tqdm(range(self.data_loader_length), desc=f"Global Round-{self.conf.graph.comm_round}, Local Round-{self.conf.graph.local_round}")
            for batch_index in batch_iter:  # 循环 len(data_loader) 次
                self.conf.logger.log(f"comm_round={self.conf.graph.comm_round} local_round={self.conf.graph.local_round} batch_index={batch_index} Master start one batch of training.")

                dist.barrier()
                # 收集所有client中间输出对应的server中间输出
                client_outputs = self._receive_output_from_selected_clients()
                server_outputs = []
                for world_id in self.world_ids:
                    self.optim[world_id - 1].zero_grad()
                    if self.conf.server_model_to_cpu:
                        self.tmp_server_model[world_id - 1].train().to(self.parallel_device_ids[world_id - 1])
                    encoder_out, attention_mask = client_outputs[world_id]['client_output'].to(self.parallel_device_ids[world_id - 1]), client_outputs[world_id]['attention_mask'].to(self.parallel_device_ids[world_id - 1])
                    server_output = self.tmp_server_model[world_id - 1](client_output=encoder_out, extended_attention_mask=attention_mask)  # labels为空，解决隐私问题
                    server_outputs.append(server_output[0].cpu())
                    if self.conf.server_model_to_cpu:
                        self.tmp_server_model[world_id - 1].cpu()

                # 发送server中间输出给client
                self._send_output_to_selected_clients(self.selected_client_ids, server_outputs)

                # 接收server的梯度
                client_grads = self._receive_grad_from_selected_clients()

                # 更新server端模型
                for world_id in self.world_ids:
                    if self.conf.server_model_to_cpu:
                        self.tmp_server_model[world_id - 1].train().to(self.parallel_device_ids[world_id - 1])
                    server_outputs[world_id - 1], client_grads[world_id], self.tmp_server_model[world_id - 1] = server_outputs[world_id - 1].to(self.parallel_device_ids[world_id - 1]), client_grads[world_id].to(self.parallel_device_ids[world_id - 1]), self.tmp_server_model[world_id - 1].to(self.parallel_device_ids[world_id - 1])
                    #if world_id == 1:
                    #    print(f"Master: pre",self.tmp_server_model[0].state_dict()['roberta.encoder.layer.0.attention.self.key.weight'])
                    self.optim[world_id - 1].zero_grad()
                    server_outputs[world_id - 1].backward(client_grads[world_id])
                    self.optim[world_id - 1].step()
                    #if world_id == 1:
                    #    print(f"Master: after",self.tmp_server_model[0].state_dict()['roberta.encoder.layer.0.attention.self.key.weight'])
                    if self.conf.server_model_to_cpu:
                        self.tmp_server_model[world_id - 1].cpu()
                # 更新learning_rate
                self.learning_rate_step()
                self.conf.logger.log(
                    f"\tcomm_round={self.conf.graph.comm_round} local_round={self.conf.graph.local_round} "
                    f"Master: optim_lr: {self.optim[0].param_groups[0]['lr']}"
                )

            self.conf.logger.log(
                f"comm_round={self.conf.graph.comm_round}, local_round={self.conf.graph.local_round}, Master finished one local_round of training!"
            )

    def  evaluate(self, comm_round):
        os.system(f'sh {self.conf.project_path}/Biobert-task/named-entity-recognition/run-all-splitFed.sh {self.conf.project_path} {self.conf.python_path}')
        os.system(f'python {self.conf.project_path}/Biobert-task/named-entity-recognition/show-all-result.py {self.file_name} {self.conf.graph.comm_round} {self.conf.project_path}')

    def aggregate_client_server(self, client_dict, client_head_dict, server_dict):

        result_dict = self.whole_model.state_dict()
        # set embedding parameters
        for key in self.embedding_state_key:
            if key not in result_dict:
                print("found error")
                return
            result_dict[key] = client_dict[key]

        # set lm_head parameters
        for key in self.lm_head_state_key:
            if key not in result_dict:
                print("found error")
                return
            result_dict[key] = client_head_dict[key]
        # set encoder parameters
        for key, value in client_dict.items():
            if key[0:21] == 'roberta.encoder.layer':
                result_dict[key] = value

        for key, value in server_dict.items():
            if key[0:21] == 'roberta.encoder.layer':
                sp = key.split('.')
                sp[3] = str(int(sp[3]) + self.conf.num_client_encoder)
                key = '.'.join(sp)
                # print(key)
                result_dict[key] = value
        self.whole_model.load_state_dict(result_dict)

# 产生每个participated_clients的epoch数量
def get_n_local_epoch(conf, n_participated):
    if conf.min_local_epochs is None:
        return [conf.local_n_epochs] * n_participated
    else:
        # here we only consider to (uniformly) randomly sample the local epochs.
        assert conf.min_local_epochs > 1.0
        random_local_n_epochs = conf.random_state.uniform(
            low=conf.min_local_epochs, high=conf.local_n_epochs, size=n_participated
        )
        return random_local_n_epochs
