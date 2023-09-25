#!/bin/bash 

export GLOO_SOCKET_IFNAME=ib0
export MIOPEN_USER_DB_PATH=/tmp/pytorch-miopen-2.8 
export HSA_USERPTR_FOR_PAGED_MEM=0 
lrank=$OMPI_COMM_WORLD_LOCAL_RANK 
comm_rank=$OMPI_COMM_WORLD_RANK 
comm_size=$OMPI_COMM_WORLD_SIZE 

#par_rate=`awk 'BEGIN{printf "%.1f\n",('$3'/'$2')}'`
#echo par_rate=${par_rate} client_encoder=${4} server_encoder=${5} local rank=${lrank} comm_rank=${comm_rank}

APP="python3 ../single_process.py \
         --arch Roberta --complex_arch master=RobertaServer,worker=RobertaClient \
         --data wikitext-pubmed --pin_memory True --batch_size 16 --num_workers 1 \
         --fl_aggregate scheme=federated_average --hostfile hostfile \
         --manual_seed 7 --pn_normalize True --same_seed_process False \
         --partition_data random \
         --world_conf 0,0,1,1,1 --on_cuda True \
         --n_clients 10 --participation_ratio 0.4 \
         --n_comm_rounds 18 --n_local_rounds 2 \
         --lr 5e-5 --end_lr 5e-6 --warmup_ratio 0.1\
         --num_client_encoder 6 --num_server_encoder 6 \
         --num_device 4 --fedavg_embedding_head True --fedavg_server_model True \
         --gpu ${lrank} --num_batch 10000 \
         --project_path /public/home/yaodz/wangzw/Roberta \
         --data_dir /public/home/yaodz/wangzw/PubMed/preprocessed_data --tot_files 18 \
         --checkpoint_save_dir /public/home/yaodz/wangzw/Roberta/FedBioBertModel/FedBioBert_10_4_6_6_16_wholePubmed \
         --init_method tcp://${1}:23456 --comm_rank ${comm_rank} \
         --from_checkpoint False --from_checkpoint_round 0 --server_model_to_cpu False"
    
${APP}