#!/bin/bash 
#SBATCH -p fat
#SBATCH -N 5
#SBATCH -J FedBioBERT_10_4_6_6_16_Parallel_wholePubmed
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=6
#SBATCH --gres=dcu:4

hostfile=./$SLURM_JOB_ID
scontrol show hostnames $SLURM_JOB_NODELIST > ${hostfile} 
num_node=$(cat $hostfile|sort|uniq |wc -l) 

num_DCU=$(($num_node*4)) 
nodename=$(cat $hostfile |sed -n "1p") 
dist_url=`echo $nodename | awk '{print $1}'` 

rm `pwd`/hostfile-dl -f 
cat $hostfile|sort|uniq >`pwd`/tmp 

tot_tasks=5
for i in `cat ./tmp` 
do
	cur_slots=1
    tot_tasks=`expr $tot_tasks - $cur_slots`
    echo for $i cur_slots=$cur_slots
	echo ${i} slots=$cur_slots >> `pwd`/hostfile-dl 
done

mpirun -np 5 --allow-run-as-root -hostfile `pwd`/hostfile-dl `pwd`/single_process.sh $dist_url