#!/bin/bash
#SBATCH -p palamut-cuda
#SBATCH -A tbag88
#SBATCH -J am31s
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 16
#SBATCH --gres=gpu:1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=yourmail@gmail.com
#SBATCH --time=20:00:00
#SBATCH --output=/truba/home/tbag88/bugrabaran/try/slurm-%j.out
#SBATCH --error=/truba/home/tbag88/bugrabaran/try/slurm-%j.err
echo CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES
###################  Bu arayi degistirmeyin ##########################
export PATH=/truba_scratch/eakbas/software/cuda-9.0/bin:$PATH
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/truba_scratch/eakbas/software/cuda-9.0/lib64
######################################################################

### Resnet-12 Protonet
#python train_fsl.py --max_epoch 200 --model_class ProtoNet --backbone_class Res12 --dataset TieredImageNet --way 5 --eval_way 5 --shot 5 --eval_shot 5 --query 15 --eval_query 15 --balance 0.1 --temperature 32 --temperature2 32 --lr 0.002 --lr_mul 10 --lr_scheduler step --step_size 40 --gamma 0.5 --gpu $CUDA_VISIBLE_DEVICES --init_weights ./saves/initialization/tieredimagenet/Res12-pre.pth --eval_interval 1 --use_euclidean 


### Resnet-12 FEAT
#python train_fsl.py  --max_epoch 200 --model_class FEAT  --backbone_class Res12 --dataset MiniImageNet --way 5 --eval_way 5 --shot 1 --eval_shot 1 --query 15 --eval_query 15 --balance 0.01 --temperature 64 --temperature2 64 --lr 0.0002 --lr_mul 10 --lr_scheduler step --step_size 40 --gamma 0.5 --gpu $CUDA_VISIBLE_DEVICES --init_weights ./saves/initialization/miniimagenet/Res12-pre.pth --eval_interval 1 --use_euclidean

### Resnet-12 DiscTransformer
python train_fsl.py  --max_epoch 200 --model_class CombinedProtoNet  --backbone_class Res12 --dataset MiniImageNet --way 5 --eval_way 5 --shot 5 --eval_shot 5 --query 15 --eval_query 15 --balance 0.1 --temperature 32 --lr 0.002 --lr_mul 10 --lr_scheduler step --step_size 40 --gamma 0.5 --gpu $CUDA_VISIBLE_DEVICES --init_weights ./saves/initialization/miniimagenet/Res12-pre.pth --eval_interval 1 --warmup_epoch 0 --num_acc 1 --vis_rate 0.85 --weight_decay 0.0005

### Resnet-12 AM3
#python train_fsl.py  --max_epoch 200 --model_class AM3  --backbone_class Res12 --dataset MiniImageNet --way 15 --eval_way 15 --shot 1 --eval_shot 1 --query 15 --eval_query 15 --balance 0.1 --temperature 32 --temperature2 32 --lr 0.002 --lr_mul 10 --lr_scheduler step --step_size 20 --gamma 0.5 --gpu $CUDA_VISIBLE_DEVICES --init_weights ./saves/initialization/miniimagenet/Res12-pre.pth  --eval_interval 1 --warmup_epoch 0 --use_euclidean 

##### Resnet-12 DiscTransformer Step vs Acc
#python test_fsl.py  --max_epoch 200 --model_class OnlySatt  --backbone_class Res12 --dataset MiniImageNet --way 5 --eval_way 5 --shot 5 --eval_shot 5 --query 15 --eval_query 15 --balance 0.1 --temperature 32 --temperature2 32 --lr 0.002 --lr_mul 10 --lr_scheduler step --step_size 40 --gamma 0.5 --gpu $CUDA_VISIBLE_DEVICES --init_weights ./saves/initialization/miniimagenet/Res12-pre.pth --eval_interval 1 --use_euclidean  --noise_type ONLYTEST 

#### Resnet-12 Pretraining 

#python pretrain.py --lr 0.1 --batch_size 128 --max_epoch 500 --backbone_class Res12 --schedule 350 400 440 460 480 --ngpu 1 --gamma 0.1 --ignore_p2v

#python visualize.py
