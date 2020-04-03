# TensorFlow benchmarks
This repository contains various TensorFlow benchmarks. Currently, it consists of two projects:


1. [PerfZero](https://github.com/tensorflow/benchmarks/tree/master/perfzero): A benchmark framework for TensorFlow.

2. [scripts/tf_cnn_benchmarks](https://github.com/tensorflow/benchmarks/tree/master/scripts/tf_cnn_benchmarks): The TensorFlow CNN benchmarks contain benchmarks for several convolutional neural networks.


# Running the benchmarks ( Tensorflow )

# Parameters that can be changed 

- FP 
- xla 
- Batch Size 
- Model 
- Number of GPUs 
- Iterations 

## FP32 - BATCH 64 

## Navigate to the the folder in BizonOS
```bash
cd /home/bizon/benchmarks/tensorflow/benchmarks-master/scripts/tf_cnn_benchmarks
```

### RUN
```bash
python tf_cnn_benchmarks.py --num_gpus=1 --batch_size=64 --model=resnet152 --variable_update=parameter_server

python tf_cnn_benchmarks.py --num_gpus=4 --batch_size=64 --model=resnet152 --variable_update=parameter_server
```

## FP32 - XLA - BATCH 64

### RUN 
```bash
python tf_cnn_benchmarks.py --data_format=NCHW --batch_size=64 --num_batches=100 --model=inception4 --optimizer=momentum --variable_update=replicated --all_reduce_spec=nccl --nodistortions --gradient_repacking=2 --datasets_use_prefetch=True --per_gpu_thread_count=2 --loss_type_to_report=base_loss --compute_lr_on_cpu=True --single_l2_loss_op=True --xla_compile=True --local_parameter_device=gpu --num_gpus=1 --display_every=10

python tf_cnn_benchmarks.py --data_format=NCHW --batch_size=64 --num_batches=100 --model=vgg16 --optimizer=momentum --variable_update=replicated --all_reduce_spec=nccl --nodistortions --gradient_repacking=2 --datasets_use_prefetch=True --per_gpu_thread_count=2 --loss_type_to_report=base_loss --compute_lr_on_cpu=True --single_l2_loss_op=True --xla_compile=True --local_parameter_device=gpu --num_gpus=4 --display_every=10
```

## FP16 - BATCH 64 

### RUN 
```bash
python tf_cnn_benchmarks.py --num_gpus=1 --batch_size=64 --model=resnet50 --variable_update=parameter_server --use_fp16=True

python tf_cnn_benchmarks.py --num_gpus=4 --batch_size=64 --model=resnet50 --variable_update=parameter_server --use_fp16=True
```

## FP 16 - XLA - BATCH 64 

### RUN 
```bash
python tf_cnn_benchmarks.py --data_format=NCHW --batch_size=64 --num_batches=100 --model=inception4 --optimizer=momentum --variable_update=replicated --all_reduce_spec=nccl --use_fp16=True --nodistortions --gradient_repacking=2 --datasets_use_prefetch=True --per_gpu_thread_count=2 --loss_type_to_report=base_loss --compute_lr_on_cpu=True --single_l2_loss_op=True --xla_compile=True --local_parameter_device=gpu --num_gpus=1 --display_every=10

python tf_cnn_benchmarks.py --data_format=NCHW --batch_size=64 --num_batches=100 --model=vgg16 --optimizer=momentum --variable_update=replicated --all_reduce_spec=nccl --use_fp16=True --nodistortions --gradient_repacking=2 --datasets_use_prefetch=True --per_gpu_thread_count=2 --loss_type_to_report=base_loss --compute_lr_on_cpu=True --single_l2_loss_op=True --xla_compile=True --local_parameter_device=gpu --num_gpus=4 --display_every=10
```

# Reults charts

- [Bizon chart results]()
