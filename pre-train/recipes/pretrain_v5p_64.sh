export LIBTPU_INIT_ARGS="--xla_tpu_scoped_vmem_limit_kib=98304 --xla_enable_async_all_gather=true --xla_tpu_overlap_compute_collective_tc=true --xla_tpu_enable_async_collective_fusion_multiple_steps=true --xla_tpu_enable_async_collective_fusion=true --xla_tpu_enable_async_collective_fusion_fuse_all_gather=true"
export XLA_IR_DEBUG=1
export XLA_HLO_DEBUG=1
export TPU_PREMAPPED_BUFFER_SIZE=40000000000

# sample recipe to run pretrain on TPU v5p-64
python torchprime/torch_xla_models/train.py \
    run_name=<run_name> \
    training_mode=pretrain \
    reshape_context=false \
    seg_attn=true \
    data=<pretrain_dataset> \
    model=coda-qwen-1b-tpu-v5p \
    model.block_masking_probability=0.01 \
    model.mask_block_sizes=[2,4,8] \
    model.truncate_probability=0.01 \
    model.prefix_probability=0.01 \
    model.masking_scheduler.schedule_type=constant \
    model.masking_scheduler.max_schedule_steps=null \
    optimizer.learning_rate=3e-4 \
    lr_scheduler.type=cosine \
    lr_scheduler.warmup_steps=2000 \
    global_batch_size=512 \
    max_steps=210000 \
    checkpoint_load_dir=null \
    checkpoint_load_step=null \
    resume_from_checkpoint=false \
    checkpoint_save_dir=gs://<bucket_name>/checkpoints/<pretrain_checkpoint_dir>/ \
    save_steps=5000 \
    logging_steps=1 \
    ici_mesh.fsdp=16 \
    ici_mesh.tensor=2 \
    ici_mesh.data=1 \
    ici_mesh.expert=1 \
    model/remat=qwen-scan
# fsdp * tensor * data * expert == num_devices
