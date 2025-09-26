TPU_VM_NAME="<TPU_VM_NAME>"
TPU_ZONE="<TPU_ZONE>"
PROJECT="<PROJECT>"

# Copy .env file to TPU VM
gcloud alpha compute tpus tpu-vm scp .env $TPU_VM_NAME:~/torchprime/ \
    --zone=$TPU_ZONE \
    --project=<PROJECT> \
    --tunnel-through-iap \
    --worker=all
