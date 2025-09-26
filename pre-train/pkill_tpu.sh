COMMAND="ps -aux | grep python | grep -v grep |  awk '{print \$2}' | xargs kill -9"
TPU_NAME="<TPU_NAME>"
TPU_ZONE="<TPU_ZONE>"
PROJECT="<PROJECT>"

gcloud alpha compute tpus tpu-vm ssh root@$TPU_NAME \
    --zone=$TPU_ZONE \
    --project=$PROJECT \
    --tunnel-through-iap \
    --worker=all \
    --command="$COMMAND"