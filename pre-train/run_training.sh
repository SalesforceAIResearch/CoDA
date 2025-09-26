TPU_VM_NAME="<TPU_VM_NAME>" # Change with your TPU VM name
TPU_ZONE="<TPU_ZONE>"
PROJECT="<PROJECT>"
BRANCH="<BRANCH>"
RECIPE="recipes/<RECIPE>"

gcloud alpha compute tpus tpu-vm ssh $TPU_VM_NAME \
    --zone=$TPU_ZONE \
    --project=$PROJECT \
    --tunnel-through-iap \
    --worker=all \
    --command='
    cd torchprime; \
    git fetch; \
    git checkout '"$BRANCH"'; \
    git pull; \
    source venv/bin/activate; \
    bash '"$RECIPE"'';
