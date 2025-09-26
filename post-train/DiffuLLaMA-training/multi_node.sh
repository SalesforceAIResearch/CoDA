#!/bin/bash
FILE="hostname.txt"
echo "hostname text: $FILE"
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
WORKDIR=<PATH TO THE REPO> e.g. "/work/nvme/bcaq/shivama2/Efficient_Diffusion/"
CONDA_COMMAND=<COMMAND TO ACTIVATE THE ENVIRONMENT ON EACH NODE> e.g. "source /u/shivama2/llm_conta_new/bin/activate"
# Function to perform SSH and run the command in the background

rank=0
ssh_command() {

    local line=$1

    echo "Attempting to SSH into $line"

    ssh -o ConnectTimeout=5 "$line" "apptainer exec --nv  --bind /work /sw/user/NGC_containers/pytorch_24.07-py3.sif bash -c 'export HOSTNAME_FILE=$FILE; $CONDA_COMMAND; cd $WORKDIR; bash run_distributed.sh $rank'" < /dev/null
    if [ $? -eq 0 ]; then

        echo "SSH to $line successful"

    else

        echo "SSH to $line failed"

    fi

    (( rank += 1))

}


while IFS= read -r line; do

    ssh_command "$line" &

    (( rank += 1))

done < "$FILE"



# Wait for all background jobs to complete

wait



echo "All SSH commands executed."    # ssh -o ConnectTimeout=5 "$line" "apptainer exec --nv --bind /projects /sw/user/NGC_containers/pytorch_24.07-py3.sif bash -c 'export HOSTNAME_FILE=$FILE; $CONDA_COMMAND; cd $MEGATRON_DIR; echo '\''hello'\''; python -c '\''import torch; print(torch.cuda.is_available())'\'';'" < /dev/null #bash ./tools/run_sft_distributed.sh $rank" < /dev/null
