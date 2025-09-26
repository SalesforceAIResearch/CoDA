from datatrove.pipeline.readers import ParquetReader

# # limit determines how many documents will be streamed (remove for all)
# # to fetch a specific dump: hf://datasets/HuggingFaceFW/fineweb/data/CC-MAIN-2024-10
# # replace "data" with "sample/100BT" to use the 100BT sample
# data_reader = ParquetReader("hf://datasets/HuggingFaceFW/fineweb/data", limit=1000) 
# for document in data_reader():
#     # do something with document
#     print(document)

###############################    
# OR for a processing pipeline:
###############################

from datatrove.executor import LocalPipelineExecutor
from datatrove.pipeline.readers import ParquetReader
from datatrove.pipeline.filters import LambdaFilter
from datatrove.pipeline.writers import JsonlWriter

if __name__ == '__main__':
    pipeline_exec = LocalPipelineExecutor(
        pipeline=[
            # replace "data/CC-MAIN-2024-10" with "sample/100BT" to use the 100BT sample
            ParquetReader("hf://datasets/HuggingFaceFW/fineweb/sample/100BT"),
            # LambdaFilter(lambda doc: "hugging" in doc.text),
            JsonlWriter("data/fineweb-sample")
        ],
        tasks=10
    )
    pipeline_exec.run()

    ## process
    import json, os

    folder = './fineweb-sample/'  # replace with your folder path

    # Iterate over all files in the folder
    for filename in os.listdir(folder):
        if filename.endswith('.jsonl'):
            new_data = []
            with open(os.path.join(folder, filename), 'r') as f:
                for line in f:
                    data = json.loads(line)  # load json object from string
                    if 'id' in data:
                        del data['id']
                    if 'metadata' in data:
                        del data['metadata']
                    new_data.append(data)
            
            # Write the modified data to a new file
            with open(os.path.join(folder, 'clean_' + filename), 'w') as f:
                for item in new_data:
                    f.write(json.dumps(item) + '\n')  # convert json object to string