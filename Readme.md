* model_artifacts:
- get_model.py: 
  Downloads the pretrained weights, sets the model to evaluation mode, and saves the weights to a file.
  This process is done once, and the resulting .pt file can be deployed with your model server, avoiding the need for each deployment to download the weights from the internet.
- imagenet_class_index.json:
  Extra file that you provide to the Torchserve server. This contains the class labels for printing the predictions.


* Generate .mar file:

torch-model-archiver \
  --model-name vit_b_16 \
  --version 1.0 \
  --model-file ./model_artifacts/get_model_vit_b_16.py \
  --serialized-file ./model_artifacts/vit_b_16.pt \
  --handler ./handler/vit_b_16_handler.py \
  --extra-files ./model_artifacts/imagenet_class_index.json \
  --export-path model_store

torch-model-archiver \
  --model-name vit_b_32 \
  --version 1.0 \
  --model-file ./model_artifacts/get_model_vit_b_32.py \
  --serialized-file ./model_artifacts/vit_b_32.pt \
  --handler ./handler/vit_b_32_handler.py \
  --extra-files ./model_artifacts/imagenet_class_index.json \
  --export-path model_store

torch-model-archiver \
  --model-name vit_l_16 \
  --version 1.0 \
  --model-file ./model_artifacts/get_model_vit_l_16.py \
  --serialized-file ./model_artifacts/vit_l_16.pt \
  --handler ./handler/vit_l_16_handler.py \
  --extra-files ./model_artifacts/imagenet_class_index.json \
  --export-path model_store

* Start Torchserve:
- Note: The first command to set the java home was required everytime before executing the torchserve command due to some issue(even though Java home was set previously).

export JAVA_HOME=$(/usr/lib/jvm/java-11-openjdk-amd64/bin/java)
torchserve --start --ncs --model-store model_store --models vit_b_16.mar --ts-config config/config.properties

export JAVA_HOME=$(/usr/lib/jvm/java-11-openjdk-amd64/bin/java)
torchserve --start --ncs --model-store model_store --models vit_b_32.mar --ts-config config/config.properties

*** Not Recommended ****
*** Each model consumes a significant GPU memory and we get out of resources error while serving inferences ***
* Start the server with multiple models:
export JAVA_HOME=$(/usr/lib/jvm/java-11-openjdk-amd64/bin/java)
torchserve --start --ncs --model-store model_store --models vit_b_16.mar vit_b_32.mar vit_l_16.mar --ts-config config/config.properties


* Useful Commands:
Current Frequency: nvidia-smi --query-gpu=clocks.gr,clocks.mem --format=csv
Supported frequencies: nvidia-smi -q -d SUPPORTED_CLOCKS
Maximum Frequency by default: nvidia-smi -q -d CLOCK
Overwrite the Maximum Frequency: sudo nvidia-smi -i [GPU_ID] -lgc [MIN_FREQUENCY],[MAX_FREQUENCY] (example: sudo nvidia-smi -i 0 -lgc 0,710)
Reset the Maximum Frequency: sudo nvidia-smi -i [GPU_ID] -rgc (example: sudo nvidia-smi -i 0 -rgc)