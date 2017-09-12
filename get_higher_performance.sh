mkdir -p ./freezed_model

# Freeze graphs
python ~/miniconda3/envs/carnd-term3/lib/python3.5/site-packages/tensorflow/python/tools/freeze_graph.py \
--input_graph=./models_l2_norm_lr00001_e100/eproch_90_loss_7867.2935 \
--input_checkpoint=./models_l2_norm_lr00001_e100/eproch_90_loss_7867.2935 \
--input_binary=true \
--output_graph=./freezed_model/frozen_graph.pb \
--output_node_names=predicted_label

python freezed_model/graph_utils.py

python /home/huboqiang/miniconda3/envs/carnd-term3/lib/python3.5/site-packages/tensorflow/python/tools/optimize_for_inference.py \
--input=frozen_graph.pb \
--output=optimized_graph.pb \
--frozen_graph=True \
--input_names=image_input \
--output_names=predicted_label                              




# bazel
echo "deb [arch=amd64] http://storage.googleapis.com/bazel-apt stable jdk1.8" | sudo tee /etc/apt/sources.list.d/bazel.list
curl https://bazel.build/bazel-release.pub.gpg | sudo apt-key add -
sudo apt-get update
sudo apt-get install bazel

cd ..
git clone -b v1.2.1 https://github.com/tensorflow/tensorflow
cd ./tensorflow
./configure
bazel build tensorflow/tools/graph_transforms:transform_graph

cd ../CarND-Semantic-Segmentation/freezed_model

../../tensorflow/bazel-bin/tensorflow/tools/graph_transforms/transform_graph \
    --in_graph=frozen_graph.pb \
    --out_graph=eightbit_graph.pb \
    --inputs=image_input \
    --outputs=predicted_label \
    --transforms='
add_default_attributes
remove_nodes(op=Identity, op=CheckNumerics)
fold_constants(ignore_errors=true)
fold_batch_norms
fold_old_batch_norms
fuse_resize_and_conv
quantize_weights
quantize_nodes
strip_unused_nodes
sort_by_execution_order'


cd CarND-Semantic-Segmentation
