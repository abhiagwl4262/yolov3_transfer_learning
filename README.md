This repo is for experimentation of Transfer Learning using Google Big Transfer Models. 


Command to run for yolov3 Training - 
CUDA_VISIBLE_DEVICES=3 python3 -W ignore train.py --data_config ../data/GrimaXray_data/grima.data --batch_size 32 --compute_map True --checkpoint_path from_scratch_new_nms --learning_rate 1e-3

Command to run for yolov3 Transfer Learning - 
CUDA_VISIBLE_DEVICES=4 python3 -W ignore train_BiT.py --data_config ../data/GrimaXray_data/grima.data --batch_size 16 --compute_map True --checkpoint_path resnet101x1_backbone_0.0001 --bit_model_type BiT-M-R101x1 --learning_rate 1e-4

CUDA_VISIBLE_DEVICES=5 python3 -W ignore train_resnet.py --data_config ../data/GrimaXray_data/grima.data --batch_size 8 --compute_map True --checkpoint_path checkpoint_resnet101 --backbone_type resnet101 --learning_rate 1e-3