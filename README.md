# Human-Centric Image Captioning

This is an official Pytorch implementation for Human-Centric Image Captioning task.

This repo is modified from the well-known [codebase](https://github.com/ruotianluo/self-critical.pytorch) by [Ruotian Luo](https://github.com/ruotianluo).

## Requirements
- Python 3.6
- Java 1.8.0
- PyTorch 1.0
The submodules (cider and coco-caption) could be downloaded [here](https://github.com/ruotianluo/self-critical.pytorch#prepare-data)  

## Prepare Data
Please refer to [here](https://github.com/ruotianluo/self-critical.pytorch#prepare-data)

## Download Pre-processed Features
1. Please download the Updown features and VC features for body part regions.
2. Please download the VC features
3. Please download the Updown features

## Pretrained models

Checkout [MODEL_ZOO.md](MODEL_ZOO.md).

If you want to do evaluation only, you can then follow [this section](#generate-image-captions) after downloading the pretrained models (and also the pretrained resnet101 or precomputed bottomup features, see [data/README.md](data/README.md)).

#### Start training
```bash
$ python train_hc.py --id HCCM --caption_model HCCM --input_json data/cocotalk.json --input_label_h5 data/cocotalk_label.h5 --input_att_dir_vc [the/path/to/VC_Feature/trainval] --input_att_dir [the/path/to/Updown_Feature] --body_part_dir [the/path/to/body_part_Updown_Feature] --body_part_vc_dir [the/path/to/body_part_VC_Feature] --part_mask_dir [the/path/to/part_mask_dir] --batch_size 10 --learning_rate 2e-4 --checkpoint_path log_hc --save_checkpoint_every 4000 --val_images_use 2500 --max_epochs 80 --rnn_size 2048 --input_encoding_size 1024 --self_critical_after 30 --language_eval 1 --learning_rate_decay_start 0 --scheduled_sampling_start 0 --use_vc
```

NOTE: This command mix the cross-entropy and self-critical training. If you want to training them separately, you may need:


## Cross Entropy Training
```bash
$ python train_hc.py --id HCCM --caption_model HCCM --input_json data/cocotalk.json --input_label_h5 data/cocotalk_label.h5 --input_att_dir_vc [the/path/to/VC_Feature/trainval] --input_att_dir [the/path/to/Updown_Feature] --body_part_dir [the/path/to/body_part_Updown_Feature] --body_part_vc_dir [the/path/to/body_part_VC_Feature] --part_mask_dir [the/path/to/part_mask_dir] --batch_size 10 --learning_rate 2e-4 --checkpoint_path log_hc --save_checkpoint_every 4000 --val_images_use 2500 --rnn_size 2048 --input_encoding_size 1024 --max_epochs 30 --language_eval 1
```
## Self-critical Training
```bash
$ python train_hc.py --id HCCM --caption_model HCCM --caption_model HCCM --input_json data/cocotalk.json --input_label_h5 data/cocotalk_label.h5 --input_att_dir_vc [the/path/to/VC_Feature/trainval] --input_att_dir [the/path/to/Updown_Feature] --body_part_dir [the/path/to/body_part_Updown_Feature] --body_part_vc_dir [the/path/to/body_part_VC_Feature] --part_mask_dir [the/path/to/part_mask_dir] --batch_size 10 --learning_rate 2e-4 --start_from log_hc --checkpoint_path log_hc --save_checkpoint_every 4000 --language_eval 1 --val_images_use 2500 --self_critical_after 30 --rnn_size 2048 --input_encoding_size 1024 --cached_tokens coco-train-idxs --max_epoch 80
```

## Evaluation
```bash
python eval_hc.py --model log_hc/model-best.pth --infos_path log_hc/infos_HCCM-best.pkl  --dump_images 0 --num_images -1 --language_eval 1 --beam_size 5 --batch_size 50 --split test
```
