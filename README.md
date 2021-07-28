# Human-Centric Image Captioning

This is an official Pytorch implementation for Human-Centric Image Captioning task.

This repo is modified from the well-known [codebase](https://github.com/ruotianluo/self-critical.pytorch) by [Ruotian Luo](https://github.com/ruotianluo).

## HC-COCO
HC-COCO contains 16,125 images and 78,462 sentences, with more than 70% of the captions focusing on human actions and more than 49% focusing on human-object interactions. Furthermore, ten body part bounding boxes for each person are annotated. The dataset can be downloaded [here](https://drive.google.com/file/d/16R3BUK6iOv9v3PgGgmPg9ACP-Y9ZxG9N/view?usp=sharing)

## Requirements
- Python 3.6
- Java 1.8.0
- PyTorch 1.0

The submodules (cider and coco-caption) could be downloaded [here](https://github.com/ruotianluo/self-critical.pytorch#prepare-data)  

## Prepare Data
Please refer to [here](https://github.com/ruotianluo/self-critical.pytorch#prepare-data) and place the [file](https://drive.google.com/file/d/1_WAxGJ3uhE7wrwKmDzvX-aFJxQxF_Crm/view?usp=sharing) into ./coco-caption/annotations/

## Download Pre-processed Features
1. Please download the [Updown features](https://drive.google.com/drive/folders/1XjsxL-hy6fG5h7EkekoUclk345BLr9hd?usp=sharing) and [VC features](https://drive.google.com/drive/folders/1XjsxL-hy6fG5h7EkekoUclk345BLr9hd?usp=sharing) for body part regions.
2. Please download the [VC features](https://drive.google.com/file/d/1O-JAYhdF3z8fkLivXZzllT8PotV1MlRv/view?usp=sharing)
3. Please download the [Updown features](https://drive.google.com/file/d/1J62N8HLjNaPell0UdByMyt-bbl8UGlSL/view?usp=sharing)
4. Please download the [part masks](https://drive.google.com/drive/folders/1XjsxL-hy6fG5h7EkekoUclk345BLr9hd?usp=sharing)

## Pretrained model

The pre-trained model can be download [here](https://drive.google.com/drive/folders/1PY6tvHWLpoZlXdhewaXv5SpJAJUKv4Mp?usp=sharing)

## Start training
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
