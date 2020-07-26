# SDAN: Stacked Diverse Attention Network for Human Action recognition

## Summary

This is the PyTorch code for the following papers:





This code includes training, fine-tuning and testing on Kinetics, UCF-101, and HMDB-51.

## Pretrained models

These models are trained on Kinetics.

```
SDAN-50-kinetics.pth: --model resnet_CBAM --model_depth 50 --resnet_shortcut B
```



## Requirements

* [PyTorch](http://pytorch.org/)

```bash
conda install pytorch torchvision cuda80 -c soumith
```

* FFmpeg, FFprobe

```bash
wget http://johnvansickle.com/ffmpeg/releases/ffmpeg-release-64bit-static.tar.xz
tar xvf ffmpeg-release-64bit-static.tar.xz
cd ./ffmpeg-3.3.3-64bit-static/; sudo cp ffmpeg ffprobe /usr/local/bin;
```

* Python 3



## Preparation

### Kinetics

* Download videos using [the official crawler](https://github.com/activitynet/ActivityNet/tree/master/Crawler/Kinetics).
  * Locate test set in ```video_directory/test```.
* Convert from avi to jpg files using ```utils/video_jpg_kinetics.py```

```bash
python utils/video_jpg_kinetics.py avi_video_directory jpg_video_directory
```

* Generate n_frames files using ```utils/n_frames_kinetics.py```

```bash
python utils/n_frames_kinetics.py jpg_video_directory
```

* Generate annotation file in json format similar to ActivityNet using ```utils/kinetics_json.py```
  * The CSV files (kinetics_{train, val, test}.csv) are included in the crawler.

```bash
python utils/kinetics_json.py train_csv_path val_csv_path test_csv_path dst_json_path
```

### UCF-101

* Download videos and train/test splits [here](http://crcv.ucf.edu/data/UCF101.php).
* Convert from avi to jpg files using ```utils/video_jpg_ucf101_hmdb51.py```

```bash
python utils/video_jpg_ucf101_hmdb51.py avi_video_directory jpg_video_directory
```

* Generate n_frames files using ```utils/n_frames_ucf101_hmdb51.py```

```bash
python utils/n_frames_ucf101_hmdb51.py jpg_video_directory
```

* Generate annotation file in json format similar to ActivityNet using ```utils/ucf101_json.py```
  * ```annotation_dir_path``` includes classInd.txt, trainlist0{1, 2, 3}.txt, testlist0{1, 2, 3}.txt

```bash
python utils/ucf101_json.py annotation_dir_path
```

### HMDB-51

* Download videos and train/test splits [here](http://serre-lab.clps.brown.edu/resource/hmdb-a-large-human-motion-database/).
* Convert from avi to jpg files using ```utils/video_jpg_ucf101_hmdb51.py```

```bash
python utils/video_jpg_ucf101_hmdb51.py avi_video_directory jpg_video_directory
```

* Generate n_frames files using ```utils/n_frames_ucf101_hmdb51.py```

```bash
python utils/n_frames_ucf101_hmdb51.py jpg_video_directory
```

* Generate annotation file in json format similar to ActivityNet using ```utils/hmdb51_json.py```
  * ```annotation_dir_path``` includes brush_hair_test_split1.txt, ...

```bash
python utils/hmdb51_json.py annotation_dir_path
```

## Running the code

Assume the structure of data directories is the following:

```misc
~/
  data/
    kinetics_videos/
      jpg/
        .../ (directories of class names)
          .../ (directories of video names)
            ... (jpg files)
    results/
      save_100.pth
    kinetics.json
```

Confirm all options.

```bash
python main.lua -h
```

Train SDAN-34 on the Kinetics dataset (400 classes) with 4 CPU threads (for data loading).  
Batch size is 128.  
Save models at every 5 epochs.
All GPUs is used for the training.
If you want a part of GPUs, use ```CUDA_VISIBLE_DEVICES=...```.

```bash
python main.py --root_path ~/data --video_path kinetics_videos/jpg --annotation_path kinetics.json \
--result_path results --dataset kinetics --model resnet_CBAM \
--model_depth 50 --n_classes 400 --batch_size 128 --n_threads 4 --checkpoint 5
```

Continue Training from epoch 101. (~/data/results/save_100.pth is loaded.)

```bash
python main.py --root_path ~/data --video_path kinetics_videos/jpg --annotation_path kinetics.json \
--result_path results --dataset kinetics --resume_path results/save_100.pth \
--model_depth 34 --n_classes 400 --batch_size 128 --n_threads 4 --checkpoint 5
```

Fine-tuning conv5_x and fc layers of a pretrained model (~/data/models/SDAN-50-kinetics.pth) on UCF-101.

```bash
python main.py --root_path ~/data --video_path ucf101_videos/jpg --annotation_path ucf101_01.json \
--result_path results --dataset ucf101 --n_classes 400 --n_finetune_classes 101 \
--pretrain_path models/SDAN-50-kinetics.pth --ft_begin_index 4 \
--model SDAN --model_depth 50 --resnet_shortcut A --batch_size 128 --n_threads 4 --checkpoint 5
```

