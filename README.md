# IMAGE-TO-IMAGE TRANSLATION USING A CROSS-DOMAIN AUTO-ENCODER AND DECODER

![architecture_applsci](https://user-images.githubusercontent.com/36982015/65848814-bf861580-e382-11e9-9d1c-1aef991849db.png)

This repository contains the official TensorFlow implementation of the following paper:

> **IMAGE-TO-IMAGE TRANSLATION USING A CROSS-DOMAIN AUTO-ENCODER AND DECODER**<br>
> Jaechang Yoo, Heesong Eom, Yong Suk Choi<br>
> 
> **Abstract:** *Recently, a number of studies have focused on image-to-image translation. However, the quality of the translation results is lacking in certain respects. We propose a new image-to-image translation method to minimize such shortcomings using an auto-encoder and an auto-decoder. This method includes pre-training two auto-encoder and decoder pairs for each source and target image domain, cross-connecting two pairs and adding a feature mapping layer. Our method is quite simple and straightforward to adopt but very effective in practice, and we experimentally demonstrate that our method can significantly enhance the quality of image-to-image translation. We use the well-known cityscapes, horse2zebra, cat2dog, maps, summer2winter, and night2day datasets. Our method shows qualitative and quantitative improvements over existing models.*

## Usage
### Train / Test
* pre-trained encoder and decoder weights must be located in save_photo\"dataset_name"_256\ (for domain A) and save_label/"dataset_name"_256/ (for domain B).
* dataset must be located in datasets/"dataset_name".
* You can download Cityscapes, horse2zebra, Maps and other useful datasets : [[Download](https://people.eecs.berkeley.edu/~taesung_park/CycleGAN/datasets/)]
* You can download cat2dog dataset here: [[Download](https://github.com/brownvc/ganimorph/)]

##### Train Example:
```bash
$ python new_main_correct.py --phase train --dataset_dir maps --epoch 200 --batch_size 1
```
##### Test Example:
```bash
$ python new_main_correct.py --phase test --dataset_dir maps --which_direction AtoB --batch_size 1
```

## Results
![results_paper](https://user-images.githubusercontent.com/36982015/65853430-6f17b380-e394-11e9-994b-31c904f7276b.jpg)



