# Discover the Unknown Ones in Fine-Grained Ship Detection

## 🛠️ Setup

### 1. Install Dependencies

```
conda create -n  DUONet python=3.8 -y
conda activate  DUONet

conda install pytorch=1.8.1 torchvision cudatoolkit=10.1 -c pytorch -y
pip install detectron2==0.5 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu101/torch1.8/index.html
git clone https://github.com/FoRGEU/DUONet.git
cd  DUONet
pip install -v -e .
```

### 2. Prepare Dataset

1. [ShipRSImageNet dataset](https://github.com/zzndream/ShipRSImageNet))
2. [DOSR dataset]([yaqihan-9898/DOSR: a public dataset for oriented ship recognition](https://github.com/yaqihan-9898/DOSR))
3. [HRSC2016 dataset](https://www.kaggle.com/datasets/guofeng/hrsc2016)

The files should be organized in the following structure:

```
DUONet/
└── datasets/
    └── ShipRSImageNet_V1/
        ├── JPEGImages
        ├── ImageSets
        └── Annotations
    └── DOSR/
        ├── JPEGImages
        ├── ImageSets
        └── Annotations        
    └── HRSC2016/
        ├── JPEGImages
        ├── ImageSets
        └── Annotations            
```

You can use the script in the `tools` folder to perform category division.

Then, Dataloader and Evaluator followed for  DUONet is in VOC format.

## 🚀 Training

The training process is the same as detectron2, or run the You can use the script in the tools file to perform category divisionfile.
```bash
python /home/quchenyu/DUONet/train_net.py --config-file /home/quchenyu/DUONet/configs/ShipRS_config_37+5.yaml
```

## 📈 Evaluation

For reproducing any of the above mentioned results please run the `eval.sh` file and add pretrained weights accordingly.

**Note:**
For more training and evaluation details please check the [Deformable DETR](https://github.com/fundamentalvision/Deformable-DETR) reposistory.

## 🔧 License

This repository is released under the Apache 2.0 license as found in the [LICENSE](LICENSE) file.

## 📋 Citation

If you use this code in your research, please cite our paper (BibTeX will be provided upon publication).

