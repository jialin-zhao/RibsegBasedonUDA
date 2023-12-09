# RibsegBasedonUDA

A framework for rib segmentation in CXR images based on unsupervised domain adaptation

![framework](/images/resunet-ribsegroute3.png)

## Data

- DRRs generation: We employ a parallel projection model [1] to generate DRR images from CT images.
- Data format in Cycle-GAN: json
  - trainA
  - trainB
  - testA
  - testB
- Data format in SegNet: json
  - train
    - imgs
    - masks
  - test
    - imgs
    - masks

## Usage

1. SegNet 
   - `cd segnet` and open the file `train_config.yaml` and set your json path and other parameters
   ```bash
    python train.py train_config.yaml
  
2. Cycle-GAN 
   - `cd cyclegan`
   - Training:
     ```bash
     python train.py --name yourExperName --gpu_ids 0,1 --n_epochs 100 --n_epochs_decay 100 --dataroot yourJsonDataRoot --batch_size 8
   - Test:
     ```bash
     cp ./log/expername/latest_net_G_A.pth ./log/expername/latest_net_G.pth
     python test.py --name yourExperName --no_dropout --dataroot yourJsonDataRoot
   
## References
[1] Campo, M.I., Pascau, J. and Estépar, R.S.J., 2018, April. Emphysema quantification on simulated X-rays through deep learning techniques. In 2018 IEEE 15th International Symposium on Biomedical Imaging (ISBI 2018) (pp. 273-276). IEEE.
