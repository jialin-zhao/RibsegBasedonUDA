# RibsegBasedonUDA

data format json
  - trainA
  - trainB
  - testA
  - testB

cycle train python train.py --name expername --gpu_ids 0,1 --n_epochs 100 --n_epochs_decay 100 --dataroot /opt/data/private/TBdetection/Task_ribsegment/cyclegan/data/szmc_0_ribfrac.json --batch_size 8
  dataroot 

test cp ./log/expername/latest_net_G_A.pth ./log/expername/latest_net_G.pth
        python test.py --name expername --no_dropout --dataroot /opt/data/private/TBdetection/Task_ribsegment/cyclegan/data/szmc_0_ribfrac.json
