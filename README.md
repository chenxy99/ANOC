Attention for Novel Object Captioning (ANOC)
=====================================

This code implements the Attention for Novel Object Captioning (ANOC)

Reference
------------------
If you use our code or data, please cite our paper:
```text
Anonymous submission for ECCV 2020, paper ID 865.
```

Disclaimer
------------------
We adopt the official implementation of the [`nocaps`](https://github.com/nocaps-org/updown-baseline) as a baseline model for novel object captioning. We use the bottom-up features provided in this repository. Please refer to these links for further README information.

Requirements
------------------
1. Requirements for Pytorch. We use Pytorch 1.1.0 in our experiments.
2. Requirements for Tensorflow. We only use the tensorboard for visualization.
3. Python 3.6+ (for most of our experiments)

ANOC
------------------
The program can be executed by
```text
CUDA_VISIBLE_DEVICES=0 python scripts/train.py --config configs/updown_plus_cbs_saliency_nocaps_val.yaml \
--checkpoint-every 1000 --gpu-ids 0 --serialization-dir checkpoints/anoc
```

If you would like to train with SCST, you can base on the previous best result and execute the following script
```text
CUDA_VISIBLE_DEVICES=0 python scripts/train_scst.py --config configs/updown_plus_cbs_saliency_nocaps_val.yaml \
--config-override OPTIM.BATCH_SIZE 50 OPTIM.LR 0.00005 OPTIM.NUM_ITERATIONS 210000 --checkpoint-every 3000 \
--gpu-ids 0 --serialization-dir checkpoints/anoc_scst \
--start-from-checkpoint checkpoints/anoc/checkpoint_best.pth
```

Results
------------------
### ANOC w/o SCST:
<table>
  <tr>
    <th colspan="2">in-domain</th>
    <th colspan="2">near-domain</th>
    <th colspan="2">out-of-domain</th>
    <th colspan="6">overall</th>
  </tr>
  <tr>
    <th>CIDEr</th><th>SPICE</th>
    <th>CIDEr</th><th>SPICE</th>
    <th>CIDEr</th><th>SPICE</th>
    <th>BLEU1</th><th>BLEU4</th><th>METEOR</th><th>ROUGE</th><th>CIDEr</th><th>SPICE</th>
  </tr>
  <tr>
    <td>79.9</td><td>12.0</td>
    <td>75.2</td><td>11.6</td>
    <td>70.7</td><td>9.7</td>
    <td>76.6</td><td>18.6</td><td>24.2</td><td>51.9</td><td>75.0</td><td>11.3</td>
  </tr>
</table>

### ANOC with SCST:
<table>
  <tr>
    <th colspan="2">in-domain</th>
    <th colspan="2">near-domain</th>
    <th colspan="2">out-of-domain</th>
    <th colspan="6">overall</th>
  </tr>
  <tr>
    <th>CIDEr</th><th>SPICE</th>
    <th>CIDEr</th><th>SPICE</th>
    <th>CIDEr</th><th>SPICE</th>
    <th>BLEU1</th><th>BLEU4</th><th>METEOR</th><th>ROUGE</th><th>CIDEr</th><th>SPICE</th>
  </tr>
  <tr>
    <td>86.1</td><td>12.0</td>
    <td>80.7</td><td>11.9</td>
    <td>73.7</td><td>10.1</td>
    <td>78.4</td><td>19.1</td><td>24.8</td><td>52.2</td><td>80.1</td><td>11.6</td>
  </tr>
</table>
