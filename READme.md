# Benchmarking Energy Efficiency of SNN Pruning Algorithms on SATA

Details of the project are provided in this ![file](https://github.com/RakshithaKalkura/benchmarking-snn-pruning/blob/main/cs6886_project_ppt.pdf)

### Installation:

```
git clone https://github.com/RakshithaKalkura/benchmarking-snn-pruning.git

cd benchmarking-snn-pruning

conda create -n env python=3.10

conda activate env

pip install torch torchvision numpy pyyaml tqdm matplotlib spikingjelly

```

### Experiments
#### Training:
All frameworks use ResNet-19 SNN on CIFAR-10.
Example training command (uticket shown; same structure for others):
```
python train_resnet_cifar10_stp.py \
  --data-path ../data \
  --output-dir ../saved_models/stp_resnet19 \
  --epochs 300 \
  --batch-size 64 \
  --T 10

```

#### Convert Checkpoints

```
python convert.py .path_to_model(saved_in_.pth.tar)

```

#### Profiling Sparsities and YAMLs

```
python profile_sparsities.py \
  --checkpoint ../saved_models/uticket_resnet19/checkpoint_state.pth \
  --dataset-root ../data \
  --outdir ../sata_inputs/uticket \
  --batch-size 128 \
  --n-profile 400 \
  --T 10

  ```

#### Simulations

Specify paths appropriately in the scripts

  ```
  cd SATA_Sim\inference-energy-cal
  python energy-cal.py
  ```


### Contact: cs25e052@smail.iitm.ac.in
