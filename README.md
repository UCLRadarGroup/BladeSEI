# The BladeSEI Dataset for Radar-Centric Specific Emitter Identification (SEI)

![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)

This repo contains the supplementary code for **BladeSEI**, a radar focussed dataset for specific emitter identification (SEI) of BladeRF software-defined radios.

- [What is this repository for?](#what-is-this-repository-for)
- [The dataset](#the-dataset)
- [Repository contents](#repository-contents)
- [Installation](#installation)
- [Getting started](#getting-started)
- [Citation](#citation)
- [Licence](#licence)
- [Questions and issues](#questions-and-issues)

## What is this repository for?

Specific emitter identification (SEI) is the task of telling individual transmitters apart from the tiny, hardware-specific distortions that each one imprints on its signals, even when they are the same make and model and transmit the same waveform.

BladeSEI contains pulses from 12 BladeRF emitters, captured at the same time by four ADC channels of an RFSoC receiver, on different days and at different temperatures. The same emitters can therefore be observed through different receiver channels and under changing conditions, which makes the dataset suited to studying:

- how well emitters can be discriminated at different signal-to-noise ratios (SNRs);
- how well a model generalises to a different receiver channel (ADC), a different day, or a different temperature.

This repository contains the code for working with the dataset: dataloaders that filter, align and augment the pulses, two neural-network baselines and a PCA + kNN baseline, and notebooks that show how to use them.

## The dataset

### Where to download

**TODO: add the download link / DOI here.**

Download the data and extract it into a `data/` folder in the repository root.

### What it contains

| | |
|---|---|
| **Emitters** | 12 BladeRF software-defined radios (`blade1`, `blade2`, ...) |
| **Waveforms** | `pulse`: an unmodulated 10 µs pulse at 2.46 GHz<br>`chirp`: a 50 µs chirped pulse with 40 MHz bandwidth, sweeping 2.43 to 2.47 GHz |
| **Receiver** | ARESTOR, a ZCU111 RFSoC-based system with four ADC channels (`adc0`, `adc2`, `adc4`, `adc6`) recording the same pulses at the same time using multi-tile synchronisation |
| **Sampling** | The ADCs sample at 3.84 GS/s. The pulses lie in the second Nyquist zone, so they are aliased in the sampled data. The aliased signals are mixed down to baseband using digital down-conversion (DDC) with a 1.39 GHz mixing frequency, and then decimated by a factor of 32, giving complex baseband at 120 MHz |
| **Captures** | Day 1 (cold), day 2 (cold) and day 2 (hot). Each pulse is labelled with its day, environment, temperature, emitter and waveform. There are roughly 4000 pulses per capture, per ADC.|

To increase the variability between the BladeRFs, the analogue bandwidth of each device was set to 20 MHz. This pushes the start and end of the chirp outside the passband of the analogue filters, which are physical components that differ between units, and so amplifies the hardware fingerprint.


### File format

A basic pulse detector was used to detect the pulses from the continuous ADC data.

Pulses are stored as HDF5 files (`*.h5`), which the loader finds recursively under `hdf5_dir`. Each file contains:

| Path | Contents |
|---|---|
| `pulses` | One row per pulse: complex baseband samples as interleaved I/Q |
| `labels` (attributes) | `day`, `environment`, `temp`, `emitter`, `pulse_type` |
| `metadata` (attributes) | `source_file` (identifies the ADC, e.g. `adc0.npy`), RFSoC settings (`RFSoC_centre_freq`, `RFSoC_sample_rate`, `RFSoC_mixing_freq`, `RFSoC_nyquist_zone`, `RFSoC_decimation_factor`), `sample_rate`, `chunk_index`, `num_pulses` |
| `pulse_descriptor_words/` | One dataset per field, one value per pulse, including `time_of_arrival`, `snr`, `noise_floor` and `pulse_power`. This is mostly used for alignment purposes and to support future capability, as all pulses should have identical pulse descriptor words (PDWs)|

The loader identifies a file's ADC from the stem of its `source_file` attribute (or of the file's own name), so `adc_names=["adc0"]` selects the files recorded from `adc0`.

## Repository contents

```
BladeSEI/
├── dataloaders/
│   ├── dataloader.py          SimplePulseDataset and AugmentedPulseDataset
│   └── augmentations.py       SNR degradation, CFO / phase shifts, magnitude channel, normalisation, flattening
├── models/
│   ├── resnet.py              ResNet12_1D
│   └── inception.py           Inception1DNet
├── dataloader_demo.ipynb      How to use the dataloaders
├── pca_demo.ipynb             Train and validate a PCA + kNN classifier
├── dnn_demo.ipynb             Train and validate a ResNet (or Inception) based classifier
├── requirements.txt           Pinned package versions (Linux, CUDA 12)
├── rsei_env.yml               Conda environment (Python 3.10)
├── LICENSE
└── README.md
```

## Installation

```bash
git clone https://github.com/UCLRadarGroup/BladeSEI.git
cd BladeSEI
```

**Option 1: conda (recommended).** This recreates the environment used for the experiments (Python 3.10, PyTorch 2.10, CUDA 12 libraries):

```bash
conda env create -f rsei_env.yml
conda activate RSEI
pip install jupyter
```

**Option 2: pip.** Install packages using requirements.txt file.

```bash
pip install -r requirements.txt
```

## Getting started 

Three example Jupyter notebooks are provided to demonstrate the dataloader and its capabilities, and to show a PCA-based and a DNN-based classifier.

The notebooks are saved with their outputs, so you can browse the results on GitHub without running them.

| Notebook | What it shows |
|---|---|
| [`dataloader_demo.ipynb`](dataloader_demo.ipynb) | Loading and viewing raw pulses with `SimplePulseDataset`; the ADC views of one physical pulse in `"single"` and `"contrastive"` mode; SNR degradation with `AugmentedPulseDataset`; CFO and phase correction, before and after |
| [`pca_demo.ipynb`](pca_demo.ipynb) | Training a PCA + kNN classifier on day-1, chamber, cold chirp pulses from `adc0`; saving the model, deleting it and reloading it from disk; evaluating across 30 to 0 dB SNR with confusion matrices and an F1-versus-SNR plot |
| [`dnn_demo.ipynb`](dnn_demo.ipynb) | Training a ResNet on the same data, with per-epoch testing, the best weights restored and early stopping; saving and reloading the model; the same SNR sweep. `Inception1DNet` can be used in place of `ResNet12_1D` |

## Citation

If you use BladeSEI, please cite:

**TODO: add the paper reference and BibTeX here.**

```bibtex
@misc{bladesei,
  title  = {TODO},
  author = {TODO},
  year   = {TODO},
  url    = {https://github.com/UCLRadarGroup/BladeSEI}
}
```

## Licence

The code in this repository is released under the [GNU General Public License v3.0](LICENSE).

The BladeSEI dataset is distributed separately. **TODO: state the dataset licence here.**

## Questions and issues

Please open an [issue](https://github.com/UCLRadarGroup/BladeSEI/issues) for questions and bug reports.
