# Ambisonic Blind Reverberation Time Estimation

This repository contains the complementary code for the paper:

> Andrés Pérez-López, Archontis Politis, and Emilia Gómez. 
"Blind reverberation time estimation from ambisonic recordings". 
Submitted to the IEEE 22nd International Workshop on Multimedia Signal Processing (MMSP 2020).

It implements a novel method for ambisonic blind reverberation time estimation, 
based on a Multichannel Auto-Recursive dereverberation followed by filter identification.



## Getting Started

1. `create_IRs.py` 
This script will create a set of 100 IRs, with which to work later on.

2. `run_experiment_dev_baseline.py`
It will run the baseline system on the development set, 
with the aim of estimating the linear fitting parameters at the output.

3. `run_experiment.py`
This is the main script. It will run the three described methods on the testing set.
It might take a while...

4. `analysis_dev_baseline.py`
It analyzes the output produced by `run_experiment_dev_baseline.py`, 
and computes the fitting parameters using the groundtruth values.

5. `analysis.py`
All method results are analyzed and plotted here.


## Datasets

The current implementation makes use of two datasets:
- [DSD100](https://sigsep.github.io/datasets/dsd100.html)
>  Antonie Liutkus, Fabian-Robert Stöter, Zafar Rafii, Daichi Kitamura, Bertrand Rivet, Nobutaka Ito, Nobutaka Ono, and JulieF ontecave.
"The 2016 Signal Separation Evaluation Campaign",
In Proceedings of the Latent Variable Analysis and Signal Separation - 12th International Conference (LVA/ICA 2015).

- [Librispeech ASR corpus](http://www.openslr.org/12)

> Vassil Panayotov, Guoguo Chen, Daniel Povey, and Sanjeev Khudanpur
"LibriSpeech: an ASR corpus based on public domain audio books",
 In Proceedings of the IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP 2015).
 
 Please make sure that you have them in your system, and you set the paths accordingly.
 
 Nonetheless, any other desired dataset could be potientially used. 
 
 
## Dependencies
- numpy
- soundfile
- masp
- matlab.engine (you will be a working matlab installation to run the ctf code. 
However, that can be easily ported to python... sorry!)


## License

All the code here, excepting the contents of the `ctf` folder, is covered under the [3-Clause BSD License](https://opensource.org/licenses/BSD-3-Clause). 