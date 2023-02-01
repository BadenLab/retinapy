The main purpose of this library is to create a predictive model for retinal ganglion cell activity: given stimulus and spike history, predict the next 100ms or so of spike activity. A secondary objective is to provide an efficient and easy to use API for interacting with data collected from multi-electrode array (MEA) data.

Installation
============

	pip install retinapy


Spike prediction
================
`spikeprediction.py` and `models.py` define neural network models and 
training objectives used for the prediction task. `dataset.py` turns the 
mea data into PyTorch datasets for consumption by the training loop.

Dataset
-------
Currently, The model is trained and tested on MEA recordings of chicken retina exposed to a full-field noise stimulus, collected by Marvin Seifert (https://doi.org/10.1101). The model is trained to predict the individual output of >1000 "cells" collected over 18 recordings. Cells in quotes as they are the puported cells identified by the spike sorting algorithm. 

Performance
-----------
Spike prediction for one cell are shown below, for about 5 seconds of test data:

https://user-images.githubusercontent.com/1439017/202433626-d06751e3-c619-472a-8491-19d28fbcfaaa.mp4

Below, for the same cell, predicted and ground truth spikes are counted in 100 ms bins. The data is for ~86 seconds of test data, without smoothing over time or averaging over multiple trials. 


![infer100ms](https://user-images.githubusercontent.com/1439017/202437623-8f740415-1a62-4bad-a07b-8d99719c6574.png)

(A line chart probably isn't so appropriate here, but it makes a visual comparison easier compared to just using points)

Clustering
----------
The model is trained once for all cells. `(recording_id, cell_id)` tuples are encoded via a variational auto-encoder. The aim here is that in the future, an additional network can be used to place unknown cells into the embedding space so as to be able to do spike prediction for additional cells from new recordings. A consequence of this approach is that the encoding space can be inspected to see if any interesting clustering has emerged. Below is a screenshot of a t-SNE plot of the latent space. On inspection, the STA kernels for nearby points are similar.

![latent](https://user-images.githubusercontent.com/1439017/202499239-ae0a0b44-f378-41fd-a35d-cb1743a7ff79.png)


Neural network components
=========================
Some reusable Pytorch modules live in `nn.py`.


Training loop
=============
`train.py` and `_logging.py` contain general purpose training infrastructure
in PyTorch, focused around a training loop. It supports a slimmed down
feature set of what you might get from a library such as FastAI or Pytorch 
Lightning.


Basics
======
If you aren't training deep learning models, you might still find some of the
functionality in the `retinapy.mea` module useful. It handles loading MEA data 
and provides some useful functions such as downsampling and data splitting. 
Look no further if you just want to extract spike snippets for 
spike-triggered-averaging.

```
>>> import retinapy.mea as mea
>>> rec = mea.single_3brain_recording(
...    rec_name="Chicken_17_08_21_Phase_00",
...    data_dir="./data/ff_noise_recordings")
>>> # Extract spike windows.
>>> # I'm downsampling by 18, and this gives ~1000 timestep per second (992).
>>> downsample = 18
>>> rec = mea.decompress_recording(rec, downsample)
>>> snippets = rec.spike_snippets(total_len=800, post_spike_len=200)
>>> print(len(snippets))
154
>>> print(snippets[0].shape)
(17993, 800, 4)

```

Plotting
========
Various plotting functions are collected in `vis.py`. Plotly is the main
plotting library being used in this module.

Spike snippet viewer
====================
For a bit of fun, there is a visualization tool in `/snippet_viewer` which
inspects cells by viewing all of the snippets that contribute to the cell's
STA kernel. Below is a video showing 500 ms leading up to every spike of a 
given cell. 

A hosted version is at: https://mea.bio/

https://user-images.githubusercontent.com/1439017/182321360-0df3c046-d300-4c88-bfea-d07060d5679f.mp4

The stimulus is 50-50 on-off 4-channel color noise. Each square shows the time pregression of the stimulus shown to the cell leading up to a spike. Our stimulator has 4 LEDs. In the visualization, the intensity of the 3 LEDs most similar to the red, green and blue sRGB primaries are mapped to the sRGB color values of the inner square, and and the intensity of the 4th LED (UV) is visualized as a purple boarder.




