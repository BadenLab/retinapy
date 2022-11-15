Python package used in the Baden Lab for handing data from retina recordings.

Installation
============

	pip install retinapy


Spike prediction
================
The main purpose of this library is to create a predictive model for 
retinal ganglion cell activity; given stimulus and spike history, predict the 
next 100ms or so of spike activity.

`spikeprediction.py` and `models.py` define neural network models and 
training objectives used for the prediction task. `dataset.py` turns the 
mea data into Pytorch datasets for consumption by the training loop.


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
The `retinapy.mea` module loads mea data and provides some useful functions
such as downsampling and splitting data. Look no further if you just want to
extract spike snippets for spike-triggered-averaging.

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







