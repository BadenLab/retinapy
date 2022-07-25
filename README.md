Python package used in the Baden Lab for handing data from retina recordings.

Installation
------------

	pip install retinapy


Usage
-----
There isn't much functionality yet. What you can do is generate the spike
windows (snippets):

```
>>> import retinapy.mea as mea
>>> rec = mea.load_3brain_recordings(
...    "./data/ff_noise.h5",
...    "./data/ff_recorded_noise.pickle",
...    "./data/ff_spike_response.pickle",
...    include=["Chicken_17_08_21_Phase_00"])[0]
>>> # Extract spike windows.
>>> # I'm downsampling by 18, and this gives ~1000 timestep per second (931).
>>> downsample_factor = 18
>>> snippet_len = 80
>>> snippet_pad = 20
>>> snippets, cluster_ids, sample_freq = mea.labeled_spike_snippets(
...    rec, snippet_len, snippet_pad, downsample_factor)
>>> print(snippets.shape)
(215941, 80, 4)

```

