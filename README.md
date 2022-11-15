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

