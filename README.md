Python package used in the Baden Lab for handing data from retina recordings.

Installation
------------

	pip install retinapy


Usage
-----
There isn't much functionality yet. What you can do is generate the spike
windows (snippets):

```

>>> import retinapy.mea_noise as mea
>>> rec_name = 'Chicken_17_08_21_Phase_00'
>>> stimulus_pattern = mea.load_stimulus_pattern('./data/ff_noise.h5')
>>> recorded_stimulus = mea.load_recorded_stimulus('./data/ff_recorded_noise.pickle')
>>> response = mea.load_response('./data/ff_spike_response.pickle')
>>> # Create an array representing the played stimulus. It can be downsampled.
>>> # I'm choosing 18, as this gives ~1000 timestep per second (931).
>>> downsample_factor = 18
>>> stimulus, freq = mea.decompress_stimulus(stimulus_pattern, 
...    recorded_stimulus, rec_name, downsample_factor)
>>> # Extract spike windows.
>>> snippets, cluster_ids = mea.labeled_spike_snippets(
...   stimulus, 
...   response, 
...   'Chicken_17_08_21_Phase_00',
...   freq,
...   mea.ELECTRODE_FREQ,
...   snippet_len=5, 
...   snippet_pad=2)
>>> print(snippets.shape)
(215941, 5, 4)

```

