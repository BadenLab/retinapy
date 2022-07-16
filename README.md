Python package used in the Baden Lab for handing data from retina recordings.

Installation
------------

	pip install retinapy


Usage
-----
There isn't much functionality yet. What you can do is generate the spike
windows (snippets):

	import retinapy
	import retinapy.mea_noise as mea

	stimulus = mea.load_fullfield_stimulus('./data/ff_noise.h5')
	response = mea.load_response('./data/ff_spike_response.pickle')
	# Choose one of the recordings.
	rec_name = mea.recording_names(response)[0]
	# Zoom the stimulus.
	stimulus_freq = 20
	stimulus_zoom = 2
	sampling_freq = stimulus_zoom * stimulus_freq
	stimulus_upsampled = mea.upsample_stimulus(stimulus, factor=stimulus_zoom)
	# Extract spike windows.
	snippets, cluster_ids = mea.labeled_spike_snippets(
		stimulus_upsampled, 
		response, 
		rec_name,
		snippet_len=5, 
		snippet_pad=2, 
		sampling_freq=sampling_freq)

