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
	sampling_freq = 40
	stimulus_upsampled = mea.upsample_stimulus(stimulus, new_freq=sampling_freq)
	spike_windows = mea.spike_windows(stimulus_upsampled, response, 
		kernel_len=5, post_kernel_pad=2, sampling_freq=sampling_freq)
		
