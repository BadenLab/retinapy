import flask
import json
import retinapy
import retinapy.vis
import retinapy.mea as mea
from typing import Dict, Tuple
import numpy as np
import numpy.typing as npt
import plotly


#DATA_DIR = "./resources/ff_noise_recordings"
#KERNEL_DIR = "chicken_kernel_plots_tooltip"
DATA_DIR = "./resources/frog_ff_noise"
KERNEL_DIR = "frog_kernel_plots_tooltip"
CACHE_DIR = "./resources/snippets"
DEFAULT_DOWNSAMPLE = 180
# DEFAULT_KERNEL_LEN_MS = 1100 # ms
# DEFAULT_KERNEL_PAD_MS = 100 # ms

"""
TODO: use caching instead.
"""
recs = None
rec_ids = None
rec_cluster_ids: mea.RecClusterIds = None
# Acceleration structure
recs_by_id: mea.RecIds = None
# Cache the downsampled stimuli for the default downsampling rate.
dc_stimulus_cache: Dict[Tuple[int, int], npt.ArrayLike] = {}


def init_data():
    global recs, recs_dc, rec_ids, rec_cluster_ids, recs_by_id, default_dc_stimuli
    recs = mea.load_3brain_recordings(DATA_DIR)
    rec_ids, rec_cluster_ids = mea.load_id_info(DATA_DIR)
    recs_by_id = {rec_ids[rec.name]: rec for rec in recs}
    # default_dc_stimuli = {
    #     rec_ids[rec.name]: mea.decompress_stimulus(
    #         rec.stimulus_pattern,
    #         rec.stimulus_events,
    #         rec.num_sensor_samples,
    #         DEFAULT_DOWNSAMPLE,
    #     )
    #     for rec in recs
    # }


def get_stimulus(rec_id: int, downsample: int = DEFAULT_DOWNSAMPLE):
    """
    Downsample or draw from cache.
    """
    res = None
    if (rec_id, downsample) in dc_stimulus_cache:
        res = dc_stimulus_cache[(rec_id, downsample)]
    else:
        recording = recs_by_id[rec_id]
        res = mea.decompress_stimulus(
            recording.stimulus_pattern,
            recording.stimulus_events,
            recording.num_sensor_samples,
            downsample,
        )
        dc_stimulus_cache[(rec_id, downsample)] = res
    return res


bp = flask.Blueprint("api", __name__)


@bp.get("/recordings")
def recordings():
    res = []
    for rec_id, rec in recs_by_id.items():
        r = {
            "id": rec_id,
            "name": rec.name,
            "sensorSampleRate": rec.sensor_sample_rate,
        }
        r["cellIds"] = rec.cluster_ids
        res.append(r)
    res = json.dumps(res)
    return res


@bp.get("/recording/<int:rec_id>/cells")
def cells(rec_id):
    print(f"present? {rec_id in recs_by_id}")
    rec = recs_by_id[rec_id]
    res = json.dumps(rec.cluster_ids)
    return res


@bp.get("/recording/<int:rec_id>/cell/<int:cell_id>/kernelplot/tooltip.png")
def kernelplot_tooltip(rec_id, cell_id):
    rec = recs_by_id[rec_id]
    # Serve from static.
    # path = f"kernel_plots_tooltip/{rec.name}_c{cell_id}.png"
    path = f"{KERNEL_DIR}/{rec.name}_c{cell_id}.png"
    return flask.send_from_directory("static", path)


# Used by 'classic'
@bp.get("/recording/<int:rec_id>/cell/<int:cell_id>")
def cell(rec_id, cell_id):
    rec = recs_by_id[rec_id].clusters({cell_id})
    rec_dc = mea.decompress_recording(rec, downsample=DEFAULT_DOWNSAMPLE)
    snippets = rec_dc.spike_snippets(cell_id, total_len=90, post_spike_len=10)
    res = json.dumps([snip.tolist() for snip in snippets])
    return res


def filter_spikes(spikes, min_dist):
    if len(spikes) == 0:
        return spikes
    first = [spikes[0]]
    other = []
    for idx in range(1, len(spikes)):
        if spikes[idx] - spikes[idx - 1] < min_dist:
            other.append(spikes[idx])
        else:
            first.append(spikes[idx])
    return first, other


def cell_spikes(rec_id, cell_id, downsample):
    rec = recs_by_id[rec_id]
    cell_idx = rec.cid_to_idx(cell_id)
    spikes = rec.spike_events[cell_idx]
    spikes = mea.rebin_spikes(spikes, downsample).tolist()
    return spikes


@bp.get("/recording/<int:rec_id>/cell/<int:cell_id>/spikes")
def cell_spikes_(rec_id, cell_id):
    downsample = flask.request.args.get(
        "downsample", default=DEFAULT_DOWNSAMPLE, type=int
    )
    if not rec_id in recs_by_id:
        return "Invalid recording id", 400
    elif not cell_id in recs_by_id[rec_id].cluster_ids:
        return "Invalid cell id", 400
    spikes = cell_spikes(rec_id, cell_id, downsample)
    res = json.dumps(spikes)
    return res


@bp.get("/recording/<int:rec_id>/stimulus")
def stimulus(rec_id):
    downsample = flask.request.args.get("downsample")
    downsample = int(downsample) if downsample else DEFAULT_DOWNSAMPLE

    if not rec_id in recs_by_id:
        return "Invalid recording id", 400
    rec = recs_by_id[rec_id]

    stimulus = mea.decompress_stimulus(
        rec.stimulus_pattern,
        rec.stimulus_events,
        rec.num_sensor_samples,
        downsample,
    )
    res = json.dumps([row.tolist() for row in stimulus])
    return res


@bp.post("/recording/<int:rec_id>/kernel-plot")
def create_kernel(rec_id: int):
    """
    Create the kernel plot for the given spike times. The request is json of
    the form:
        {
            "spikes": int[],
            "in_downsample": int,
            "out_downsample": int,
            "snippetLen": int,
            "snippetPad": int,
        }
    """
    if not rec_id in recs_by_id:
        return "Invalid recording id", 400
    req = flask.request.get_json()
    ds_conversion = req["in_downsample"] / req["out_downsample"]
    spikes = np.floor(np.array(req["spikes"]) * ds_conversion).astype(int)
    downsample = req["out_downsample"]
    snippet_len = req["snippetLen"]
    snippet_pad = req["snippetPad"]
    # Get decompressed stimulus
    stimulus = get_stimulus(rec_id, downsample)
    snippets = mea.spike_snippets(stimulus, spikes, snippet_len, snippet_pad)
    kernel = np.mean(snippets, axis=0)
    kernel_fig = retinapy.vis.kernel(
        kernel,
        t_0=(snippet_len - snippet_pad),
        bin_duration_ms=1000
        * (downsample / recs_by_id[rec_id].sensor_sample_rate),
    )
    res = json.dumps(kernel_fig, cls=plotly.utils.PlotlyJSONEncoder)
    return res


@bp.post("/recording/<int:rec_id>/merged-kernel-plot")
def merged_kernel(rec_id: int):
    """
    Create the kernel plot for the given spike times. The request is json of
    the form:
        {
            "spikes": int[][],
            "downsample": int,
            "snippetLen": int,
            "snippetPad": int,
        }
    """
    if not rec_id in recs_by_id:
        return "Invalid recording id", 400
    req = flask.request.get_json()
    spikes = list(req["spikes"])
    downsample = req["downsample"]
    snippet_len = req["snippetLen"]
    snippet_pad = req["snippetPad"]
    # Get decompressed stimulus
    stimulus = get_stimulus(rec_id, downsample)
    snippets = mea.spike_snippets(
        stimulus, np.concatenate(spikes), snippet_len, snippet_pad
    )
    # split apart the snippets into their original groups.
    snippets = np.split(snippets, np.cumsum([len(s) for s in spikes])[:-1])
    kernels = [np.mean(s, axis=0) for s in snippets]
    ave_kernel = np.mean(kernels, axis=0)
    kernel_fig = retinapy.vis.kernel(
        ave_kernel,
        t_0=(snippet_len - snippet_pad),
        bin_duration_ms=1000
        * (downsample / recs_by_id[rec_id].sensor_sample_rate),
    )
    res = json.dumps(kernel_fig, cls=plotly.utils.PlotlyJSONEncoder)
    return res
