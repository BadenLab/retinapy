from typing import Optional

import numpy as np

import plotly
import plotly.graph_objects as go
import plotly.subplots
import retinapy
import retinapy.dataset
import retinapy.mea as mea
import retinapy.spikeprediction as sp
import torch

"""
This module is starting out as somewhere to create the most commonly figures 
needed in notebooks. Going forward, it's hoped that some of the plots or 
images created here can be recorded automatically when training a model.
"""


def create_title(title: str, subtitle: Optional[str]) -> str:
    """
    Returns a styled title string for use in plotly.

    Subtitle is optional.
    """
    if subtitle is None:
        res = f"{title}"
    else:
        res = f'{title}<br><span style="font-size:90%; color:grey">{subtitle}</span>'
    return res


def create_axis_title(title: str, units: str) -> str:
    """Returns a styled title string for use in plotly."""
    return f"{title}<span style='font-size:90%;white-space:pre;color:grey'>  ({units})</span>"


def default_fig_layout():
    """Some default fig options.

    For quite a few of these, like margin, I don't quite understand their
    full implications. This is part of the reason it's useful to move them
    here, as they will likely change over time.
    """
    res = {
        "title_x": 0.5,
        "title_pad": dict(l=0, r=0, b=0, t=0),
        # The title and xaxis label fit within the margin, so it needs
        # to be big enough to fit them. Top margin of 50 or so works for a
        # title with no subtitle. 70 works for a title with a subtitle.
        # 50 is okay for bottom, but it's a little too tight in Jupyter Lab,
        # so making it a tad bigger (60).
        "margin": {"l": 0, "r": 0, "t": 50, "b": 60, "pad": 0},
    }
    return res


def kernel(
    kernel: np.ndarray,
    t_0: int,
    bin_duration_ms,
    vline: Optional[int] = None,
    title=None,
    subtitle=None,
):
    """
    Args:
        kernel: The kernel to plot. This is a 2D array with shape
            (n_channels, n_timesteps). There can be 1-4 channels.
        t_0: the index of the bin that corresponds to t=0.
    """
    fig = go.Figure()
    xs = (np.arange(kernel.shape[0]) - t_0) * bin_duration_ms
    if vline is not None:
        fig.add_vline(
            x=-100,
            line_width=2,
            line_dash="dot",
            line_color="grey",
            annotation_text="-100ms",
            annotation_position="bottom right",
        )
    for idx, stim in enumerate(mea.stimuli):
        fig.add_trace(
            go.Scatter(
                x=xs,
                y=kernel[idx],
                name=stim.name,
                mode="lines",
                line=dict(width=2),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=xs,
                y=kernel[:, idx],
                line_color=stim.display_hex,
                mode="lines",
            )
        )
    fig.update_layout(default_fig_layout())
    fig.update_layout(
        margin=dict(l=1, r=1, b=1, t=25, pad=10),
        yaxis_fixedrange=True,
        showlegend=False,
        title="Kernel",
        xaxis={"title": "time (ms), with spike at 0"},
        yaxis={"title": "Stimulus", "range": [0, 1]},
    )
    return fig


def stimulus_fig(
    stimulus: np.ndarray, start_ms: float = 0, bin_duration_ms: float = 1.0
):
    """
    A figure to display a stimulus.

    Args:
        stimulus: the 4 color stimulus, with shape (n_channels=4, n_timesteps)
    """
    fig = plotly.subplots.make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        row_width=[0.2, 0.8],
        x_title=create_axis_title("time", "ms"),
    )
    xs = np.arange(stimulus.shape[1]) * bin_duration_ms + start_ms
    # RGBU
    for idx, stim in enumerate(mea.stimuli):
        fig.append_trace(
            go.Scatter(
                x=xs,
                y=stimulus[idx, :],
                line_color=stim.display_hex,
                name=f"{stim.wavelength} nm",
                mode="lines",
            ),
            row=1,
            col=1,
        )
    # sum
    fig.append_trace(
        go.Scatter(
            x=xs,
            y=stimulus.sum(axis=0),
            line_color="black",
            name="sum",
            mode="lines",
        ),
        row=2,
        col=1,
    )
    fig.update_yaxes(
        {"tickmode": "array", "tickvals": [-5, 0, 5]}, row=2, col=1
    )
    # Set a default layout.
    fig.update_layout(default_fig_layout())
    fig.update_layout(
        {
            "title": {"text": "Stimulus"},
            # The title and xaxis label fit within the margin, so it needs
            # to be big enough to fit them.
            "yaxis": {
                "title": {"text": create_axis_title("stimulus", "no units")},
                "fixedrange": True,
                # The range is dependent on the agumentation.
                "range": [-2, 2],
                "tickmode": "array",
                "tickvals": [-1, 0, 1],
            },
        }
    )
    return fig


def distfield_fig(
    actual: np.ndarray,
    pred: np.ndarray,
    start_ms: float = 0,
    bin_duration_ms: float = 1.0,
    stride_bins: int = 0,
    log_space: bool = False,
):
    """
    A figure containing 1 or multiple line charts comparing distance fields.

    Args:
        stride: the jump in the x-axis origin label between each subplot. This
            is used to plot multiple times where each subplot starts say
            50 ms after the previous one.
    """
    if actual.shape != pred.shape:
        raise ValueError(
            "Both distfield inputs must have the same shape. Got "
            f"{actual.shape} and {pred.shape}."
        )
    # Expand dims if only one distfield is given (single or multiple distfields
    # are accepted).
    if len(actual.shape) == 1:
        actual = actual.reshape(1, -1)
        pred = pred.reshape(1, -1)
    num_rows = len(actual)
    # Initialize the figure, in its subplot glory.
    fig = plotly.subplots.make_subplots(
        rows=num_rows,
        cols=1,
        shared_xaxes=False,
        # Default vertical spacing is 0.3/num_rows
        vertical_spacing=0.5 / num_rows,
        # Make the plots have the same y-range.
        shared_yaxes="columns",
        x_title=create_axis_title("time", "ms"),
        y_title=(
            create_axis_title("log time to nearest spike", "log(ms)")
            if log_space
            else create_axis_title("time to nearest spike", "ms")
        ),
    )
    # Create the scatters.
    stride_ms = stride_bins * bin_duration_ms
    num_bins = actual.shape[1]
    for r in range(num_rows):
        start = start_ms + r * stride_ms
        xs = np.arange(num_bins) * bin_duration_ms + start
        showlegend = r == 0
        scatter_actual = go.Scatter(
            x=xs,
            y=actual[r, :],
            name="actual",
            mode="lines",
            line_color="tomato",
            showlegend=showlegend,
            legendgroup="actual",
        )
        scatter_pred = go.Scatter(
            x=xs,
            y=pred[r, :],
            name="pred",
            mode="lines",
            line_color="gray",
            showlegend=showlegend,
            legendgroup="pred",
        )
        fig.add_trace(scatter_actual, row=r + 1, col=1)
        fig.add_trace(scatter_pred, row=r + 1, col=1)
    # Determine the y-axis scale. These are defaults, free to be changed.
    if log_space:
        yaxis_range = [-4, 4]
    else:
        yaxis_range = [-1, 600]
    # Create a title.
    total_len = actual.shape[1] * bin_duration_ms
    title_start = "Log distance fields" if log_space else "Distance fields "
    stride_str = (
        f"strided by {(stride_bins)} bins ({stride_ms:.1f} ms)"
        if stride_bins > 0
        else "no stride specified"
    )

    title_str = create_title(
        f"{title_start}, actual and predicted",
        f"{num_rows} snippets, "
        f"{num_bins} bins x {bin_duration_ms:.1f} ms/bin "
        f"= {total_len:.1f} ms each, {stride_str} ",
    )

    height = 140 * num_rows
    fig.update_layout(default_fig_layout())
    fig.update_layout(
        {
            "height": height,
            "yaxis": {
                "range": yaxis_range,
                "fixedrange": True,
            },
            "title": {"text": title_str},
            # Need left margin to fit the shared y-axis title, and some extra
            # on top for the subtitle.
            "margin": {"l": 70, "t": 70},
        }
    )
    return fig


def distfield_model_in_out(
    stimulus: np.ndarray,
    in_spikes: np.ndarray,
    target_dist: np.ndarray,
    model_out: np.ndarray,
    start_ms=0,
    bin_duration_ms=1.0,
    cluster_label=None,
):
    """A figure to view the inputs and outputs of a distfield model.

    Create a figure that has the stimulus on top, and the input spikes and
    the pair of distfields on the bottom (actual, pred). Roughly formatted
    like so:

    +--------------------+
    |      stimulus      |
    +--------------------+
    | spikes      | dist |
    +--------------------+

    """
    # The figure will be built in 4 steps:
    #   1. The 4-channel stimuli that spans the whole x-axis.
    #   2. The sum of the 4-channel stimuli (in black). This line is plotted
    #      below the 4-channel stimuli and also spans the whole x-axis.
    #   3. The model output distance field and the true (target) distance field.
    #      These lines occupy a small time-slice on the right, typically around
    #      50-200 ms depending on the model.
    #   4. The input spike data, shown as vertical lines on the same sub-plot
    #      as the distance fields.

    # It would be nice to reuse the functionality of the above two functions;
    # however, there are just enough differences for it to be not so easy.

    fig = plotly.subplots.make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        row_width=[0.4, 0.15, 0.45],
        x_title=create_axis_title("time", "ms"),
    )

    # 1. The 4-channel stimulus data.
    stim_len = stimulus.shape[1]
    xs = np.arange(stim_len) * bin_duration_ms + start_ms
    # RGBU
    for idx, stim in enumerate(mea.stimuli):
        fig.append_trace(
            go.Scatter(
                x=xs,
                y=stimulus[idx, :],
                line_color=stim.display_hex,
                name=f"{stim.wavelength} nm",
                mode="lines",
            ),
            row=1,
            col=1,
        )

    # 2. The summed stimulus data.
    fig.append_trace(
        go.Scatter(
            x=xs,
            y=stimulus.sum(axis=0),
            line_color="black",
            name="sum",
            mode="lines",
        ),
        row=2,
        col=1,
    )
    fig.update_yaxes(
        {"tickmode": "array", "tickvals": [-5, 0, 5], "fixedrange": True},
        row=2,
        col=1,
    )

    # 3. The model output distance field and the target (actual).
    # Quite a dependency (on DistFieldDataset) creeping in here! But we need
    # to know what mask value the dataset is using. So it is necessary. I just
    # find it interesting to note the dependency. Can there be a project wide
    # mask value? Probably not a useful idea, is my guess.
    mask_val = retinapy.dataset.DistFieldDataset.MASK_VALUE
    mask_start_idx = np.min(np.flatnonzero(in_spikes == mask_val))
    xs_dist = xs[mask_start_idx:]
    if len(xs_dist) != len(model_out):
        raise ValueError(
            "The spike mask isn't the same lengths as the model output. "
            f"Got ({len(xs_dist)}) and ({len(model_out)})"
        )
    if mask_start_idx + len(model_out) != stim_len:
        raise ValueError(
            "The stimulus length doesn't allign with the model " "output."
        )
    fig.append_trace(
        go.Scatter(
            x=xs_dist,
            y=model_out,
            name="pred",
            mode="lines",
            line_color="gray",
        ),
        col=1,
        row=3,
    )
    fig.append_trace(
        go.Scatter(
            x=xs_dist,
            y=target_dist,
            name="actual",
            mode="lines",
            line_color="tomato",
        ),
        col=1,
        row=3,
    )
    fig.update_yaxes({"range": [-4, 4]}, row=3, col=1)
    # Cover the whole output region with a vrect to separate the regions.
    fig.add_vrect(
        x0=xs[mask_start_idx],
        x1=xs[-1],
        fillcolor="aqua",
        opacity=0.25,
        col=1,
        row=3,
        line_width=0,
        layer="below",
    )

    # 4. The vertical lines marking the input spikes.
    index_of_spikes = np.flatnonzero(in_spikes > 0)
    for idx in index_of_spikes:
        spike_loc = start_ms + idx * bin_duration_ms
        fig.add_vline(
            x=spike_loc, line_color="tomato", line_dash="dot", row=3, col=1
        )

    # Set a default layout.
    fig.update_layout(default_fig_layout())
    # Create title
    main_title_str = "Distfield model input-output"
    if cluster_label is not None:
        main_title_str += f" ({cluster_label})"
    title = create_title(
        main_title_str,
        "Output region is highlight blue. Spikes are red v-lines.",
    )
    fig.update_layout(
        {
            "title": {"text": title},
            "margin": {"t": 70},
            # The title and xaxis label fit within the margin, so it needs
            # to be big enough to fit them.
            "yaxis": {
                "title": {"text": create_axis_title("stimulus", "no units")},
                "fixedrange": True,
                # The range is dependent on the agumentation.
                "range": [-2, 2],
                "tickmode": "array",
                "tickvals": [-1, 0, 1],
            },
            "xaxis": {
                "range": [0, stim_len],
            },
        }
    )
    return fig


def latent_fig(trainable):
    # 1. Gather the data to plot. We will actually do that here, so this is
    # quite a proactive plotting function.
    rec_idxs = [0]  # Only supports 1 recording for now.
    cluster_idxs = []
    cluster_ids = []
    with torch.no_grad():
        for r_idx in rec_idxs:
            rec = trainable.train_ds.datasets[r_idx].recording
            cluster_idxs = torch.arange(len(rec.cluster_ids))
            cluster_idxs.append(np.array(cluster_idxs))
            rec_idxs.append(torch.full_like(cluster_idxs[-1], r_idx))
            # For labels, use the cell/cluster-id given by the spike sorter.
            cluster_ids.extend([rec.cluster_ids[i] for i in cluster_idxs])
        rec_idxs = torch.concat(rec_idxs)
        cluster_idxs = torch.concat(cluster_idxs)
        zs = trainable.encode(rec_idxs, cluster_idxs).cpu().numpy()
    # 2. Make the figure
    fig = go.Figure()
    assert zs is not None
    xs = zs[:, 0]
    y = zs[:, 1]
    labels = [
        f"({r_idx}, {c_id})" for (r_idx, c_id) in zip(rec_idxs, cluster_ids)
    ]
    scatter = go.Scatter(
        x=xs,
        y=y,
        text=labels,
        textposition="bottom center",
        mode="markers+text",
    )
    fig.add_trace(scatter)
    #fig.update_layout(
    #    xaxis={"range": [-3, 3]},
    #    yaxis={"range": [-3, 3]},
    #)
    return fig
