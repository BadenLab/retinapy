<script lang="ts">
	import * as engine from "./engine.js";
	// Sadly, I don't know how to use the $store syntax through a namespace.
	import { pauseCtrl, groupSelection, GroupSelectData } from "./engine.js";
	import Canvas from "./Canvas.svelte";
	import tippy from "tippy.js";
	import { tick } from "svelte";
	import Plotly from "plotly.js-dist-min";

	// workspace_id comes from the template.
	let recordingId = 0;
	let cellIds = [];
	let tippyTooltips = [];
	let aveKernel = null;

	function updateCellIds() {
		// First, remove tooltips.
		//removeTooltips();
		fetch(`/api/recording/${recordingId}/cells`)
			.then((response) => response.json())
			.then((cIds) => {
				cIds.sort((a, b) => a - b);
				cellIds = cIds;
			});
	}

	function addToolTips() {
		tick().then(() => {
			const cellButtons = document.querySelectorAll(".cell_button");
			console.log("adding tippy");
			for (let i = 0; i < cellButtons.length; i++) {
				const element = cellButtons[i];
				const cellId = parseInt(element.innerHTML);
				// First, remove existing tooltip
				if (element._tippy) {
					element._tippy.destroy();
				}
				// from: https://atomiks.github.io/tippyjs/v6/ajax/
				const tippyTt = tippy(element, {
					onCreate(instance) {
						instance._src = null;
					},
					onShow(instance) {
						if (instance._src) {
							return;
						}
						fetch(
							`/api/recording/${recordingId}/cell/${cellId}/kernelplot/tooltip.png`
						)
							.then((response) => response.blob())
							.then((blob) => {
								const url = URL.createObjectURL(blob);
								const image = new Image();
								image.width = 150;
								image.height = 150;
								image.style.display = "block";
								image.src = url;
								instance.setContent(image);
								instance._src = url;
							});
					},
				});
				tippyTooltips.push(tippyTt);
			}
		});
	}

	function clearKernelPlots() {
		const plots = document.querySelectorAll("#kernelplots div.plotly-plot");
		for (let i = 0; i < plots.length; i++) {
			const plot = plots[i];
			Plotly.purge(plot);
		}
	}

	function singlePlot(
		spikes: number[],
		recId: number,
		divElement: HTMLElement
	) {
		fetchKernel(spikes, recId).then((fig) => {
			clearKernelPlots();
			fig.layout.title.text = "Snippet average";
			Plotly.newPlot(divElement, fig.data, fig.layout);
		});
	}

	async function fetchKernel(spikes: number[], recId: number) {
		const fig = await fetch(`/api/recording/${recId}/kernel-plot`, {
			method: "POST",
			headers: { "Content-Type": "application/json" },
			body: JSON.stringify({
				spikes: spikes,
				in_downsample: 180,
				out_downsample: 18,
				snippetLen: 1100,
				snippetPad: 100,
			}),
		}).then((response) => response.json());
		return fig;
	}

	async function fetchMergedKernel(spikes: number[][]) {
		const fig = await fetch(
			`/api/recording/${recordingId}/merged-kernel-plot`,
			{
				method: "POST",
				headers: { "Content-Type": "application/json" },
				body: JSON.stringify({
					spikes: spikes,
					downsample: 180,
					snippetLen: 110,
					snippetPad: 10,
				}),
			}
		).then((response) => response.json());
		return fig;
	}

	function avePlot(groups: GroupSelectData[], divElement: HTMLElement) {
		if (groups.length == 0) {
			// Skip if so groups.
			return;
		}
		let spikes: number[][] = [];
		for (let g = 0; g < groups.length; g++) {
			spikes.push(groups[g].spikes);
		}
		if (spikes.length == 0) {
			console.log("No spikes to plot");
			return;
		}
		fetchMergedKernel(spikes).then((fig) => {
			fig.layout.title.text = "Snippet average";
			Plotly.purge(divElement);
			Plotly.newPlot(divElement, fig.data, fig.layout);
		});
	}

	groupSelection.subscribe((groups: GroupSelectData[]) => {
		// Return if no groups
		if (groups.length === 0) {
			return;
		}
		const plotDiv = document.getElementById("kernelplots");
		console.log("plotting.");
		const spikeArrs = [];
		for (let g = 0; g < groups.length; g++) {
			spikeArrs.push(groups[g].spikes);
		}
		let spikes = [];
		spikes = spikes.concat(...spikeArrs);
		// We will assume all spikes are from the same recording.
		// TODO: need to make this assumption explicit.

		// Try find a non empty group, which will have a recording id.
		let recId = null;
		for (let g of groups) {
			if(g.recId != null) {
				recId = g.recId;
				break;
			}
		}
		//recId = recordingId;
		singlePlot(spikes, recId, plotDiv);
	});

	$: recordingId, updateCellIds();
	$: cellIds, addToolTips();

	let _pause = false;
	engine.pauseCtrl.subscribe((value) => {
		_pause = value;
	});
</script>

<ul>
	<select bind:value={recordingId}>
		{#await engine.recordings()}
			<p>Loading...</p>
		{:then recordings}
			{#each recordings as rec}
				<option value={rec.id}>
					{rec.name}
				</option>
			{/each}
		{/await}
	</select>
</ul>
<h3>Cells</h3>
<div class="cells">
	{#each cellIds as cId}
		<div class="cell_box">
			<button
				class="cell_button"
				on:click={() => engine.addGroupFromCell(recordingId, cId)}
				>{cId}</button
			>
		</div>
	{/each}
</div>

<!-- Make a separate component at some point -->
<div id="kernelplots" />

<div class="button">
	<button type="button" name="play-pause" on:click={() => pauseCtrl.toggle()}
		>{_pause ? "▶️" : "⏸️"}
	</button>
</div>

<Canvas workspace_id={window.workspace_id} />

<style>
	:global(body) {
		margin: 0;
		padding: 0;
	}

	div.cell_box {
		display: inline-block;
		padding: 5px 4px 5px 4px;
		text-align: center;
		background: white;
		border-radius: 5px;
		margin: 2px;
	}

	div.cell_box button {
		width: 50px;
		margin-bottom: 0;
	}

	div.cells {
		margin-bottom: 1rem;
	}

	div.button {
		display: flex;
		vertical-text-align: center;
		display: flex;
		align-items: center;
		justify-content: center;
		margin-bottom: 1rem;
	}
	div.button > input {
		margin: 1rem;
	}

	#kernelplots {
		margin: auto;
		padding: 2rem 1rem 2rem 1rem;
		width: 85%;
		min-height: 250px;
	}

	#kernelplots div.halfwidth {
		width: 50%;
		display: inline-block;
	}

	/* Add horizontal padding between radio buttons */
	.cell-options input {
		margin-left: 1rem;
	}
</style>
