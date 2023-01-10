<script lang="ts">
import * as engine from './engine.js';
// Sadly, I don't know how to use the $store syntax through a namespace.
import {pauseCtrl, groupSelection, GroupSelectData} from './engine.js';
import Canvas  from './Canvas.svelte';
import tippy from 'tippy.js';
import { tick } from 'svelte';
import Plotly from 'plotly.js-dist-min';


let recordingId = 0;
let clusterIds = [];
let tippyTooltips = [];
let aveKernel = null;

function updateClusterIds() {
	// First, remove tooltips.
	//removeTooltips();
	fetch(`/api/recording/${recordingId}/cells`)
		.then((response) => response.json())
		.then( cIds => {
		cIds.sort((a,b) => a - b);
		clusterIds = cIds;
	});
}

function addToolTips() {
    tick().then(() => {
		const clusterButtons = document.querySelectorAll('.cluster_button');
		console.log("adding tippy");
		for(let i = 0; i < clusterButtons.length; i++) {
			const element = clusterButtons[i];
			const cellId = parseInt(element.innerHTML);
			// First, remove existing tooltip
			if(element._tippy) {
				element._tippy.destroy();
			}
			// from: https://atomiks.github.io/tippyjs/v6/ajax/
			const tippyTt = tippy(element, {
				onCreate(instance) {
					instance._src = null;
				},
				onShow(instance) {
					if(instance._src) {
						return;
					}
					fetch(`/api/recording/${recordingId}/cell/${cellId}/kernelplot/tooltip.png`)
					.then((response) => response.blob())
					.then((blob) => {
						const url = URL.createObjectURL(blob);
						const image = new Image();
						image.width = 150;
						image.height = 150;
						image.style.display = 'block';
						image.src = url;
						instance.setContent(image);
						instance._src = url;
					});
				}
			});
			tippyTooltips.push(tippyTt);
		}
	});
}

function clearKernelPlots() {
	const plots = document.querySelectorAll('#kernelplots div.plotly-plot');
	for(let i = 0; i < plots.length; i++) {
		const plot = plots[i];
		Plotly.purge(plot);
	}
}

function comparisonPlot(group1 : GroupSelectData, group2 : GroupSelectData, 
	divElement : HTMLElement) {
	// For the moment, just do average.
	avePlot([group1, group2], divElement);
}

function singlePlot(group : GroupSelectData, divElement : HTMLElement) {
	fetchKernel(group.spikes).then((fig) => {
		clearKernelPlots();
		Plotly.newPlot(divElement, fig.data, fig.layout);
	});
}

async function fetchKernel(spikes : number[]) {
	const fig = await fetch(`/api/recording/${recordingId}/kernel_plot`, {
		method: 'POST',
		headers: {'Content-Type': 'application/json',},
		body: JSON.stringify({
			"spikes": spikes,
			"downsample": 180,
			"snippetLen": 110,
			"snippetPad": 10,
			})
	}).then((response) => response.json())
	return fig
}

async function fetchMergedKernel(spikes : number[][]) {
	const fig = await fetch(`/api/recording/${recordingId}/merged_kernel_plot`, {
		method: 'POST',
		headers: {'Content-Type': 'application/json',},
		body: JSON.stringify({
			"spikes": spikes,
			"downsample": 180,
			"snippetLen": 110,
			"snippetPad": 10,
			})
	}).then((response) => response.json())
	return fig
}

function avePlot(groups : GroupSelectData[], divElement : HTMLElement) {
	if(groups.length == 0) {
		// Skip if so groups.
		return;
	}
	let spikes : number[][] = [];
	for(let g = 0; g < groups.length; g++) {
		spikes.push(groups[g].spikes);
	}
	if(spikes.length == 0) {
		console.log("No spikes to plot");
		return;
	}
	fetchMergedKernel(spikes).then((fig) => {
		Plotly.purge(divElement);
		Plotly.newPlot(divElement, fig.data, fig.layout);
	});
}


groupSelection.subscribe( (groups : GroupSelectData[]) => {
	// Return if no groups 
	if(groups.length === 0) {
		return;
	}
	const plotDiv = document.getElementById('kernelplots');
	console.log("plotting.")
	if(groups.length == 1) {
		singlePlot(groups[0], plotDiv);
	} else if(groups.length == 2) {
		comparisonPlot(groups[0], groups[1], plotDiv);
	} else {
		avePlot(groups, plotDiv);
	}
});

$: recordingId, updateClusterIds();
$: clusterIds, addToolTips();


let _pause = false
engine.pauseCtrl.subscribe(value => {
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
<h3>Clusters</h3>
<div class="clusters">
	{#each clusterIds as cId}
		<div class="cluster_box">
			<button
				class="cluster_button"
				on:click={() => engine.addCellAsGroup(recordingId, cId)}
				>{cId}</button
			>
		</div>
	{/each}
</div>

<div class="button">
	<button type="button" name="play-pause" on:click={() => pauseCtrl.toggle()}
		>{_pause ? "▶️" : "⏸️"}
	</button>
</div>

<!-- Make a separate component at some point -->
<div id="kernelplots" />

<Canvas />

<style>
	:global(body) {
		margin: 0;
		padding: 0;
	}

	div.cluster_box {
		display: inline-block;
		padding: 5px 4px 5px 4px;
		text-align: center;
		background: white;
		border-radius: 5px;
		margin: 2px;
	}

	div.cluster_box button {
		width: 50px;
		margin-bottom: 0;
	}

	div.clusters {
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
</style>
