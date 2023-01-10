<script lang="ts">
	import * as engine from './engine.js';
	// Sadly, I don't know how to use the $store syntax through a namespace.
	import {pauseCtrl} from './engine.js';

	import Canvas from './Canvas.svelte';
	import { onMount } from 'svelte';


let num_x_boxes = 30;
let recording_id = 0;
let cluster_ids = [];

 
function update_cluster_ids() {
	fetch(
	`/api/recording/${recording_id}/cells`)
		.then((response) => response.json())
		.then( c_ids => {
		c_ids.sort((a,b) => a - b);
		cluster_ids = c_ids;
	});
}

$: recording_id, update_cluster_ids();


let _pause = false
engine.pauseCtrl.subscribe(value => {
	_pause = value;	
});

</script>

<ul>
<select bind:value={recording_id}>
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
{#each cluster_ids as c_id}
<div class="cluster_box">
<button on:click={() => engine.addCellAsGroup(recording_id, c_id)}
data-tip
>{c_id}</button>
</div>
{/each}
</div>

<div>
<label>
<button type="button" name="play-pause" on:click="{() => pauseCtrl.toggle()}">{_pause ? '▶️' : '⏸️'} </button>
</label>
</div>


<Canvas></Canvas>


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
		max-height: 400px;
		overflow-y: scroll;
	}

	div.cluster_box button {
		width: 50px;
		margin-bottom: 0;
	}

	div.clusters {
		margin-bottom: 1rem;
	}

	div > label {
		display: flex;
		vertical-text-align: center;
		display: flex;
		align-items: center;
		justify-content: center;
		margin-bottom: 1rem;
	}
	div > label > input {
		margin: 1rem;
	}

</style>
