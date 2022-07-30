<script>
	import * as engine from './engine.js';

	import Canvas from './Canvas.svelte';
	import Background from './Background.svelte';
	import DotGrid from './DotGrid.svelte';
	import Character from './Character.svelte';
	import Text from './Text.svelte';
	import FPS from './FPS.svelte';
	import Snippets from './Snippets.svelte';
	import Timeline from './Timeline.svelte';

const DATA_PATH = './data/';
const CLUSTER_ID_FILE = DATA_PATH + 'cluster_ids.json';

async function get_cluster_ids() {
	const res = await fetch(CLUSTER_ID_FILE);
	return res.json().then( cluster_ids => {
		cluster_ids.sort((a,b) => a - b);
		return cluster_ids;
	});
}

let cluster_id = 0;

async function load_cluster(idx) {
	const res = await fetch(DATA_PATH + idx + '.json');
	let s = await res.json();
	engine.set_snippets(s);
	cluster_id = idx;
	return s;
}

</script>

<h2>MEA snippets</h2>
<p>Snippets for each cluster.</p>
<h3>Clusters</h3>
<div class="clusters">
{#await get_cluster_ids()}
<p>Loading...</p>
{:then cluster_ids}
{#each cluster_ids as c_id}
<div class="{c_id == cluster_id ? 'current cluster_box' : 'cluster_box'}">
<button on:click={() => load_cluster(c_id)}>{c_id}</button>
</div>
{/each}
{/await}
</div>

<Canvas>
	<Background color='hsl(0, 0%, 10%)'>
		<!--<DotGrid divisions={40} color='hsla(0, 0%, 100%, 0.5)' />-->
	</Background>
	<!--<FPS />-->
	<Timeline />
	<Snippets  />
</Canvas>


<!--<div class="snippets">
<p style='font-size:5px'>{debug_data}</p>
</div>-->


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

	div.cluster_box.current {
		background: grey;
	}

	div.cluster_box button {
		width: 50px;
		margin-bottom: 0;
	}

	div.clusters {
		margin-bottom: 1rem;
	}

</style>
