<script>
	import { width, height, snippets } from './game.js';

	import Canvas from './Canvas.svelte';
	import Background from './Background.svelte';
	import DotGrid from './DotGrid.svelte';
	import Character from './Character.svelte';
	import Text from './Text.svelte';
	import FPS from './FPS.svelte';
	import Snippets from './Snippets.svelte';

const DATA_PATH = './data/';
const CLUSTER_ID_FILE = DATA_PATH + 'cluster_ids.json';

async function get_cluster_ids() {
	const res = await fetch(CLUSTER_ID_FILE);
	return res.json().then( cluster_ids => {
		cluster_ids.sort((a,b) => a - b);
		return cluster_ids;
	});
}

let debug_data = [];

async function load_cluster(idx) {
	const res = await fetch(DATA_PATH + idx + '.json');
	let s = await res.json();
	snippets.set(s)
	debug_data = s
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
<div class="cluster_box">
<button on:click={() => load_cluster(c_id)}>{c_id}</button>
</div>
{/each}
{/await}
</div>

<Canvas>
	<Background color='hsl(0, 0%, 10%)'>
		<!--<DotGrid divisions={40} color='hsla(0, 0%, 100%, 0.5)' />-->
	</Background>
	<Snippets  />
	<!--
	<Text
		text='Click and drag around the page to move the character.'
		fontSize={12}
		align='right'
		baseline='bottom'
		x={$width - 20}
		y={$height - 20} />
	<FPS />
	<Character size={10} />-->
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
		padding: 8px 5px 8px 5px;
		text-align: center;
		background: gray;
		margin: 2px;
	}

	div.cluster_box button {
		width: 50px;
		margin-bottom: 0;
	}
</style>
