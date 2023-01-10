<script>
	import * as engine from "./engine.js";
	// Sadly, I don't know how to use the $store syntax through a namespace.
	import { pause_ctrl } from "./engine.js";

	import Canvas from "./Canvas.svelte";
	import Background from "./Background.svelte";
	import Snippets from "./Snippets.svelte";
	import Timeline from "./Timeline.svelte";

	let num_x_boxes = 30;
	let recording_id = 0;
	let cluster_ids = [];
	let cluster_id = 0;

	function get_rec_names() {
		const res = fetch("/api/recordings").then((response) =>
			response.json()
		);
		return res;
	}

	function update_cluster_ids() {
		fetch(`/api/recording/${recording_id}/cells`)
			.then((response) => response.json())
			.then((c_ids) => {
				c_ids.sort((a, b) => a - b);
				cluster_ids = c_ids;
			});
	}

	$: recording_id, update_cluster_ids();

	async function load_cluster(idx) {
		const res = await fetch(`/api/recording/${recording_id}/cell/${idx}`);
		let s = await res.json();
		engine.set_snippets(s);
		cluster_id = idx;
		return s;
	}

	let _pause = false;
	engine.pause_ctrl.subscribe((value) => {
		_pause = value;
	});
</script>

<h2>MEA snippets</h2>
<p>Snippets for each cluster.</p>
<h3>Recordings</h3>
<ul>
	{#await get_rec_names()}
		<p>Loading...</p>
	{:then rec_id_names}
		<select bind:value={recording_id}>
			{#each rec_id_names as rec}
				<option value={rec.id}>
					{rec.name}
				</option>
			{/each}
		</select>
	{/await}
</ul>
<h3>Clusters</h3>
<div class="clusters">
	{#each cluster_ids as c_id}
		<div class={c_id == cluster_id ? "current cluster_box" : "cluster_box"}>
			<button on:click={() => load_cluster(c_id)}>{c_id}</button>
		</div>
	{/each}
</div>

<div>
	<label
		>Horizontal boxes:
		<input
			type="number"
			name="hbox-count"
			bind:value={num_x_boxes}
			min="20"
			max="200"
		/>
		<input
			type="range"
			name="hbox-count-range"
			bind:value={num_x_boxes}
			min="20"
			max="100"
		/>
		<button
			type="button"
			name="play-pause"
			on:click={() => pause_ctrl.toggle()}
			>{_pause ? "▶️" : "⏸️"}
		</button>
	</label>
</div>

<Canvas>
	<Background color="hsl(0, 0%, 10%)">
		<!--<DotGrid divisions={40} color='hsla(0, 0%, 100%, 0.5)' />-->
	</Background>
	<!--<FPS />-->
	<Timeline />
	<Snippets {num_x_boxes} />
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
