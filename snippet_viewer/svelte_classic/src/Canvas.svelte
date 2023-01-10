<script>
import { onMount, onDestroy, setContext } from 'svelte';

import {
	key,
	width,
	height,
	canvas as canvasStore,
	context as contextStore,
	pixelRatio,
	props,
	update_clock,
} from './engine.js';

export let killLoopOnError = true;
export let attributes = {};

let listeners = [];
let canvas;
let context;
let frame;

onMount(() => {
	// prepare canvas stores
	context = canvas.getContext('2d', attributes);
	canvasStore.set(canvas);
	contextStore.set(context);

	// setup entities
	listeners.forEach(async entity => {
		if (entity.setup) {
			let p = entity.setup($props);
			if (p && p.then) await p;
		}
		entity.ready = true;
	});
	
	// start game loop
	return createLoop(render)
});

setContext(key, {
	add (fn) {
		this.remove(fn);
		listeners.push(fn);
	},
	remove (fn) {
		const idx = listeners.indexOf(fn);
		if (idx >= 0) listeners.splice(idx, 1);
	}
});

function handleResize () {
	width.set(window.innerWidth);
	height.set(window.innerHeight);
	pixelRatio.set(window.devicePixelRatio);
}

/**
 * Called by `loop()`, which is run repeatedly in `createLoop()`.
 */
function render (elapsed, dt) {
	update_clock(elapsed, dt);
	context.save();
	context.scale($pixelRatio, $pixelRatio);
	listeners.forEach(entity => {
		try {
			if (entity.mounted && entity.ready && entity.render) {
				entity.render($props, dt);
			}
		} catch (err) {
			console.error(err);
			if (killLoopOnError) {
				cancelAnimationFrame(frame);
				console.warn('Animation loop stopped due to an error');
			}
		}
	});
	context.restore();
}

/**
 * Typically called as `createLoop(render)`. 
 *
 * The equivalent two.js function: 
 *
 * 		https://github.com/jonobr1/two.js/blob/ea7491d0b2741dde4f62f5fedf035910368ac433/src/two.js#L1141
 */
function createLoop (fn) {
	let elapsed = 0;
	let lastTime = performance.now();
	(function loop() {
		frame = requestAnimationFrame(loop);
		const beginTime = performance.now();
		const dt = (beginTime - lastTime) / 1000;
		lastTime = beginTime;
		elapsed += dt;
		fn(elapsed, dt);
	})();
	return () => {
		cancelAnimationFrame(frame);
	};
}


</script>

<canvas
	bind:this={canvas}
	width={$width * $pixelRatio}
	height={$height * $pixelRatio}
	style="width: {$width}px; height: {$height}px;"
/>
<svelte:window on:resize|passive={handleResize} />
<slot></slot>

