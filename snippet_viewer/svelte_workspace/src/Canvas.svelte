<script lang="ts">
	import * as PIXI from "pixi.js";
	import { Viewport } from "pixi-viewport";
	import { onMount } from "svelte";
	import SnippetsContextMenu from "./SnippetsContextMenu.svelte";
	import {
		width,
		height,
		canvas as canvasStore,
		context as contextStore,
		pixelRatio,
		startEngine,
		props,
		playbackTime,
		updateClock,
		Workspace,
	} from "./engine.js";

	export let workspace_id = null;
	let canvas: HTMLCanvasElement;
	let pixiApp: PIXI.Application;
	let viewport: Viewport;
	let workspace: Workspace; 

	onMount(() => {
		// Disable context menu.
		canvas.oncontextmenu = (e) => e.preventDefault();

		pixiApp = new PIXI.Application({
			view: canvas,
			// Need to manually resize, so that we can resize the viewport also, however
			// this next line is still needed to get the initial size correct.
			resizeTo: canvas,
			backgroundColor: 0x010101,
			/*antialias: true,*/
		});

		viewport = new Viewport({
			//worldWidth: 5000,
			//worldHeight: 5000,
			passiveWheel: false, // If true, then preventDefault() won't be called.
			interaction: pixiApp.renderer.plugins.interaction,
		});
		//pixiApp.stage
		viewport
			.drag({
				// These are AND requirements, unfortunately. So let's just
				// stick with middle button only.
				// keyToPress: ["Space"],
				mouseButtons: "middle",
				clampWheel: true,
			})
			.wheel();
		// Disable middle drag.
		canvas.addEventListener("pointerdown", (event) => {
			// Disable middle mouse drag.
			if (event.button == 1) {
				event.preventDefault();
			}
		});
		// Disable space scroll.
		window.addEventListener("keydown", (event) => {
			if (event.code == "Space" && event.target == document.body) {
				event.preventDefault();
			}
		});
		border(viewport);
		// If hitArea isn't restricted to the screen, then magically the app is
		// able to receive mouse events from outside the canvas area. This gives
		// drag&drop functionality an interesting out of bounds behaviour. Not 100%
		// sure which is the best, but I'm leaning towards restricting it to the
		// screen.
		pixiApp.stage.hitArea = pixiApp.screen;
		workspace = startEngine(workspace_id, canvas, pixiApp, viewport);
		// add a red box
		/*const sprite = viewport.addChild(new PIXI.Sprite(PIXI.Texture.WHITE))
	sprite.tint = 0xff0000
	sprite.width = sprite.height = 100
	sprite.position.set(100, 100)
	*/
	});

	function border(viewport: Viewport) {
		const line = viewport.addChild(new PIXI.Graphics());
		line.lineStyle(10, 0xff0000).drawRect(
			0,
			0,
			viewport.worldWidth,
			viewport.worldHeight
		);
	}

	function handleResize() {
		width.set(window.innerWidth);
		height.set(window.innerHeight);
		pixelRatio.set(window.devicePixelRatio);
		const w = window.innerWidth * window.devicePixelRatio;
		const h = window.innerHeight * window.devicePixelRatio;
		pixiApp.renderer.resize(w, h);
		viewport.resize(w, h);
	}
</script>

<SnippetsContextMenu />
<canvas
	bind:this={canvas}
	width={$width * $pixelRatio}
	height={$height * $pixelRatio}
	style="width: {$width}px; height: {$height}px;"
/>
<svelte:window on:resize|passive={handleResize} />
<slot />
