<script lang="ts">
	import { afterUpdate } from "svelte";
	import { selectionMode } from "./engine.js";
	import * as engine from "./engine.js";
	import type { SelectionManager, Snippet, Group } from "./engine.js";
	import type { InteractionEvent } from "pixi.js";

	const pad = 5;

	let _x: number = 0;
	let _y: number = 0;
	let isMenuOpen = false;
	let isSimilarOptionEnabled = false;
	let similarColor = "white";

	let contextMenuElement: HTMLElement;

	function getSelectionMgr(): SelectionManager<Snippet> {
		const selectionManager = engine.workspace.snippetSelectionMgr;
		if (selectionManager === undefined || selectionManager === null) {
			throw Error(`Snippet selection manager is ${selectionManager}.`);
		}
		return selectionManager;
	}

	// whenever x and y is changed, restrict box to be within bounds
	function fitToScreen() {
		if (!contextMenuElement) return;
		const rect = contextMenuElement.getBoundingClientRect();
		// _x and _y are page coordinates, while window.innerWidth and
		// window.innerHeight are viewport dimensions. Instead of checking for
		// overflow using _x and _y, we need to use rect.x and rect.y from
		// getBoundingClientRect(), which is relative to the viewport.
		const overflowX = rect.x + rect.width + pad - window.innerWidth;
		const overflowY = rect.y + rect.height + pad - window.innerHeight;
		if (overflowX > 0) {
			_x -= overflowX;
		}
		if (overflowY > 0) {
			_y -= overflowY;
		}
	}
	afterUpdate(fitToScreen);

	selectionMode.subscribe((mode) => {
		if (mode === "snippet") {
			getSelectionMgr().onContextMenuShow(
				(event: InteractionEvent, snippetSelection: Snippet[]) => {
					// With the current features, we only care about the
					// first snippet in the selection.
					const x = event.data.originalEvent.pageX;
					const y = event.data.originalEvent.pageY;
					//const x = event.data.originalEvent.screenX;
					//const y = event.data.originalEvent.screenY;
					isSimilarOptionEnabled =
						snippetSelection != null &&
						snippetSelection.length === 1;
					if (isSimilarOptionEnabled) {
						const rgb = snippetSelection[0]._colorRGB();
						similarColor = `rgb(${255 * rgb[0]}, ${255 * rgb[1]}, ${
							255 * rgb[2]
						})`;
					}
					console.log("in the context menu!!");
					console.log(`x: ${x}, y: ${y}`);
					console.log(event);
					showMenu(x, y, snippetSelection);
					const destroy = () => {
						closeMenu();
					};
					return destroy;
				}
			);
		}
	});

	function showMenu(x: number, y: number, _selectedSnippets: Snippet[]) {
		_x = x;
		_y = y;
		isMenuOpen = true;
	}

	function selectByLED(ledIdx: number, exclusive: boolean) {
		engine.workspace.selectSnippetsByOnLED(ledIdx, exclusive);
		closeMenu();
	}

	function selectFirst(duration_secs: number) {
		engine.workspace.selectSnippetsAfterQuiet(duration_secs);
		closeMenu();
	}

	function selectSimilar() {
		if (!isSimilarOptionEnabled) {
			return;
		}
		engine.workspace.selectSnippetsSimimlarToSelected();
		closeMenu();
	}

	function closeMenu() {
		isMenuOpen = false;
	}
</script>

{#if isMenuOpen}
	<menu
		class="context-menu"
		bind:this={contextMenuElement}
		on:contextmenu={(e) => e.preventDefault()}
		style="top: {_y}px; left: {_x}px;"
	>
		<li>
			<button
				class="context-menu-item"
				on:click={(e) => selectByLED(0, false)}
			>
				<div class="context-menu-label">
					<span class="r-led">610 nm</span> on
				</div>
				<kbd class="context-menu-shortcut">r</kbd>
			</button>
		</li>
		<li>
			<button
				class="context-menu-item"
				on:click={(e) => selectByLED(1, false)}
			>
				<div class="context-menu-label">
					<span class="g-led">560 nm</span> on
				</div>
				<kbd class="context-menu-shortcut">g</kbd>
			</button>
		</li>
		<li>
			<button
				class="context-menu-item"
				on:click={(e) => selectByLED(2, false)}
			>
				<div class="context-menu-label">
					<span class="b-led">460 nm</span> on
				</div>
				<kbd class="context-menu-shortcut">b</kbd>
			</button>
		</li>
		<li>
			<button
				class="context-menu-item"
				on:click={(e) => selectByLED(3, false)}
			>
				<div class="context-menu-label">
					<span class="uv-led">413 nm</span> on
				</div>
				<kbd class="context-menu-shortcut">u</kbd>
			</button>
		</li>
		<li>
			<button
				class="context-menu-item"
				on:click={(e) => selectByLED(0, true)}
			>
				<div class="context-menu-label">
					Only <span class="r-led">610 nm</span> on
				</div>
				<kbd class="context-menu-shortcut">SHIFT+r</kbd>
			</button>
		</li>
		<li>
			<button
				class="context-menu-item"
				on:click={(e) => selectByLED(1, true)}
			>
				<div class="context-menu-label">
					Only <span class="g-led">560 nm</span> on
				</div>
				<kbd class="context-menu-shortcut">SHIFT+g</kbd>
			</button>
		</li>
		<li>
			<button
				class="context-menu-item"
				on:click={(e) => selectByLED(2, true)}
			>
				<div class="context-menu-label">
					Only <span class="b-led">460 nm</span> on
				</div>
				<kbd class="context-menu-shortcut">SHIFT+b</kbd>
			</button>
		</li>
		<li>
			<button
				class="context-menu-item"
				on:click={(e) => selectByLED(3, true)}
			>
				<div class="context-menu-label">
					Only <span class="uv-led">413 nm</span> on
				</div>
				<kbd class="context-menu-shortcut">SHIFT+u</kbd>
			</button>
		</li>
		<li class:disabled={!isSimilarOptionEnabled}>
			<button class="context-menu-item" on:click={(e) => selectSimilar()}>
				<div class="context-menu-label">
					Similar to:
					{#if isSimilarOptionEnabled}
						<span
							class="context-menu-circ-icon"
							style="background-color: {similarColor};"
						/>
					{:else}
						<span style="font-size: 80%; margin-left:0.5rem;"
							>&lt;color&gt;</span
						>
					{/if}
				</div>
				<kbd class="context-menu-shortcut">s</kbd>
			</button>
		</li>
		<li>
			<button
				class="context-menu-item"
				on:click={(e) => selectFirst(0.1)}
			>
				<div class="context-menu-label">Follows (100 ms)</div>
			</button>
		</li>
		<li>
			<button
				class="context-menu-item"
				on:click={(e) => selectFirst(0.5)}
			>
				<div class="context-menu-label">Follows (500 ms)</div>
			</button>
		</li>
		<li>
			<button
				class="context-menu-item"
				on:click={(e) => selectFirst(1.0)}
			>
				<div class="context-menu-label">Follows (1000 ms)</div>
			</button>
		</li>
		<li>
			<button class="context-menu-item">
				<div class="context-menu-label">First (100 ms)</div>
			</button>
		</li>
		<li>
			<button class="context-menu-item">
				<div class="context-menu-label">First (500 ms)</div>
			</button>
		</li>
		<li>
			<button class="context-menu-item">
				<div class="context-menu-label">First (1000 ms)</div>
			</button>
		</li>
	</menu>
{/if}

<style>
	menu {
		margin: 0;
		width: 14rem;
		background-color: rgb(241, 243, 245);
		padding: 0.5rem 0;
		position: absolute;
		display: grid;
		border: 1px solid rgb(173, 181, 189);
		border-radius: 4px;
		box-shadow: 0px 3px 10px rgba(0, 0, 0, 0.2);
		/* Show hand rather than insertion cursor. */
		cursor: default;
	}
	menu li {
		/* Remove list styling */
		list-style-type: none;
	}
	menu li:hover {
		background-color: rgba(50, 50, 50, 0.2);
	}
	menu li.disabled:hover {
		background-color: rgb(241, 243, 245);
	}

	menu li.disabled button {
		color: #0006;
	}
	menu li button {
		padding: 3px 14px;
		margin: 0;
		width: 100%;
		display: grid;
		grid-template-columns: 1fr 0.4fr;
		/* The button can contain a label and shortcut div. Align the vertically. */
		align-items: center;
		/* Remove button styling */
		background-color: transparent;
		border: none;
	}
	menu li button div.context-menu-label {
		justify-self: start;
	}

	menu li button kbd.context-menu-shortcut {
		justify-self: end;
		font-size: 0.75rem;
		opacity: 0.6;
	}

	.r-led {
		color: hsl(0, 70%, 30%);
	}
	.g-led {
		color: hsl(114, 70%, 30%);
	}
	.b-led {
		color: hsl(181, 70%, 30%);
	}
	.uv-led {
		color: hsl(235, 70%, 30%);
	}

	.context-menu-circ-icon {
		height: 0.6rem;
		width: 0.6rem;
		border-radius: 50%;
		border: 1px solid #000;
		display: inline-block;
		margin-left: 0.5rem;
	}
</style>
