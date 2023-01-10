import {getContext, onMount, onDestroy} from 'svelte';
import {writable, derived, readable, get} from 'svelte/store';
import * as PIXI from 'pixi.js';
// math extras for easy point manipulation, like point.subtract(other)
import '@pixi/math-extras';
import {Viewport} from 'pixi-viewport';


let _playbackSpeed = 0.05;
let _sampleRate = 99.182;
let _snippetLen = 90;
let _snippetPad = 10;
let snippetDurationSecs = _snippetLen / _sampleRate;
let padDurationSecs = _snippetPad / _sampleRate;
const downsample = 180;
let _recordings: Recording[] = null;
let recsById = null;
let eventHandler = null;

// Some props for the app
export const width: any = writable(window.innerWidth);
export const height: any = writable(window.innerHeight);
export const pixelRatio: any = writable(window.devicePixelRatio);
export const context = writable();
export const canvas = writable();
export const playbackTime = writable(0);
/* A read-only store that is internally tied to playbackTime. 

There are many objects that only want to be updated when the snippet
data to be shown changes. This may not be on every update to the
playback time, and so we use another store. The subscribers won't be
notified if the value is set to the same as the current value. 
*/
export const snippetIdx = new function () {
	// We will only make subscribe public. Snippet idx should only
	// be updated by callbacks from playbackTime updates.
	const {subscribe, set, update} = writable(0);
	this.subscribe = subscribe
	// The update is called here, but nowhere else.
	playbackTime.subscribe((t) => {
		const nextSnippetIdx = sampleIdx(t);
		set(nextSnippetIdx);
	});
}();


export class GroupSelectData {
	groupName : string;
	spikes : number[];

	constructor(group : Group) {
		this.groupName = group.title;
		this.spikes = []; 
		for(let s = 0; s < group.snippets.length; s++) {
			this.spikes.push(group.snippets[s].spikeIdx);
		}
	}
}

export const groupSelection = new function () {
	const {subscribe, set, update} = writable([]);
	this.subscribe = subscribe;
	// Update the spike selection when a group selection changes.
	this.setGroups = (groupSelection : Group[]) => {
		const selectData = []
		for(let g = 0; g < groupSelection.length; g++) {
			selectData.push(new GroupSelectData(groupSelection[g]));
		}
		set(selectData);
	};
}

groupSelection.subscribe((spikes) => {
	console.log("spike selection changed", spikes);
});



// Same thing as above, just different style. 
// Refer to "new" behaviour: https://javascript.info/constructor-new
export const pauseCtrl = (() => {
	const {subscribe, set, update} = writable(false);
	const pause = () => {
		console.log('paused');
		set(true);
	}
	const resume = () => {
		console.log('resumed');
		set(false);
	}
	const toggle = () => {
		update(v => !v);
	}

	const current = () => {
		//return get(subscribe);
		// Hacky
		return get(pauseCtrl);
	}
	/* This is ECMAScript 6 syntax equavalent to:
		return {
		 "subscribe": subscribe,
		 "pause": pause,
		 "resume": resume,
		 "toggle": toggle,
		 "current": current
		}
	*/
	return {
		subscribe,
		pause,
		resume,
		toggle,
		current
	}
})();

// A more convenient store for grabbing all game props
export const props = deriveObject({
	context,
	canvas,
	width,
	height,
	pixelRatio,
});


/**
	input is [0, 1] values.
*/
function rgbToNumeric(r: number, g: number, b: number): number {
	const res = (r * 255 << 16) + (g * 255 << 8) + (b * 255 | 0);
	return res;
}

function absSnippetIdx(spikeIdx: number, snippetIdx: number): number {
	const res = spikeIdx - (_snippetLen - _snippetPad) + snippetIdx;
	return res
}

function sampleIdx(_playbackTime: number): number {
	const res = Math.floor((_playbackTime * _sampleRate) % _snippetLen);
	return res;
}

class Recording {
	id: string;
	name: string;
	sensorSampleRate: number;
	stimulus: number[][];
	clusterIds: number[];

	constructor(id: string, name: string, sensorSampleRate: number,
		clusterIds: number[]) {
		this.id = id;
		this.name = name;
		this.sensorSampleRate = sensorSampleRate;
		this.clusterIds = clusterIds;
		this.stimulus = null;
	}

	async fetchStimulus(): Promise<number[][]> {
		if (this.stimulus == null) {
			// Fetch data
			this.stimulus = await fetch(`/api/recording/${this.id}/stimulus?downsample=${downsample}`)
				.then(response => response.json());
		}
		return this.stimulus
	}

	/*stimulusAtTime(time: number): number[] {
		const index = Math.floor(time * this.sensorSampleRate / downsample);
		return this.stimulus[index];
	}*/
}


export async function populateRecordings() {
	return fetch('/api/recordings')
		.then((response) => response.json())
		.then((data) => {
			_recordings = []
			recsById = new Map();
			for (let i = 0; i < data.length; i++) {
				let rec = new Recording(
					data[i].id,
					data[i].name,
					data[i].sensorSampleRate,
					data[i].clusterIds);
				_recordings.push(rec);
				recsById.set(rec.id, rec);
			}
		});
}


// Don't let duplicate calls trigger multiple fetches.
let recFetchPromise = null;
export async function recordings() {
	if (recFetchPromise == null) {
		recFetchPromise = populateRecordings();
	}
	await recFetchPromise;
	return _recordings;
}

// export function getRecById(id: number) {
// 	return recsById.get(id);
// }


/* pixijs UI elements */

interface Selectable {
	setSelected(selected: boolean): void;
	//setFocused(focused: boolean): void;
	dragTarget(): PIXI.DisplayObject;
}
/* Encaplusates drag to move functionality */
class SelectionManager {
	selected: Selectable[] = [];
	focused: Selectable = null;
	selectedType: string = null;
	dragStartPosition: PIXI.Point;
	dragObjStarts: PIXI.Point[] = [];
	viewport: Viewport;
	workspace: Workspace;
	// We need the event handlers to be referencable functions for off to work.
	// https://api.pixijs.io/@pixi/interaction/PIXI/InteractionEvent.html
	onGroupDragMove: (e: PIXI.InteractionEvent) => void;
	onGroupDragEnd: (e: PIXI.InteractionEvent) => void;
	onGroupNoDrag: (e: PIXI.InteractionEvent) => void;
	onKeyDown: (e) => void;
	selectionChangedListeners: ((selected: Selectable[], selectedType: string) => void)[] = [];

	constructor(canvas: HTMLCanvasElement, viewport: Viewport,
		workspace: Workspace) {
		this.viewport = viewport;
		this.workspace = workspace;
		this.onGroupDragMove = this.createOnGroupDragMove();
		this.onGroupDragEnd = this.createOnGroupDragEnd();
		this.onGroupNoDrag = this.createOnGroupNoDrag();
		this.viewport.on('pointerdown', this.onPointerDown());
		this.onKeyDown = this.createOnKeyDown();
		// Add keypress listener.
		canvas.addEventListener(
			'mouseleave', () => {
				window.removeEventListener('keydown', this.onKeyDown)
			});
		canvas.addEventListener(
			'mouseenter', () => {
				window.addEventListener('keydown', this.onKeyDown)
			});
	}

	addSelectionChangedListener(
		listener: (selected: Selectable[], selectedType: string) => void) {
		this.selectionChangedListeners.push(listener);
		function unsubscribe() {
			let idx = this.selectionChangedListeners.indexOf(listener);
			if (idx >= 0) {
				this.selectionChangedListeners.splice(idx, 1);
			}
		}
		return unsubscribe;
	}

	notifySelectionChangedListeners() {
		for (let listener of this.selectionChangedListeners) {
			listener(this.selected, this.selectedType);
		}
	}

	addSelection(obj: Selectable, type: string, ctrlKey: boolean = false,
		shiftKey: boolean = false, onup: boolean = false) {
		console.log('onup', onup);
		// Only one type of object is selectable at a time (group or snippet).
		let dirty = false;
		if (!(this.selectedType === type)) {
			this.clearSelection(null, false);
			dirty = true;
		}
		const add = (s: Selectable) => {
			this.selected.push(s);
			s.setSelected(true);
			dirty = true;
		};
		const remove = (s: Selectable) => {
			s.setSelected(false);
			this.selected = this.selected.filter((x) => x !== s);
			dirty = true;
		};
		// Multi or single selection.
		if (ctrlKey || shiftKey) {
			// Multi-selection requested.
			// When using modifiers, we don't have the drag issue like below,
			// so we can safely act on down press. 
			if(!onup) {
				// If already selected, toggle.
				if (this.selected.includes(obj)) {
					// Remove
					remove(obj);
				} else {
					// Add
					add(obj);
				}
			} else {
				// Do nothing!
			}
		} else {
			// Single select requested.
			// Aleady selected?
			if (this.selected.includes(obj)) {
				// No drag on up, so clear others.
				if (onup) {
					// Being a bit sneaky here, and letting the clearSelection()
					// handle the notify().
					const dirtyFromClear = this.clearSelection(obj, false);
					dirty = dirty || dirtyFromClear;
				} else {
					// Drag might happen, so do nothing.
				}
			} else {
				this.clearSelection(null, false);
				add(obj);
			}
		}
		this.selectedType = type;
		if(dirty) {
			this.notifySelectionChangedListeners();
		}
	}



	/**
		Clear selection, leaving an optional item still selected.
	*/
	clearSelection(leaveSelected: Selectable = null, notify: boolean = true) {
		let res = [];
		let selectedType = null;
		let dirty = false;
		for (let i = 0; i < this.selected.length; i++) {
			if (this.selected[i] === leaveSelected) {
				res.push(this.selected[i]);
				selectedType = this.selectedType;
			} else {
				this.selected[i].setSelected(false);
				dirty = true;
			}
		}
		this.selected = res;
		this.selectedType = selectedType;
		if(dirty && notify) {
			this.notifySelectionChangedListeners();
		}
		return dirty;
	}

	addGroup(group: Group) {
		const dragTarget = group.dragTarget();
		dragTarget.on('pointerdown',
			(event: PIXI.InteractionEvent) => {
				// If middle click, then ignore. We want the viewport to 
				// handle it as a pad.
				if (event.data.button === 1) {
					return;
				}
				// Only select or drag on left click.
				if (!(event.data.button === 0)) {
					return;
				}
				// I didn't think there was any propagation/bubbling, but it 
				// seems as event.stopPropagation() is needed to prevent the
				// selection from being cleared by the viewport listener.
				event.stopPropagation();
				// Selection
				this.focused = group;
				this.addSelection(
					group,
					'group',
					event.data.originalEvent.ctrlKey);
				// Drag
				// The event seems to be modified inplace by subsequent events,
				// so we need a copy.
				// Get pos in viewport coords.
				const pointerPos = event.data.getLocalPosition(this.viewport);
				//const pointerPos = event.data.global.clone();
				this.onGroupDragStart(pointerPos);
			});
	}

	/**
		Behaviour:
			- This callback will only be called if another object doesn't 
			  handle the event. Thus, we can safely clear selection here.  
	*/
	onPointerDown() {
		const __this = this;
		return function (event: PIXI.InteractionEvent) {
			// Only clear on left or middle click (right and middle shouldn't clear).
			if (event.data.button === 0) {
				__this.clearSelection();
			}
		};
	}

	createOnKeyDown() {
		const __this = this;
		return function (event) {
			// Delete any selected items.
			if (event.code === 'Delete') {
				// Delete selected groups.
				for (let i = 0; i < __this.selected.length; i++) {
					const group = __this.selected[i] as Group;
					__this.workspace.deleteGroup(group);
				}
				__this.selected = [];
				__this.selectedType = null;
			}
		}
	}

	onGroupDragStart(dragStartPos: PIXI.Point) {
		this.dragStartPosition = dragStartPos;
		this.dragObjStarts = [];
		for (let i = 0; i < this.selected.length; i++) {
			this.dragObjStarts.push(
				this.selected[i].dragTarget().position.clone());
		}
		this.viewport.on('pointermove', this.onGroupDragMove);
		this.viewport.on('pointerup', this.onGroupNoDrag);
		this.viewport.on('pointerup', this.onGroupDragEnd);
		this.viewport.on('pointerupoutside', this.onGroupDragEnd);
	}

	/**
		Create the single onGroupDragMove callback for all groups.

		It needs to be a single callback so that it can be referenced for
		deregistration. If this wasn't needed, then we could partially apply
		the arguments and not need to store an object property.
	*/
	createOnGroupDragMove() {
		const __this = this;
		return function (event: PIXI.InteractionEvent) {
			// Movement means we don't need to deselect on pointerup.
			__this.viewport.off('pointerup', __this.onGroupNoDrag);
			// Skip if selection is empty, or if selection type is not group.
			if (!__this.selected.length) {
				return
			}
			if (!(__this.selectedType === 'group')) {
				throw new Error('Groups should be selected');
			}
			//const newPosition = event.data.global;
			const newPosition = event.data.getLocalPosition(__this.viewport);
			const moveVector = newPosition.subtract(__this.dragStartPosition);
			// toLocal(position, <what?>, <point to store result>)
			for (let i = 0; i < __this.selected.length; i++) {
				const dragTarget = __this.selected[i].dragTarget();
				dragTarget.alpha = 0.7;
				dragTarget.position = __this.dragObjStarts[i].add(moveVector);
			}
		};
	}

	createOnGroupDragEnd() {
		const __this = this;
		return function (event: PIXI.InteractionEvent) {
			__this.viewport.off('pointermove', __this.onGroupDragMove);
			if (!__this.selected.length) {
				return
			}
			if (__this.selectedType != 'group') {
				throw new Error('Groups should be selected');
			}
			for (let i = 0; i < __this.selected.length; i++) {
				__this.selected[i].dragTarget().alpha = 1;
			}
			__this.dragStartPosition = null;
			__this.dragObjStarts = null;
		};
	}

	/**
		pointerdown on a selected object in the presence of multiple selected
		objects shouldn't clear the selection until pointerup, as we need
		the pointerdown to be used for dragging. This means we need to wait
		until a move happens to determine if any selection should be cleared.
		The below callback is added when dragging is initialized but removed
		once a movement occurs.
	*/
	createOnGroupNoDrag() {
		const __this = this;
		return function (event: PIXI.InteractionEvent) {
			// It's a one time event.
			__this.viewport.off('pointerup', __this.onGroupNoDrag);
			__this.addSelection(__this.focused,
				'group',
				event.data.originalEvent.ctrlKey,
				event.data.originalEvent.shiftKey,
				true);
		};
	}
}

class TimeControl {
	// Appearance
	width = 400;
	height = 70;
	padding = [20, 10];
	innerWidth: number;
	innerHeight: number;

	// Data
	time: number;
	spikeTime: number;
	totalTime: number;
	container: PIXI.Container;
	graphics: PIXI.Graphics;
	timeText: PIXI.Text;

	// Controls
	isDragging: boolean;
	wasPausedBefore: boolean;
	// Keep these as methods so that we can refer to them in "container.off".
	// We want to remove the callbacks when we can to both reduce spurious
	// behaviour and to improve performance.
	onDragMove: (event: PIXI.InteractionEvent) => void;
	onDragEnd: (event: PIXI.InteractionEvent) => void;


	constructor(snippetSecs: number, padSecs: number, playbackTimer) {
		this.innerWidth = this.width - this.padding[0] * 2;
		this.innerHeight = this.height - this.padding[1] * 2;
		this.spikeTime = snippetSecs - padSecs;
		this.totalTime = snippetSecs;
		this.time = 0;
		this.isDragging = false;
		this.wasPausedBefore = isPaused();
		this.onDragMove = this.createOnDragMove();
		this.onDragEnd = this.createOnDragEnd();

		this.initContainer();
		playbackTimer.subscribe((t: number) => {
			this.updateTime(t);
		});
	}

	initContainer() {
		this.container = new PIXI.Container();
		this.container.interactive = true;
		this.container.on("pointerdown", this.onPointerDown());
		// Don't do this here, wait until dragging starts.
		// this.container.on("pointerup", this.onDragEnd);
		// this.container.on("pointermove", this.onDragMove);
		this.timeText = new PIXI.Text(
			this.timeString(),
			{
				fontFamily: 'sans-serif',
				fontSize: 16,
				fill: 0xffffff,
				align: 'center'
			});
		this.graphics = new PIXI.Graphics();
		this.container.addChild(this.graphics);
		this.container.addChild(this.timeText);
		this.rebuild();
		// Not sure why this is required. Without it, only clicks on the text
		// are registered.
		this.container.hitArea = new PIXI.Rectangle(
			0, 0, this.width, this.height);
	}

	updateTime(time: number) {
		if (time !== this.time) {
			this.time = time;
			this.rebuild();
		}
	}

	toPos(px: number, py: number): number[] {
		if (px < 0 || px > 1 || py < 0 || py > 1) {
			throw new Error("Invalid position");
		}
		const res = [
			px * this.innerWidth + this.padding[0],
			py * this.innerHeight + this.padding[1]];
		return res;
	}

	xToRel(x: number) {
		x = x - this.padding[0] / 2;
		x = Math.max(x, 0);
		x = Math.min(x, this.innerWidth);
		const rel = x / this.innerWidth;
		return rel;
	}

	timeString() {
		const timeSinceSpike = this.time - this.spikeTime;
		return `t = ${timeSinceSpike.toFixed(2)} s`;
	}

	rebuild() {
		this.graphics.clear();
		// Line
		const fraction = this.time / this.totalTime;
		const lineY = 0.5;
		// Dot
		const dotY = lineY;
		const dotRadius = 3;
		// Text
		const textY = dotY + 0.2;

		// Rect boundary
		//this.graphics.beginFill(0xd0d0d0);
		//this.graphics.drawRect(0, 0, this.width, this.height);
		//this.graphics.endFill();

		// Hit area, in red
		//const hitArea = this.container.hitArea;
		//this.graphics.lineStyle(1, 0xff0000, 1);
		//this.graphics.drawRect(hitArea.x, hitArea.y, hitArea.width, hitArea.height);
		//this.graphics.endFill();

		// Draw line
		this.graphics.lineStyle(1, 0xffffff, 1);
		this.graphics.moveTo(...this.toPos(0, lineY));
		this.graphics.lineTo(...this.toPos(1, lineY));
		this.graphics.endFill()

		// Draw spike marker
		this.graphics.lineStyle(1, 0xff9090, 1);
		const spikeRel = this.spikeTime / this.totalTime;
		this.graphics.moveTo(...this.toPos(spikeRel, 0.0));
		this.graphics.lineTo(...this.toPos(spikeRel, 1.0));
		this.graphics.endFill()

		// Draw moving dot
		this.graphics.lineStyle(1, 0xffffff, 1);
		this.graphics.beginFill(0xffffff, 1);
		this.graphics.drawCircle(...this.toPos(fraction, dotY), dotRadius);
		this.graphics.endFill();

		// Draw moving text.
		// Text
		this.timeText.text = this.timeString();
		let textPos = this.toPos(fraction, textY);
		const textXOffset = -this.timeText.width / 2;
		textPos[0] += textXOffset;
		this.timeText.position.set(textPos[0], textPos[1]);

	}

	onPointerDown() {
		const __this = this;
		return (event: PIXI.InteractionEvent) => {
			// Register the drag-move and drag-exit callbacks.
			__this.container.on("pointermove", this.onDragMove);
			__this.container.on("pointerup", this.onDragEnd);
			__this.container.on("pointerupoutside", this.onDragEnd);
			// Pause if not paused.
			__this.wasPausedBefore = isPaused();
			pauseCtrl.pause();
			// Jump to the chosen playback time.
			setPlaybackTimeRel(__this.xToRel(
				// local pos
				event.data.getLocalPosition(__this.container).x));
			__this.isDragging = true;
		};
	}

	createOnDragMove() {
		const __this = this;
		return (event: PIXI.InteractionEvent) => {
			if (__this.isDragging) {
				const pos = event.data.getLocalPosition(this.container);
				setPlaybackTimeRel(__this.xToRel(pos.x));
			}
		};
	}

	createOnDragEnd() {
		const __this = this;
		return (e) => {
			// Remove the temporary listeners.
			__this.container.off("pointerup", this.onDragEnd);
			__this.container.off("pointermove", this.onDragMove);
			if (__this.isDragging) {
				// Only resume if it was running beforehand.
				if (!__this.wasPausedBefore) {
					pauseCtrl.resume();
				}
				__this.isDragging = false;
			}
		};
	}
}

class MainLayout {
	app: PIXI.Application;
	viewport: Viewport;
	timeControl: TimeControl;
	container: PIXI.Container;

	constructor(app: PIXI.Application, viewport: Viewport,
		timeControl: TimeControl) {
		this.app = app;
		this.viewport = viewport;
		this.timeControl = timeControl;
		this.container = new PIXI.Container();
		this.initContainer()
	}

	initContainer() {
		this.container.addChild(this.viewport);
		this.container.addChild(this.timeControl.container);
		this.rebuild();
	}

	rebuild() {
		//this.timeControl.container.pivot.x = 0.5
		this.timeControl.container.position.x =
			(this.app.renderer.width - this.timeControl.container.width) / 2;
	}
}


class Snippet {
	static defaultSquareLen = 10;
	static padFactor = 0.05;
	recId: number;
	cellId: number;
	//cell_gid : number;
	spikeIdx: number;
	snipIdx: number;
	container: PIXI.Container;
	rect1: PIXI.Graphics;
	squareLen: number;
	// This is saved and called on destroy() so that we can clean up
	// references to this object when it's removed from the scene. It's
	// the unsubscribe for the Svelte store that tracks the snippet idx.
	idxUnsubscribe;
	// Something like this to check if the snippet has been 
	// placed on the canvas or is still as an outline while
	// being placed.
	// isPlaced : boolean;

	constructor(
		recId: number,
		cellId: number,
		spikeIdx: number) {
		this.recId = recId;
		this.cellId = cellId;
		this.spikeIdx = spikeIdx;
		this.squareLen = Snippet.defaultSquareLen;
		this.initContainer();
		this.idxUnsubscribe = snippetIdx.subscribe(this.onSnippetIdxUpdate());
	}

	destroy() {
		this.idxUnsubscribe();
	}

	initContainer() {
		// Add children to container
		this.container = new PIXI.Container();
		this.rect1 = new PIXI.Graphics();
		this.container.addChild(this.rect1);
		this.snipIdx = 0;
		this.rebuild();
	}

	rebuild() {
		// Clear and rebuild square.
		this.rect1.clear();
		const color = this.color();
		this.rect1.beginFill(color);
		const pad = this.squareLen * Snippet.padFactor;
		this.rect1.drawRect(pad, pad,
			this.squareLen - pad * 2,
			this.squareLen - pad * 2);
		this.rect1.endFill();
	}

	onSnippetIdxUpdate() {
		let instance = this;
		return (idx: number) => {
			instance.snipIdx = idx;
			instance.rebuild();
		};
	}

	color() {
		const absIdx = absSnippetIdx(this.spikeIdx, this.snipIdx);
		let res: number;
		if (absIdx < 0) {
			res = 0x000000;
		} else {
			const stim = recsById.get(this.recId).stimulus[absIdx];
			//res = PIXI.utils.rgb2hex([Math.max(0, Math.min(1, stim[0])), 
			const scale = 0.9;
			res = rgbToNumeric(scale * Math.max(0, Math.min(1, stim[0])),
				scale * Math.max(0, Math.min(1, stim[1])),
				scale * Math.max(0, Math.min(1, stim[2])));
		}
		return res;
	}
}

class Group implements Selectable {
	// Initial shape will be square
	static lenOnCreation = 100;
	static padding = 3;
	static maxSquareSize = 50;
	// Colors inspired from: https://reasonable.work/colors/
	static groupColors = [
		0x302100, // amber
		0x510018, // raspberry
		0x002812, // emerald
		0x44003c, // magenta
		0x001d2a, // cerulean
		0x292300, // yellow
	];
	title: string;
	snippets: Snippet[];
	container: PIXI.Container;
	header: PIXI.Container;
	gRect: PIXI.Graphics;
	color: number;
	width: number;
	height: number;
	// Selection, drag & drop and resize state.
	isSelected: boolean = false;


	constructor(title: string) {
		this.title = title;
		this.snippets = [];
		this.color = Group.groupColors[Math.floor(Math.random() * 6)];
		this.initContainer();
	}

	destroy() {
		for (const snip of this.snippets) {
			snip.destroy();
		}
	}

	initContainer() {
		this.container = new PIXI.Container();
		this.container.interactive = true;
		// Header for group title
		// A better way is to create a "resize" container, and have the
		// cursor change change automatically when over it. This will also
		// make the mouse-up and mouse-down event easier to handle.
		//this.container.on('mousemove', this.onMouseMove());
		this.gRect = new PIXI.Graphics();
		this.container.addChild(this.gRect);
		this.header = Group.addHeader(this.container, this.title);
		this.width = Group.lenOnCreation;
		this.height = Group.lenOnCreation;
		this.rebuild();
	}

	static addHeader(container: PIXI.Container, text: string) {
		// The text objects are made once, so make them big and scale them
		// down so as to give better resolution when zooming in. Larger text
		// is more memory and cpu intensive.
		const scale = 5
		const size = 12 * scale
		// bottom margin is needed to raise above the highlight stroke.
		const bottomMargin = 3;
		const header = new PIXI.Container();
		const textObj = new PIXI.Text(text, {fontSize: size, fill: 0xf0f0f0});
		textObj.scale.set(1 / scale, 1 / scale);
		header.addChild(textObj);
		container.addChild(header);
		header.position.set(0, -(textObj.height + bottomMargin));
		return header;
	}

	rebuild() {
		this.gRect.clear();
		if (this.isSelected) {
			this.gRect.lineStyle(
				{
					width: 1,
					color: 0xFEEB77,
					// Allignment (0 = inner, 0.5 = middle, 1 = outer)
					alignment: 1
				}
			);
		} else {
			this.gRect.lineStyle(0);
		}
		this.gRect.beginFill(this.color);
		this.gRect.drawRect(0, 0, this.width, this.height);
		this.gRect.endFill();
		this.positionSnippets();
	}

	positionSnippets() {
		if (this.snippets.length == 0) {
			return
		}
		const squareLen = this.calcSquareLen();
		let x = Group.padding;
		let y = Group.padding;
		for (let i = 0; i < this.snippets.length; i++) {
			const snippet = this.snippets[i];
			snippet.container.x = x;
			snippet.container.y = y;
			snippet.squareLen = squareLen;
			snippet.rebuild();
			x += squareLen;
			if ((x + squareLen) > (this.width - Group.padding)) {
				x = Group.padding;
				y += squareLen;
			}
		}
	}

	calcSquareLen() {
		const w = this.width - (Group.padding * 2);
		const h = this.height - (Group.padding * 2);
		const n = this.snippets.length;
		const upperBound = Group.maxSquareSize;
		const lowerBound = Math.floor(Math.min(w, h) / Math.ceil(Math.sqrt(n)));
		function gridCapacity(l: number) {
			return Math.floor(w / l) * Math.floor(h / l);
		}
		let L = lowerBound;
		let H = upperBound;
		const resolution = 0.01;
		const maxIter = 20;
		let iter = 0;
		while (L < H && iter < maxIter) {
			const M = (L + H) / 2;
			if (gridCapacity(M) >= n) {
				L = M;
			} else {
				H = M - resolution;
			}
			iter++;
		}
		return L;
	}

	addSnippets(snippets: Snippet[]) {
		for (let i = 0; i < snippets.length; i++) {
			this.snippets.push(snippets[i]);
			this.container.addChild(snippets[i].container);
		}
		this.rebuild();
	}

	setSelected(selected: boolean) {
		if (this.isSelected == selected) {
			return;
		}
		this.isSelected = selected;
		this.rebuild();
	}

	dragTarget() {
		return this.container;
	}
}

class Workspace {
	static margin = 20;
	static numCols = 6;
	groups: Group[];
	container: PIXI.Container;
	col: number = 0;
	row: number = 0;

	constructor() {
		this.groups = [];
		this.container = new PIXI.Container();
	}

	addGroup(group: Group) {
		this.groups.push(group);
		const nextX = Group.lenOnCreation + this.col *
			(Group.lenOnCreation + Workspace.margin);
		const nextY = Group.lenOnCreation + this.row *
			(Group.lenOnCreation + Workspace.margin);
		this.col++;
		if (this.col > Workspace.numCols) {
			this.col = 0;
			this.row++;
		}
		group.container.x = nextX;
		group.container.y = nextY;
		this.container.addChild(group.container);
		eventHandler.addGroup(group);
	}

	deleteGroup(group: Group) {
		const index = this.groups.indexOf(group);
		if (index < 0) {
			return;
		}
		this.groups.splice(index, 1);
		this.container.removeChild(group.container);
		group.destroy();
	}
}

export let workspace = new Workspace();


export async function addCellAsGroup(recId: number, cellId: number) {
	// Preload the stimulus data.
	const rec = recsById.get(recId);
	await rec.fetchStimulus();
	const spikes = await fetch(`/api/recording/${recId}/cell/${cellId}/spikes`)
		.then(response => response.json());
	let snippets = [];
	for (let i = 0; i < spikes.length; i++) {
		snippets.push(new Snippet(recId, cellId, spikes[i]));
	}
	const group = new Group(`r-${rec.id} c-${cellId}`);
	group.addSnippets(snippets);
	workspace.addGroup(group);
}


export function updateClock(dt: number) {
	if (!get(pauseCtrl)) {
		playbackTime.update(curr => {
			return (curr + dt * _playbackSpeed) % playbackDuration();
		});
	}
}

function onSelectionChange(selected : Selectable[], selectionType : string) {
	console.log(selected, selectionType);
	if(selectionType == "group") {
		groupSelection.setGroups(selected);
	}
}

export function startEngine(canvas, app: PIXI.Application, viewport: Viewport) {
	viewport.addChild(workspace.container);
	app.ticker.add(ontick);
	// Outer container
	const layout = new MainLayout(
		app,
		viewport,
		new TimeControl(snippetDurationSecs, padDurationSecs, playbackTime));
	app.stage.addChild(layout.container);
	eventHandler = new SelectionManager(canvas, viewport, workspace);
	eventHandler.addSelectionChangedListener(onSelectionChange);
}

/**
 * I think the delta is in fractional number of frames, not seconds.
 */
function ontick(deltaFrames: number) {
	const dt = deltaFrames / 60;
	updateClock(dt);
}

// Old stuff again

export function playbackDuration() {
	return _snippetLen / _sampleRate;
}

export function snippetLen() {
	return _snippetLen;
}

export function snippetPad() {
	return _snippetPad;
}

export function isPaused() {
	return get(pauseCtrl);
}

export function setPlaybackTime(t: number) {
	playbackTime.set(t);
}

/* Used? */
export function setPlaybackTimeRel(r: number) {
	// Either here or somewhere else, handle the the fact that modulo will
	// send the end to the begininning.
	r = Math.min(Math.max(0, r), 0.9999999);
	playbackTime.set(r * _snippetLen / _sampleRate);
}


function deriveObject(obj) {
	const keys = Object.keys(obj);
	const list = keys.map(key => {
		return obj[key];
	});
	return derived(list, (array) => {
		return array.reduce((dict, value, i) => {
			dict[keys[i]] = value;
			return dict;
		}, {});
	});
}
