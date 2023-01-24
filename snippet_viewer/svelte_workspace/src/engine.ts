import {writable, derived, get} from 'svelte/store';
import * as PIXI from 'pixi.js';
// math extras for easy point manipulation, like point.subtract(other)
import '@pixi/math-extras';
import type {Viewport} from 'pixi-viewport';
import * as colorutils from './color.js';


let _playbackSpeed = 0.05;
let _sampleRate = 99.182;
let _snippetLen = 90;
let _snippetPad = 10;
let snippetDurationSecs = _snippetLen / _sampleRate;
let padDurationSecs = _snippetPad / _sampleRate;
const downsample = 180;
let _recordings: Recording[] = null;
let recsById = null;

// Some props for the app
export const width: any = writable(window.innerWidth);
export const height: any = writable(window.innerHeight);
export const pixelRatio: any = writable(window.devicePixelRatio);
export const context = writable();
export const canvas = writable();
export const playbackTime = writable(0);
// Selection mode: group or snippet
export const selectionMode = writable('group');
/* A read-only store that is internally tied to playbackTime. 

There are many objects that only want to be updated when the snippet
data to be shown changes. This may not be on every update to the
playback time, and so we use another store. The subscribers won't be
notified if the value is set to the same as the current value. 
*/
export const snippetIdx = new function () {
	// We will only make subscribe public. Snippet idx should only
	// be updated by callbacks from playbackTime updates.
	const {subscribe, set, /* update */} = writable(0);
	this.subscribe = subscribe
	// The update is called here, but nowhere else.
	playbackTime.subscribe((t) => {
		const nextSnippetIdx = sampleIdx(t);
		set(nextSnippetIdx);
	});
}();


export class GroupSelectData {
	groupName: string;
	spikes: number[];

	constructor(group: Group) {
		this.groupName = group.title;
		this.spikes = [];
		for (let s = 0; s < group.snippets.length; s++) {
			this.spikes.push(group.snippets[s].spikeIdx);
		}
	}
}

export const groupSelection = new function () {
	const {subscribe, set, /* update */} = writable([]);
	this.subscribe = subscribe;
	// Update the spike selection when a group selection changes.
	this.setGroups = (groupSelection: Group[]) => {
		const selectData = []
		for (let g = 0; g < groupSelection.length; g++) {
			selectData.push(new GroupSelectData(groupSelection[g]));
		}
		set(selectData);
	};
}

// Same thing as above, just different style. 
// Refer to "new" behaviour: https://javascript.info/constructor-new
export const pauseCtrl = (() => {
	const {subscribe, set, update} = writable(true);
	const pause = () => {
		set(true);
	}
	const resume = () => {
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



/**
 * Enum for 3 types of selection: none, single, double (focus).
 */
class SelectState {
	static None = new SelectState("None");
	static Selected = new SelectState("Selected");

	name: string;

	private constructor(name: string) {
		this.name = name;
	}

	toString() {
		return this.name;
	}
}


class DragState {
	static OverParent = new DragState("OverParent");
	static OverNothing = new DragState("OverNothing");
	static OverDropReceiver = new DragState("OverDropReceiver");

	name: string;

	private constructor(name: string) {
		this.name = name;
	}
}

class DragMode {
	static DragToMove = new DragMode("DragToMove");
	static DragDrop = new DragMode("DragDrop");

	name: string;

	private constructor(name: string) {
		this.name = name;
	}
}

class DragTarget {
	obj: PIXI.DisplayObject;
	// Maybe DragMode should not be a property of DragTarget.
	// Currently, DragTarget is created by Group or Snippet to be returned by
	// onDrag() and in this way, the Group or Snippet controls whether it is
	// draggable or moveable. What if it wanted both at different times? If this
	// is ever needed, then DragMode should be mapped to each object within the
	// selection manager.
	mode: DragMode;
	onDragStateChange: (state: DragState) => void;

	constructor(obj: PIXI.DisplayObject, mode: DragMode,
		onDragStateChange: null | ((state: DragState) => void) = null) {
		this.obj = obj;
		this.mode = mode;
		if (onDragStateChange == null) {
			this.onDragStateChange = (_state) => {};
		} else {
			this.onDragStateChange = onDragStateChange;
		}
	}
}

interface Selectable {
	setSelectState(state: SelectState): void;
	// Two behaviours are supported based on the return type of onDragStart().
	// 1. Drag to move
	// Return null to signal that the drag target itself should be repositioned
	// on drag. This makes drag a repositioning operation. In the current setup,
	// this is used for Group movement.
	// 1. Drag and drop
	// Return a "ghost" PIXI.DisplayObject that is not yet added to the scene.
	// The selection manager will use this for creating the drag animation.
	// Return the object which you want to have moved. This could be the main
	// container itself, if you want direct object repositioning on drag; or
	// it could be a "ghost" object
	onDragStart(): DragTarget;
	onDragEnd(): void;
	// The container that should be clickable.
	selectTarget(): PIXI.DisplayObject;
}

interface DropReceiver {
	// Here, we abandon any types. Otherwise, we would have trouble accepting
	// multiple  types. A switch on the mimetype isn't that hard.
	onDrop(objects: any[], mimetype: string, mousePos: PIXI.Point): void;
	dropTarget(): PIXI.DisplayObject;
	onDragEnter(objects: any[], mimetype: string): void;
	onDragLeave(): void;
}

/**
 * Tries to generalize some of the work involved in selecting objects in the
 * scene. At this point, groups and snippets are the ownly object groups that
 * need a selection manager.
 * 
 * Note: the SelectionManager tries to not hold references to the PIXI objects
 * so as avoid needing to clean up references to these objects if they are
 * removed from the scene.
 */
export class SelectionManager<SelectableT extends Selectable> {
	static DOUBLE_CLICK_MS = 300;
	positionParent: PIXI.Container;
	canvas: HTMLCanvasElement;
	mimetype: string;

	// We might need this eventually, to manage references better and not have
	// references stuck in ananymous functions that can't be deleted.
	// selectables: SelectableT[] = [];
	// If we have references to selectables, what happens on delete? Who is 
	// responsible for telling the selection manager to remove the item?
	// I think I prefer the idea that the selection manager avoids storing too
	// much state, and the objects like Workspace hold it instead.
	selected: SelectableT[] = [];
	dragTargets: DragTarget[] = [];
	focused: SelectableT = null;
	isContextMenu: boolean = false;
	dropReceivers: DropReceiver[] = [];
	dragStartPosition: PIXI.Point = null;
	dragObjStarts: PIXI.Point[] = [];
	lastClickTimeMs: number = 0;
	selectionChangedListeners: ((selected: SelectableT[]) => void)[] = [];
	// Callbacks to be provided to this class based on the use case.
	// This is separate from the SelectableT interface, as the selectables 
	// themselves shouldn't managed their own deletion.
	// It starts as an empty function.
	_onDelete: (s: SelectableT[]) => void = () => {};
	_onDoubleClick: (s: SelectableT) => void = () => {};
	_onPreDoubleClick: (s: SelectableT) => void = () => {};
	_onDoubleSelectLeave: (s: SelectableT) => void = () => {};
	// Callback that shows a context menu. Must return a destroy function.
	_onContextMenuShow: (event: PIXI.InteractionEvent, s: SelectableT[]) => (() => void) = () => {return () => {};};
	_onContextMenuHide: () => void = () => {};
	unsubscribes: (() => void)[] = [];
	dragUnsubscribes: (() => void)[] = [];
	onSelectUnsubscribes: (() => void)[] = [];

	/**
	 * @param {HTMLCanvasElement} canvas - this is needed to track when the 
			pointer leaves the canvas. When this happens, selection actions such as
			delete on "Delete" key are disabled.
	 * @param {PIXI.Container} positionParent - for getting the position, we need 
				the parent whose coordinate system will be used.
	 */
	constructor(canvas: HTMLCanvasElement, positionParent: PIXI.Container,
		mimetype: string) {
		this.positionParent = positionParent;
		this.mimetype = mimetype;
		this.canvas = canvas;
		this.setupParentAndCanvasListeners();
	}

	/**
	 * Called after the selection becomes non-empty.
	 */
	setupParentAndCanvasListeners() {
		console.log('setup parent and canvas listeners');
		const onKeyDown = (event: KeyboardEvent) => {
			// Delete any selected items.
			console.log('selection manager, on key down');
			console.log(`event.code: ${event.code}`);
			if (event.code === 'Delete') {
				// Delete selected groups.
				for (let i = 0; i < this.selected.length; i++) {
					this._onDelete([this.selected[i]]);
				}
				this.selected = [];
			}
		};

		/**
		 * Behaviour:
		 *	- This callback will only be called if another object doesn't 
		 *		handle the event. Thus, we can safely clear selection here.  
		 */
		const onPointerDownInParent = (event: PIXI.InteractionEvent) => {
			console.log('pointer down in position parent');
			if (event.data.button == 2) {
				this.showContextMenu(event);
			}
			else {
				this.hideContextMenu();
				if (event.data.button === 0) {
					// Only clear on left (right and middle shouldn't clear).
					this.clearSelection();
				}
			}
		};

		this.positionParent.on('pointerdown', onPointerDownInParent);
		window.addEventListener('keydown', onKeyDown);
		const onLeaveCanvas = () => {
			window.removeEventListener('keydown', onKeyDown);
		};
		const onEnterCanvas = () => {
			window.addEventListener('keydown', onKeyDown);
		};
		this.canvas.addEventListener('mouseleave', onLeaveCanvas);
		this.canvas.addEventListener('mouseenter', onEnterCanvas);

		this.unsubscribes.push(() => {
			this.positionParent.off('pointerdown', onPointerDownInParent);
			this.canvas.removeEventListener('mouseleave', onLeaveCanvas);
			this.canvas.removeEventListener('mouseenter', onEnterCanvas);
			window.removeEventListener('keydown', onKeyDown);
		});
	}

	destroy() {
		console.log("destroy selection manager.")
		// Deselect all objects.
		for (const obj of this.selected) {
			obj.setSelectState(SelectState.None);
		}
		// Remove all listeners.
		const unsubs = this.unsubscribes.concat(
			this.dragUnsubscribes, this.onSelectUnsubscribes);
		for (const unsub of unsubs) {
			unsub();
		}
		// Hide context menu.
		this._onContextMenuHide();
		this.selected = [];
		this.dragStartPosition = null;
		this.dragObjStarts = [];
		this.clearDragTargets();
	}

	clearDrag() {
		for (let i = 0; i < this.selected.length; i++) {
			this.selected[i].onDragEnd();
		}
		for (const unsubFunc of this.dragUnsubscribes) {
			unsubFunc();
		}
		// This should only be registered on when dragging starts!
		this.clearDragTargets();
		this.dragStartPosition = null;
		this.dragObjStarts = [];
	}

	clearDragTargets() {
		for (let i = 0; i < this.dragTargets.length; i++) {
			const t = this.dragTargets[i];
			if (t.mode === DragMode.DragDrop) {
				this.positionParent.removeChild(t.obj);
			}
		}
		this.dragTargets = [];
	}

	onDelete(f: (s: SelectableT[]) => void) {
		this._onDelete = f;
	}

	onDoubleClick(f: (s: SelectableT) => void) {
		this._onDoubleClick = f;
	}

	onContextMenu(f: (s: SelectableT[]) => void) {
		this._onContextMenu = f;
	}

	onPreDoubleClick(f: (s: SelectableT) => void) {
		this._onPreDoubleClick = f;
	}

	isDragging() {
		return this.dragStartPosition != null;
	}

	addSelectionChangedListener(listener: (selected: SelectableT[]) => void) {
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
			listener(this.selected);
		}
	}

	addSelection(obj: SelectableT, ctrlKey: boolean = false,
		shiftKey: boolean = false, onup: boolean = false) {
		// Only one type of object is selectable at a time (group or snippet).
		let dirty = false;
		const add = (s: SelectableT) => {
			this.selected.push(s);
			s.setSelectState(SelectState.Selected);
			dirty = true;
		};
		const remove = (s: SelectableT) => {
			s.setSelectState(SelectState.None);
			this.selected = this.selected.filter((x) => x !== s);
			const isSelectionEmpty = this.selected.length === 0;
			if (isSelectionEmpty) {
				this.onSelectUnsubscribes.forEach((f) => f());
			}
			dirty = true;
		};
		// Multi or single selection.
		if (ctrlKey || shiftKey) {
			// Multi-selection requested.
			// When using modifiers, we don't have the drag issue like below,
			// so we can safely act on down press. 
			if (!onup) {
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
		if (dirty) {
			this.notifySelectionChangedListeners();
		}
	}

	/**
	 * Set selection programatically.
	 */
	setSelection(toSelect: SelectableT[]) {
		console.log("setSelection", toSelect);
		// We need two-way comparison to see if anything should be done.
		const toSelectSet = new Set(toSelect);
		const selectedSet = new Set(this.selected);
		const setEq = (setX, setY) => {
			const res = setX.size === setY.size &&
				[...setX].every((x) => setY.has(x));
			return res;
		};
		if (setEq(toSelectSet, selectedSet)) {
			console.log("setSelection: no change");
			// Nothing to do.
			return;
		}
		// TODO: should probably just add the ones that have changed.
		this.clearSelection(null, false);
		for (let i = 0; i < toSelect.length; i++) {
			this.addSelection(toSelect[i], true, false, false);
		}
	}

	/**
	 * Clear selection, leaving an optional item still selected.
	 */
	clearSelection(leaveSelected: SelectableT = null, notify: boolean = true) {
		let res = [];
		let dirty = false;
		for (let i = 0; i < this.selected.length; i++) {
			if (this.selected[i] === leaveSelected) {
				res.push(this.selected[i]);
			} else {
				this.selected[i].setSelectState(SelectState.None);
				dirty = true;
			}
		}
		this.selected = res;
		const isSelectionEmpty = this.selected.length === 0;
		if (isSelectionEmpty) {
			this.onSelectUnsubscribes.forEach((f) => f());
		}
		if (dirty && notify) {
			this.notifySelectionChangedListeners();
		}
		return dirty;
	}

	addSelectable(s: SelectableT) {
		const target = s.selectTarget();
		target.interactive = true;
		const callback = this.createOnPointerDownInside(s);
		// Initialize to known state.
		s.setSelectState(SelectState.None);
		target.on('pointerdown', callback);
		this.unsubscribes.push(() => {
			// TODO: do this better.
			// Hm.... doing this will keep a reference to dragTarget which isn't
			// ideal. We want to be able to delete objects and not have them lying
			// around. We should probably have a method removeSelectable() where
			// we slice it out of a list of objects.
			target.off('pointerdown', callback);
		});
	}

	addDropReceiver(receiver: DropReceiver) {
		this.dropReceivers.push(receiver);
		receiver.dropTarget().interactive = true;
	}

	// TODO
	// removeDropReciver(receiver : DropReceiver) {
	// }

	/**
	 * Selection behaviour begins here. 
	 *
	 * Until one of these callbacks is called, there will be no other listeners
	 * registered to anything. There is one callback created for each selectable
	 * that is added. See `addSelectable()`.
	 */
	createOnPointerDownInside(target: SelectableT) {
		return (event: PIXI.InteractionEvent) => {
			// No matter what button, we will close any open context menus.
			this.hideContextMenu();
			console.log(`pointer down (selection manager ${this.mimetype})`);
			if (event.data.button === 1) {
				// Middle-click.
				// Ignore. We want the viewport to handle it as a pan.
				return;
			}

			// Right-click and left-click.
			// I didn't think there was any propagation/bubbling, but it 
			// seems as event.stopPropagation() is needed to prevent the
			// selection from being cleared by the viewport listener.
			event.stopPropagation();
			this.focused = target;
			this.addSelection(target, event.data.originalEvent.ctrlKey);

			if (event.data.button === 2) {
				// Right-click.
				this.showContextMenu(event);
			} else {
				// Left-click.
				//   - double-click
				//   - drag initiation
				if (!(event.data.button === 0)) {
					throw new Error(`Unexpected button ${event.data.button}`);
				}
				const pointerPos = event.data.getLocalPosition(this.positionParent);
				const clickTimeMs = Date.now();
				// Possibly handle double click.
				if (clickTimeMs - this.lastClickTimeMs < SelectionManager.DOUBLE_CLICK_MS) {
					// Wait until up click. Most other editors seem to do this. One thing
					// it avoids is any click events that are registered in the 
					// _onDoubleSelectEnter callback being called immediately, which is
					// undesirable, as a click event is probably used to leave the double
					// click focus mode.
					const onPointerUp = () => {
						// Remove listener.
						target.selectTarget().off('pointerup', onPointerUp);
						this._onDoubleClick(target);
					};
					this._onPreDoubleClick(target);
					target.selectTarget().on('pointerup', onPointerUp);
				}
				this.lastClickTimeMs = clickTimeMs;
				this.onDragStart(pointerPos);
			}
		};
	}

	onDragStart(dragStartPos: PIXI.Point) {
		this.dragStartPosition = dragStartPos;
		this.dragObjStarts = [];
		this.dragTargets = [];
		for (let i = 0; i < this.selected.length; i++) {
			const dragTarget = this.selected[i].onDragStart();
			this.dragTargets.push(dragTarget);
			// Do this before asking for positioning, else it has no parent.
			if (dragTarget.mode === DragMode.DragDrop) {
				this.positionParent.addChild(dragTarget.obj);
				dragTarget.obj.position = this.positionParent.toLocal(
					this.selected[i].selectTarget().getGlobalPosition());
			}
			// temp
			const gpos = dragTarget.obj.getGlobalPosition()
			const pos = this.positionParent.toLocal(gpos);
			this.dragObjStarts.push(pos);
			dragTarget.onDragStateChange(DragState.OverParent);
		}
		if (this.selected.length !== this.dragTargets.length) {
			throw new Error('SelectionManager: a drag target for each selected item '
				+ 'should have been added.');
		}
		this.addDropListeners();
		this.addParentDragListeners();
	}


	/**
	 * Called after drag initiation to attach listeners to the position parent.
	 */
	addParentDragListeners() {
		/**
			pointerdown on a selected object in the presence of multiple selected
			objects shouldn't clear the selection until pointerup, as we need
			the pointerdown to be used for dragging. This means we need to wait
			until a move happens to determine if any selection should be cleared.
			The below callback is added when dragging is initialized but removed
			once a movement occurs.
		*/
		const onNoDrag = (event: PIXI.InteractionEvent) => {
			console.log("onNoDrag");
			this.addSelection(this.focused,
				event.data.originalEvent.ctrlKey,
				event.data.originalEvent.shiftKey,
				true);
			this.clearDrag();
		};
		// It's a one time event.

		const onDragMove = (event: PIXI.InteractionEvent) => {
			// An actual drag has happen, so remove the listener for shortcircuiting 
			// a drag/drop if no movement actually happened. 
			this.positionParent.off('pointerup', onNoDrag);
			if (this.selected.length !== this.dragTargets.length) {
				throw new Error('SelectionManager: there should be as many drag targets '
					+ 'as there are selected objects.');
			}
			// Skip if selection is empty.
			if (!this.dragTargets.length) {
				return;
			}
			//const newPosition = event.data.global;
			const newPosition = event.data.getLocalPosition(this.positionParent);
			const moveVector = newPosition.subtract(this.dragStartPosition);
			// toLocal(position, <what?>, <point to store result>)
			for (let i = 0; i < this.dragTargets.length; i++) {
				//const dragTarget =this.selected[i].selectTarget();
				this.dragTargets[i].obj.position = this.dragObjStarts[i].add(
					moveVector);
			}
		};

		const onDragEnd = () => {
			this.clearDrag();
		};

		// On the avoidance of using "once".
		// One time events can be added like so:
		// 		displayObject.once('eventlabel', () => {/* do something */});
		// Because there are multiple events channelled to onDragEnd, once is
		// not sufficient for clean up, as only one of the listeners (the one that
		// was triggered) will be cleaned up. At the time of writing, we could use
		// a "once" for onNoDrag; however, it is a bit brittle as I might not 
		// notice in future that the once behaviour relied on the listener only
		// having one triggering event type.
		this.positionParent.on('pointermove', onDragMove);
		this.positionParent.on('pointerup', onNoDrag);
		this.positionParent.on('pointerup', onDragEnd);
		this.positionParent.on('pointerupoutside', onDragEnd);
		this.dragUnsubscribes.push(() => {
			this.positionParent.off('pointermove', onDragMove);
			this.positionParent.off('pointerup', onNoDrag);
			this.positionParent.off('pointerup', onDragEnd);
			this.positionParent.off('pointerupoutside', onDragEnd);
		});
	}

	/**
	 * Called after drag initiation to attach listeners to all drop receivers.
	 */
	addDropListeners() {
		for (const receiver of this.dropReceivers) {
			const __this = this;
			const onPointerUp = (event: PIXI.InteractionEvent) => {
				console.log("pointerup");
				if (__this.isDragging()) {
					const pointerPos = event.data.getLocalPosition(this.positionParent);
					receiver.onDrop(__this.selected, __this.mimetype, pointerPos);
					this.clearDrag();
				}
			}
			const onPointerOver = (_event: PIXI.InteractionEvent) => {
				console.log("pointerover");
				if (__this.isDragging()) {
					console.log("entering ", receiver);
					receiver.onDragEnter(__this.selected, __this.mimetype);
					// Notes on drag-drop ghosts.
					// These next 3 lines are commented out to keep a note of how this
					// approach is not going to work in its current state.
					// The motivation is to allow a "New group" animation to be shown
					// when drag targets are over the viewport only. However, each drag
					// target doesn't know about how many drag targets there are, so
					// they cannot be in charge of making this ghost object. It could
					// instead be done in a mousemove listener of a drop receiver.
					// An alternative is for DragTargets of a drag-drop object to have
					// an interface method like addSibling(). Thus, only the first 
					// drag object provieds a DragTarget, and all subsequent dragTargets
					// are added to this object. Then, this single drag target can 
					// decide how to display itself based on how many objects are inside.
					//
					// for(const dragTarget of __this.dragTargets) {
					//		dragTarget.onDragStateChange(DragState.OverDropReceiver);
					//}
				}
			}
			const onPointerOut = (_event: PIXI.InteractionEvent) => {
				console.log("pointerout");
				if (__this.isDragging()) {
					receiver.onDragLeave();
				}
			}
			receiver.dropTarget().on('pointerup', onPointerUp);
			receiver.dropTarget().on('pointerover', onPointerOver);
			receiver.dropTarget().on('pointerout', onPointerOut);
			this.dragUnsubscribes.push(() => {
				receiver.dropTarget().off('pointerup', onPointerUp);
				receiver.dropTarget().off('pointerover', onPointerOver);
				receiver.dropTarget().off('pointerout', onPointerOut);
			});
		}
	}

	showContextMenu(event: PIXI.InteractionEvent) {
		if (this.isContextMenu) {
			throw Error("Context menu already being shown.");
		}
		this._onContextMenuHide = this._onContextMenuShow(event, this.selected);
		this.isContextMenu = true;
	}

	hideContextMenu() {
		if (!this.isContextMenu) {
			return;
		}
		this._onContextMenuHide();
		this.isContextMenu = false;
	}

	onContextMenuShow(fn: (e: PIXI.InteractionEvent, s: SelectableT[]) => (() => void)) {
		this._onContextMenuShow = fn;
	}
}

class TimeControl {
	// Appearance
	width = 400;
	height = 70;
	// Total left right padding will be padding[0]*2.
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


	constructor(snippetSecs: number, padSecs: number, playbackTimer: any) {
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
		x = x - this.padding[0];
		x = Math.max(x, 0);
		x = Math.min(x, this.innerWidth);
		const rel = x / this.innerWidth;
		return rel;
	}

	timeString() {
		const timeSinceSpike = this.time - this.spikeTime;
		return `t = ${timeSinceSpike.toFixed(3)} s`;
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
		return (_e: PIXI.InteractionEvent) => {
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

export class Snippet implements Selectable {
	static defaultSquareLen = 10;
	static selectBorderWidth = 1;
	static padFactor = 0.05;
	// Data
	// The recording ID, cell ID and the sample at which the spike occured
	// identify the snippet.  
	recId: number;
	cellId: number;
	spikeIdx: number;
	// The current index within the snippet to display.
	snipIdx: number;
	isDragging: boolean = false;
	selectState: SelectState = SelectState.None;
	// The display objects.
	container: PIXI.Container;
	rect1: PIXI.Graphics;
	squareLen: number;
	// This is saved and called on destroy() so that we can clean up
	// references to this object when it's removed from the scene. It's
	// the unsubscribe for the Svelte store that tracks the snippet idx.
	idxUnsubscribe: () => void;
	// Something like this to check if the snippet has been 
	// placed on the canvas or is still as an outline while
	// being placed.
	// isPlaced : boolean;
	// We currently need a backref for the drop behaviour.
	groupBackRef: Group = null;

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
		this.container.interactive = false;
		this.rect1 = new PIXI.Graphics();
		this.container.addChild(this.rect1);
		this.snipIdx = 0;
		this.rebuild();
		//this.container.hitArea = new PIXI.Rectangle(
		//		0, 0, this.container.width, this.container.height);
	}

	setSelectState(state: SelectState) {
		if (this.selectState == state) {
			return;
		}
		this.selectState = state;
		this.rebuild();
	}

	onDragStart(): DragTarget {
		// Make a copy with slight transparency and border highlight.
		this.isDragging = true;
		this.rebuild();
		return this.createDragGhost();
	}

	onDragEnd() {
		console.log("onDragEnd actual");
		this.isDragging = false;
		this.rebuild();
	}

	createDragGhost(): DragTarget {
		// Make a copy with slight transparency and border highlight.
		const dragGhost = new PIXI.Container();
		const rect1 = new PIXI.Graphics();
		const color = this.color();
		rect1.beginFill(color);
		const pad = this.squareLen * Snippet.padFactor;
		rect1.lineStyle(pad, 0xffffff, 1, 0.5);
		rect1.drawRect(pad, pad,
			this.squareLen - pad * 2,
			this.squareLen - pad * 2);
		rect1.endFill();
		dragGhost.addChild(rect1);
		return new DragTarget(dragGhost, DragMode.DragDrop);
	}

	selectTarget() {
		return this.container;
	}

	rebuild() {
		// Clear and rebuild square.
		this.rect1.clear();
		const pad = this.squareLen * Snippet.padFactor;
		// Border
		if (this.isDragging || this.selectState === SelectState.Selected) {
			this.rect1.lineStyle(
				{
					width: pad,
					color: 0xffffff,
					// Allignment (0 = inner, 0.5 = middle, 1 = outer)
					alignment: 0.5
				}
			);
		} else {
			this.rect1.lineStyle(0);
		}
		// Fill color
		const color = this.isDragging ? 0x000000 : this.color();
		this.rect1.beginFill(color);
		this.rect1.drawRect(pad, pad,
			this.squareLen - pad * 2,
			this.squareLen - pad * 2);
		this.rect1.endFill();
	}

	onSnippetIdxUpdate() {
		let __this = this;
		return (idx: number) => {
			__this.snipIdx = idx;
			__this.rebuild();
		};
	}


	stimToShow(): number[] | null {
		const absIdx = absSnippetIdx(this.spikeIdx, this.snipIdx);
		let res: number[];
		if (absIdx < 0) {
			res = null;
		} else {
			res = recsById.get(this.recId).stimulus[absIdx];
		}
		return res;
	}

	color(): number {
		const colorAsNumber = colorutils.rgbToNumeric(this.colorRGB());
		return colorAsNumber;
	}

	colorRGB(): number[] {
		let stim = this.stimToShow();
		if (stim === null) {
			stim = [0, 0, 0, 0];
		}
		const sRGB = colorutils.stimToSRGB(stim);
		return sRGB;
	}
}

class GroupViewMode {
	static EditThis = new GroupViewMode("EditThis");
	static EditOther = new GroupViewMode("EditOther");
	static View = new GroupViewMode("View");

	name: string;

	constructor(name: string) {
		this.name = name;
	}

	toString() {
		return this.name;
	}
}

export class Group implements Selectable, DropReceiver {
	// Initial shape is square. 
	static widthOnCreation = 150;
	static heightOnCreation = 150;
	static padding = 3;
	static maxSquareSize = 50;
	static selectBorderWidth = 1;
	// Colors inspired from: https://reasonable.work/colors/
	static groupColors = [
		0x302100, // amber
		0x510018, // raspberry
		0x002812, // emerald
		0x44003c, // magenta
		0x001d2a, // cerulean
		0x292300, // yellow
	];
	static dragCandidateColor = 0x72ff6c;
	static fontSize = 8;

	title: string;
	snippets: Snippet[];
	container: PIXI.Container;
	header: PIXI.Container;
	gRect: PIXI.Graphics;
	blurredBg: PIXI.Graphics;
	color: number;
	width: number;
	height: number;
	// Selection, drag & drop and resize state.
	selectState: SelectState = SelectState.None;
	isDragging: boolean = false;
	viewMode: GroupViewMode = GroupViewMode.View;
	isDragCandidate = false;
	// Set to true to have "(edited)" appended to group title.
	edited: boolean = false;

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

	/**
	 * Remove a snippet from this group.
	 * 
	 * @param snip - the snippet to remove.
	 * @param destroy - whether to destroy the snippet being removed. The use of
	 * 		setting this to false is to allow the transfer of snippets between 
	 *		groups.
	 * @param rebuild - whether to rebuild the group. The use of setting this to
	 * 		false is to allow batching of changes before calling rebuild.
	 */
	removeSnippet(snip: Snippet, destroy = true, rebuild = true) {
		const index = this.snippets.indexOf(snip);
		if (index < 0) {
			return;
		}
		this.snippets.splice(index, 1);
		this.container.removeChild(snip.container);
		if (rebuild) {
			console.log("Removing snippet, rebuild");
			this.rebuild();
		}
		if (destroy) {
			snip.destroy();
		}
	}

	onDrop(objects: any[], mimetype: string, _pointerPos: PIXI.Point) {
		// For the moment at least, dropping will trigger the same dragLeave
		// display changes, so just call it.
		this.onDragLeave();
		if (mimetype !== 'snippet') {
			return;
		}
		const snippets = objects as Snippet[];
		if (snippets.length === 0) {
			throw new Error("Dropped an empty array.");
		}
		const origGroup = snippets[0].groupBackRef;
		if (origGroup === this) {
			// Nothing to do, dropped on the orig group.
			return;
		}
		const toAdd: Snippet[] = [];
		for (const snip of snippets) {
			if (snip.groupBackRef !== origGroup) {
				throw new Error("Expected all snippets to come from one group.");
			}
			toAdd.push(snip);
			const destroy = false;
			const rebuild = false;
			origGroup.removeSnippet(snip, destroy, rebuild);
			// If we are receiving the snippet, then we are not supposed to be 
			// interactive. The fact that this behaviour must be controlled here
			// suggests that the onDrop handler should be done at a higher level,
			// such as Workspace, and that the onDrop callback also receives the
			// target dropReceiver. So, TODO: consider a refactor.
			snip.container.interactive = false;
		}
		this.edited = true;
		origGroup.edited = true;
		// Now we can rebuild the original group.
		origGroup.rebuild();
		// Add snippets will do a rebuild.
		this.addSnippets(toAdd);
	}

	onDragEnter(_objects: any[], mimetype: string) {
		if (mimetype !== 'snippet') {
			return;
		}
		console.log("onDragEnter");
		const dirty = !this.isDragCandidate;
		this.isDragCandidate = true;
		if (dirty) {
			this.rebuild();
		}
	}

	onDragLeave() {
		console.log("onDragLeave");
		const dirty = this.isDragCandidate;
		this.isDragCandidate = false;
		if (dirty) {
			this.rebuild();
		}
	}

	dropTarget(): PIXI.DisplayObject {
		return this.container;
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
		this.blurredBg = new PIXI.Graphics();
		// Filters require WebGL which isn't working on NVidia+Ubuntu+Firefox at the 
		// moment, it seems. Anyway, whithout antialiasing, it looks pretty bad,
		// so lets just stick to a square boarder for now.
		// this.blurredBg.filters = [new PIXI.filters.BlurFilter()];
		// Add the blurred background first, behind.
		this.container.addChild(this.blurredBg);
		this.container.addChild(this.gRect);
		this.header = Group.addHeader(this.container, this.title);
		this.width = Group.widthOnCreation;
		this.height = Group.heightOnCreation;
		this.rebuild();
	}

	static addHeader(container: PIXI.Container, text: string) {
		// The text objects are made once, so make them big and scale them
		// down so as to give better resolution when zooming in. Larger text
		// is more memory and cpu intensive.
		const scale = 5
		const size = Group.fontSize * scale
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
		const isSelected = this.selectState === SelectState.Selected;
		const invalidState = isSelected && this.isDragCandidate;
		if (invalidState) {
			// Throw exception
			throw new Error("Can't be both selected and have drag objects over " +
				"the container.");
		}
		if (isSelected) {
			this.gRect.lineStyle(
				{
					width: Group.selectBorderWidth,
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
		// Drop receiving
		if (this.isDragCandidate) {
			this.container.alpha = 1.0;
			this.gRect.alpha = 1.0;
			this.blurredBg.clear();
			this.blurredBg.beginFill(Group.dragCandidateColor);
			this.blurredBg.drawRect(-1, -1, this.width + 2, this.height + 2);
			this.blurredBg.endFill();
		} else {
			this.blurredBg.clear();
			this.container.alpha = this.viewMode == GroupViewMode.EditOther ? 0.7 : 1.0;
			this.gRect.alpha = this.viewMode == GroupViewMode.EditThis ? 0.7 : 1.0;
		}

		// Switch on view mode
		// switch (this.viewMode) {
		// 	case GroupViewMode.EditThis:
		// 		this.gRect.alpha = 0.7;
		// 		break;
		// 	case GroupViewMode.EditOther:
		// 		this.container.alpha = 0.7;
		// 		break;
		// 	default:
		// 		this.container.alpha = 1.0;
		// 		this.gRect.alpha = 1.0;
		// }
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
			snippets[i].groupBackRef = this;
		}
		this.rebuild();
	}

	setSelectState(state: SelectState) {
		if (this.selectState == state) {
			return;
		}
		this.selectState = state;
		this.rebuild();
	}

	onDragStart(): DragTarget {
		this.isDragging = true;
		return new DragTarget(this.container, DragMode.DragToMove);
	}

	onDragEnd() {
		this.isDragging = false;
	}

	selectTarget() {
		return this.container;
	}

	setViewMode(mode: GroupViewMode) {
		if (this.viewMode == mode) {
			return;
		}
		// Don't disable interactivity on the group, as we use them as drop targets.

		// Enable/Disable interactivity for child snippets.
		for (const snip of this.snippets) {
			// interactive iff editing
			snip.container.interactive = (mode == GroupViewMode.EditThis);
		}
		this.viewMode = mode;
		this.rebuild();
	}
}

export class Workspace implements DropReceiver {
	static margin = 20;
	static numCols = 6;
	groups: Group[];
	openGroup: Group | null = null;
	container: PIXI.Container;
	col: number = 0;
	row: number = 0;
	groupSelectionMgr: SelectionManager<Group> | null;
	snippetSelectionMgr: SelectionManager<Snippet> | null = null;
	canvas: HTMLCanvasElement;
	viewport: Viewport;
	_onDestroy = [];

	// Option
	// The snippet drop onto the viewport might become a bit annoying.
	enableViewportDrop: boolean = true;

	constructor(canvas: HTMLCanvasElement, viewport: Viewport) {
		this.canvas = canvas;
		this.viewport = viewport;
		this.groups = [];
		this.container = new PIXI.Container();
		this.groupSelectionMgr = this.createGroupSelectionMgr([]);
		this.addKeyboardShortcuts();
	}

	destroy() {
		for (const f of this._onDestroy) {
			f();
		}
	}

	addKeyboardShortcuts() {
		const handler = (event: KeyboardEvent) => {
			const isGroupOpen = this.openGroup != null;
			if (isGroupOpen) {
				// Keybindings for when a group is open.
				if (event.key.toLowerCase() == "r") {
					if (event.shiftKey) {
						this.selectSnippetsByOnLED(0, true);
					} else {
						this.selectSnippetsByOnLED(0, false);
					}
				} else if (event.key.toLowerCase() == "g") {
					if (event.shiftKey) {
						this.selectSnippetsByOnLED(1, true);
					} else {
						this.selectSnippetsByOnLED(1, false);
					}
				} else if (event.key.toLowerCase() == "b") {
					if (event.shiftKey) {
						this.selectSnippetsByOnLED(2, true);
					} else {
						this.selectSnippetsByOnLED(2, false);
					}
				} else if (event.key.toLowerCase() == "u") {
					if (event.shiftKey) {
						this.selectSnippetsByOnLED(3, true);
					} else {
						this.selectSnippetsByOnLED(3, false);
					}
				} else if (event.key.toLowerCase() == "s") {
					if (event.shiftKey) {
						// nothing
					} else {
						this.selectSnippetsSimimlarToSelected();
					}
				}
			}
		};
		const removeHandler = () => {
			window.removeEventListener('keydown', handler);
		};
		this._onDestroy.push(removeHandler);
		removeHandler();
		window.addEventListener('keydown', handler);
	}

	/**
	 * Create the selection manager than handles group movement.
	 * 
	 * This function and the next are part of a predicament: when to enable and
	 * disable interactivity for the group and snippet PIXI containers. 
	 * 
	 * The SelectionManager can make a good case that it should _enable_ the
	 * interactivity of the select and drop target containers when Selectables and
	 * DropReceivers are added via addSelectable() and addDropReceiver(). At the
	 * moment, this is what is done; but it's only _enable_ of needed interactivity 
	 * and not _disable_ of unneeded interactivity that is carried out. The
	 * SelectionManager makes no effort to revert the selection behaviour after
	 * its own destruction, so effort must be made somewhere to reset interactive
	 * state to something known (all off?) when a selection manager is destroyed.
	 */
	createGroupSelectionMgr(initialGroups: Group[]): SelectionManager<Group> {
		const manager = new SelectionManager<Group>(
			this.canvas, this.viewport, 'group');
		for (const group of initialGroups) {
			manager.addSelectable(group);
		}
		manager.addSelectionChangedListener(onGroupSelectionChanged);
		manager.onDelete((selected: Group[]) => {
			console.log("Delete group!");
			for (let i = 0; i < selected.length; i++) {
				workspace.deleteGroup(selected[i]);
			}
		});
		manager.onDoubleClick((selected: Group) => {
			// Enter a special "single group edit" mode.
			// Stop using this selection manager and transition to the snippet 
			// selection manager. Disable all interactivity too. The snippet manager
			// will reenable what it needs.
			manager.destroy();
			this.groupSelectionMgr = null;
			this.disableAllInteractivity();
			this.snippetSelectionMgr = this.createSnippetSelectionMgr(selected);
			selectionMode.set("snippet");
		});
		return manager;
	}

	createSnippetSelectionMgr(selectedGroup: Group): SelectionManager<Snippet> {
		const manager = new SelectionManager<Snippet>(
			this.canvas, this.viewport, 'snippet');
		// Enable interativity for the selected group so that it captures clicks
		// and does nothing with them. We don't want clicks within this group to
		// exit it's own edit mode.
		// Without registering a callback, the parent will receive it, even if
		// the click is over the child.
		const emptyCallback = (event: PIXI.InteractionEvent) => {
			// Only catch left-click. We still want to be able to pan with mouse over
			// a group or snippet, and are not in the middle of dragging.
			// If we are dragging, then we don't want to get stuck in 
			// drag mode if we drop back on ourself.
			if (event.data.button == 0 && !manager.isDragging()) {
				// I didn't think there was any propagation/bubbling, but it 
				// seems as event.stopPropagation() is needed here.
				event.stopPropagation();
			}
		};
		selectedGroup.setViewMode(GroupViewMode.EditThis);
		this.openGroup = selectedGroup;
		// Set mode of remaining groups.
		for (const group of this.groups) {
			if (group != selectedGroup) {
				group.setViewMode(GroupViewMode.EditOther);
			}
		}
		selectedGroup.container.interactive = true;
		selectedGroup.selectTarget().on('pointerdown', emptyCallback);
		selectedGroup.selectTarget().on('pointerup', emptyCallback);

		for (const snippet of selectedGroup.snippets) {
			manager.addSelectable(snippet);
		}
		// Accept drops onto the viewport. Slightly experimental feature. 
		// Probably a bit buggy. It might be best to add a method like
		// manager.setParentDropFallback(fn). The issue with treating the viewport
		// like a normal drop target is that the pointer never enters it, as it
		// never left. And so, something like a ghost sprite for when over the
		// viewport only is hard to implement.
		if (this.enableViewportDrop) {
			manager.addDropReceiver(this);
		}
		const isOriginGroupATarget = this.enableViewportDrop;
		for (const group of this.groups) {
			if (group === selectedGroup) {
				if (!isOriginGroupATarget) {
					continue;
				}
			}
			console.log("adding drop receiver:", group.title);
			manager.addDropReceiver(group);
		}
		manager.onDelete((selected: Snippet[]) => {
			console.log("Delete snippet!");
			for (let i = 0; i < selected.length; i++) {
				selectedGroup.removeSnippet(selected[i]);
			}
		});
		manager.onPreDoubleClick((selected: Snippet) => {
			console.log("double click on stimulus");
			// Not used anymore. Was previously used for filtering, but that is now
			// done by context menu and keyboard shortcuts.
		});

		/*manager.onRightClick((selected: Snippet[]) => {
			console.log("right click on stimulus");
			const selected0 = selected[0];
			const selectedStim = selected0.stimToShow();
			const idxOfMax = selectedStim.indexOf(Math.max(...selectedStim));
			const threshold = 0.3
			const dist = (a: number, b: number) => Math.abs(a - b);
			for (const snip of selectedGroup.snippets) {
				if (snip != selected0) {
					const otherStim = snip.stimToShow();
					if (otherStim !== null && dist(selectedStim[idxOfMax],
						otherStim[idxOfMax]) < threshold) {
						//console.log("Stim close: ", selectedStim, " vs. ", otherStim,
						//				  ", dist: ", dist(selectedStim, otherStim));
						// A bit hacky here, but the selection is the "ctrl" behaviour, and
						// that's the addSelection() interface at the moment. TODO: refactor.
						const ctrlKey = true;
						manager.addSelection(snip, ctrlKey);
					}
				}
			}
		});
		*/
		// Prepair leave methods.
		const leaveGroupEdit = () => {
			this.viewport.off('pointerdown', leaveGroupEditByMouse);
			this.viewport.off('keydown', leaveGroupEditByButton);
			this.openGroup = null;
			selectedGroup.selectTarget().off('pointerdown', emptyCallback);
			selectedGroup.selectTarget().off('pointerup', emptyCallback);
			// Change the display mode of the groups back to View. 
			for (const group of this.groups) {
				group.setViewMode(GroupViewMode.View);
				group.setSelectState(SelectState.None);
			}
			// Distroy the snippet selection manager and clear interactivity.
			manager.destroy();
			this.snippetSelectionMgr = null;
			this.disableAllInteractivity();
			// Remake the group selection manager.
			this.groupSelectionMgr = this.createGroupSelectionMgr(this.groups);
			selectionMode.set("group");
		};
		const leaveGroupEditByButton = (event) => {
			if (event.key == 'Escape') {
				leaveGroupEdit();
			}
		};
		const leaveGroupEditByMouse = (event: PIXI.InteractionEvent) => {
			// Only take action on left click.
			console.log("leave group edit on pointer down");
			if (!(event.data.button === 0)) {
				return;
			}
			leaveGroupEdit();
		};
		this.viewport.on('pointerdown', leaveGroupEditByMouse);
		this.viewport.on('keydown', leaveGroupEditByButton);
		return manager;
	}

	disableAllInteractivity() {
		for (const group of this.groups) {
			group.container.interactive = false;
			for (const snippet of group.snippets) {
				snippet.container.interactive = false;
			}
		}
	}

	addGroup(group: Group) {
		this.groups.push(group);
		const nextX = Group.widthOnCreation + this.col *
			(Group.widthOnCreation + Workspace.margin);
		const nextY = Group.heightOnCreation + this.row *
			(Group.heightOnCreation + Workspace.margin);
		this.col++;
		if (this.col > Workspace.numCols) {
			this.col = 0;
			this.row++;
		}
		group.container.x = nextX;
		group.container.y = nextY;
		this.container.addChild(group.container);
		if (this.groupSelectionMgr !== null) {
			this.groupSelectionMgr.addSelectable(group);
		} else if (this.snippetSelectionMgr !== null) {
			this.snippetSelectionMgr.addDropReceiver(group);
			group.setViewMode(GroupViewMode.EditOther);
		}
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

	/**
	 * [begin] DropReceiver interface
	 * 
	 * This can all be removed if dropping snippets on viewport to create a new
	 * group gets annoying and is abandoned.
		 */
	onDrop(objects: any[], mimetype: string, pointerPos: PIXI.Point) {
		console.log("drop on workspace");
		if (mimetype === "snippet") {
			const snippets = objects as Snippet[];
			if (snippets.length === 0) {
				return;
			}
			const group = new Group("Custom group");
			const origGroup = snippets[0].groupBackRef;
			for (const snip of snippets) {
				const destroy = false;
				const rebuild = false
				origGroup.removeSnippet(snip, destroy, rebuild);
				snip.container.interactive = false;
			}
			origGroup.rebuild();
			group.addSnippets(snippets);
			this.addGroup(group);
			group.container.position = pointerPos;
		}
	}

	dropTarget(): PIXI.DisplayObject {
		// Which one?
		//return this.container;
		return this.viewport;
	}

	onDragEnter(objects: any[], mimetype: string): void {
		// This is only called when actually leaving the viewport and back again.
		console.log("drag enter viewport");
	}
	onDragLeave(): void {
		console.log("drag enter viewport");
	}
	/**
	 * [end] DropReceiver interface
	 */

	/**
	 * [Begin] Snippet selection routines.
	 * 
	 * Selection behaviours like:
	 *   - select all that are currently red
	 *   - select all that follow a spike select 
	 * 
	 * These methods are called by the context menu, and are also tied to 
	 * keyboard shortcuts.
	 */

	/**
	 * Select snippets that currently have the given stimumlus component on.
	 * 
	 * Snippets are selected from the snippets of the open group. It's an error 
	 * to call this method when a group is not open.
	 *
	 * @param ledIdx - the stimulus component to select. This is a value from 0 
	 * 	to 3 for a 4-LED stimulus.
	 * @param exclusive - wheather other LEDs must be off.
	 */
	selectSnippetsByOnLED(ledIdx: number, exclusive: boolean) {
		if (this.snippetSelectionMgr == null) {
			throw new Error("Cant' select snippets: no snippet selection manager.");
		}
		if (ledIdx < 0 || ledIdx >= 4) {
			throw new Error("Caller error: invalid color idx. Only 4 colors supported.");
		}
		if (this.openGroup == undefined || this.openGroup == null) {
			throw new Error("Unexpected error: no open group.");
		}
		// Currently, we assume full on-off stimulus. While full on means
		// stimulus value == 1, the downsampling means that the stimulus value
		// transitions from 0 through intermediate values to 1 and then 
		// oscillates a bit. With this in mind, we use a <1 threshold to filter
		// in the on stimuli.
		const onThreshold = 0.5;
		const filtered = this.openGroup.snippets.filter((s) => {
			const stim = s.stimToShow();
			if (stim === null) {
				return false;
			}
			const match = stim.every((v, i) => {
				if (i === ledIdx) {
					return v >= onThreshold;
				} else {
					return exclusive ? v < onThreshold : true;
				}
			});
			return match;
		});
		this.snippetSelectionMgr.setSelection(filtered);
	}

	/**
	 * Select snippets that currently have a color similar to the selected snippet.
	 * 
	 * Snippets are selected from the snippets of the open group. It's an error 
	 * to call this method when a group is not open.
	 * 
	 * Similarity is euclidean distance between the LED values.
	 */
	selectSnippetsSimimlarToSelected() {
		if (this.snippetSelectionMgr == null) {
			throw new Error("Can't select snippets: no snippet selection manager.");
		}
		const selectedSnippets = this.snippetSelectionMgr.selected;
		if (selectedSnippets == null || selectedSnippets.length == 0) {
			throw new Error("Can't select similar snippets: no snippet selected.");
		}
		if (selectedSnippets.length !== 1) {
			throw new Error(`Can't select similar snippets: more than 1 snippet 
											selected.`);
		}
		const threshold = 0.3;
		function dist(arrayA: number[], arrayB: number[]) {
			return arrayA.map((v, i) => Math.pow(v - arrayB[i], 2))
				.reduce((acc, v) => acc + v);
		}
		if (this.openGroup == undefined || this.openGroup == null) {
			throw new Error(`Unexpected error: no open group.`);
		}
		// Check all other snippets for stimulus value proximity.
		const selectedStim = selectedSnippets[0].stimToShow();
		const toSelect = this.openGroup.snippets.filter((s) => {
			const stim = s.stimToShow();
			if (stim === null) {
				return false;
			}
			const match = dist(selectedStim, stim) < threshold;
			return match;
		});
		this.snippetSelectionMgr.setSelection(toSelect);
	}
	/**
	 * [End] Snippet selection routines.
	 */
}

export let workspace = null;

export async function addCellAsGroup(recId: number, cellId: number, cellSelectMode: string) {
	// Preload the stimulus data.
	const rec = recsById.get(recId);
	await rec.fetchStimulus();
	// TODO: hardcoded mindist=100, which is tied to downsample rate.
	const url_all = `/api/recording/${recId}/cell/${cellId}/spikes`;
	const url_nooverlap_first = `/api/recording/${recId}/cell/${cellId}/spikes?mindist=100&mode=first`;
	const url_nooverlap_other = `/api/recording/${recId}/cell/${cellId}/spikes?mindist=100&mode=allButFirst`;
	let url = null;
	switch (cellSelectMode) {
		case "all":
			url = url_all;
			break
		case "first":
			url = url_nooverlap_first;
			break
		case "allButFirst":
			url = url_nooverlap_other;
			break
		default:
			throw new Error(`Unknown cellSelectMode: ${cellSelectMode}`);
	}
	const spikes = await fetch(url)
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

function onGroupSelectionChanged(selected: Selectable[]) {
	groupSelection.setGroups(selected);
}

// Note: it may be the case that we only want these triggered when the canvas
// has focus. In that case, they should be registered by the workspace.
function addKeyControls(): () => void {
	// Time controls: forward backwards with arrows.
	const createOnArrowDown = (keycode: string, timeUpdateFn: (t: number) => number) => {
		const res = (event: KeyboardEvent) => {
			if (event.code !== keycode) {
				return;
			}
			// Pause and update time.
			pauseCtrl.pause();
			playbackTime.update(curr => timeUpdateFn(curr));
		};
		return res;
	};
	const jumpSecs = 10 / 1000;
	const onLeftArrow = createOnArrowDown(
		"ArrowLeft", (curr: number) => Math.max(0, (curr - jumpSecs)));
	const onRightArrow = createOnArrowDown(
		"ArrowRight", (curr: number) => Math.min(snippetDurationSecs, (curr + jumpSecs)));
	window.addEventListener('keydown', onLeftArrow);
	window.addEventListener('keydown', onRightArrow);
	const unsub = () => {
		window.removeEventListener('keydown', onLeftArrow);
		window.removeEventListener('keydown', onRightArrow);
	};
	return unsub;
}

export function startEngine(canvas: HTMLCanvasElement,
	app: PIXI.Application, viewport: Viewport): Workspace {
	console.log("Init workspace.");
	workspace = new Workspace(canvas, viewport);
	viewport.addChild(workspace.container);
	app.ticker.add(ontick);
	// Outer container
	const layout = new MainLayout(
		app,
		viewport,
		new TimeControl(snippetDurationSecs, padDurationSecs, playbackTime));
	app.stage.addChild(layout.container);
	const unsubs = addKeyControls();
	return workspace;
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


function deriveObject(obj: any) {
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
