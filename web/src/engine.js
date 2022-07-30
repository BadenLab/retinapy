import { getContext, onMount, onDestroy } from 'svelte';
import { writable, derived, readable, get } from 'svelte/store';

// Some props for the app
export const width = writable(window.innerWidth);
export const height = writable(window.innerHeight);
export const pixelRatio = writable(window.devicePixelRatio);
export const context = writable();
export const canvas = writable();
export const time = writable(0);

//export const snippets = writable([]);
//export const playback_speed = writable(0.05);
//export const sample_rate = readable(99.182);
//export const snippet_len = readable(120);
//export const snippet_pad = readable(20);
//export const is_paused = writable(false);
//export const playback_time = writable(0);
//export const sample_idx = derived(
//	[playback_time, sample_rate, snippet_len], 
//	([p, sr, sl]) => cur_sample(p, sr, sl));

// A more convenient store for grabbing all game props
export const props = deriveObject({
	context,
	canvas,
	width,
	height,
	pixelRatio,
	time
});

let _playback_time = 0;
let _time = 0;
let _is_paused = false;
let _playback_speed = 0.05;
let _sample_rate = 99.182;
let _snippet_len = 120;
let _snippet_pad = 20;
let _snippets = [];

export function update_clock(cur_time, dt) {
	_time = cur_time;
	if(!_is_paused) {
		_playback_time = (_playback_time + dt * _playback_speed) % playback_duration();
	}
}


export function playback_duration() {
	return _snippet_len / _sample_rate;
}

export function snippet_len() {
	return _snippet_len;
}

export function snippet_pad() {
	return _snippet_pad;
}

export function is_paused() {
	return is_paused;
}

export function pause() {
	_is_paused = true;
}

export function unpause() {
	_is_paused = false;
}

export function playback_time() {
	return _playback_time;
}

export function snippet_time() {
	const spike_at = (_snippet_len - _snippet_pad) / _sample_rate;
	return playback_time() - spike_at;
}

export function set_playback_time(t) {
	_playback_time = t;
}

export function set_playback_time_rel(r) {
	// Either here or somewhere else, handle the the fact that modulo will
	// send the end to the begininning.
	r = Math.min(Math.max(0, r), 0.9999999);
	_playback_time = r * _snippet_len / _sample_rate, r;
}



export function sample_idx() {
	const res = Math.floor((_playback_time * _sample_rate) % _snippet_len);
	return res;
}

export function snippets() {
	return _snippets;
}

export function set_snippets(snippets) {
	_snippets = snippets;
}


// Javascript built-in function that returns a unique symbol primitive.
export const key = Symbol();

export const getState = () => {
	const api = getContext(key);
	return api.getState();
};

export const renderable = (render) => {
	const api = getContext(key);
	const element = {
		ready: false,
		mounted: false
	};
	if (typeof render === 'function') element.render = render;
	else if (render) {
		if (render.render) element.render = render.render;
		if (render.setup) element.setup = render.setup;
	}
	api.add(element);
	onMount(() => {
		element.mounted = true;
		return () => {
			api.remove(element);
			element.mounted = false;
		};
	});
}

function deriveObject (obj) {
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

function canvas_pos(global_pos) {
  var rect = get(canvas).getBoundingClientRect();
  var x = (global_pos[0] - rect.left) 
  var y = (global_pos[1] - rect.top) 
  return [x, y]
}

/*export function canvas_pos_from_rel(rel_pos) {
  var rect = canvas.getBoundingClientRect();
  var w = rect.right - rect.left
  var h = rect.bottom - rect.top
  return [rel_pos[0] * h + rel_pos[1] * w]
}*/

export function mouse_pos(event) {
  return canvas_pos([event.clientX, event.clientY])
}

export function mouse_pos_rel(event) {
  return canvas_pos_rel([event.clientX, event.clientY])
}
