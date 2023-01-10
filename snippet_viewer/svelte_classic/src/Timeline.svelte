<script>
import * as engine from './engine.js';

export const top_left = [100, 20];
export const width = 800;
export const height = 40;

const PADDING = [0, 0];
const inner_width = width - PADDING[0];
const inner_height = 40 - PADDING[1];


let is_dragging = false;
let pre_down_state = engine.is_paused();

function pos(px, py) {
	const res = [top_left[0] + px*inner_width + PADDING[0]/2, 
		top_left[1] + py*inner_height + PADDING[0]/2];
	return res;
}

function x_to_rel(x) {
	x = x - (top_left[0] + PADDING[0]/2);
	x = Math.max(x, 0);
	x = Math.min(x, inner_width);
	const rel = x / inner_width;
	return rel;
}

function draw_spike(sample_idx, width, ctx) {
	// Line
	const fraction = sample_idx / engine.snippet_len();
	const line_y = 0.5;
	const time_dot_size = 8;
	const spike_line_height = 20;
	// Dot
	const dot_y = line_y ;
	const dot_radius = 3;
	// Text
	const text_y = dot_y + 0.4;
	const fontFamily = 'sans-serif';
	const fontSize = 12;
	const align = 'center';
	const baseline = 'middle';

	// Draw line
	ctx.strokeStyle = 'white';
	ctx.stroke();
	ctx.beginPath();
	ctx.moveTo(...pos(0, line_y));
	ctx.lineTo(...pos(1, line_y));
	ctx.stroke();

	// Draw spike marker
	ctx.strokeStyle = 'rgb(255, 150, 150)';
	ctx.beginPath();
	const spike_point_rel = 1 - engine.snippet_pad() / engine.snippet_len();
	ctx.moveTo(...pos(spike_point_rel, 0.0));
	ctx.lineTo(...pos(spike_point_rel, 1.0));
	ctx.stroke();

	// Draw moving dot
	ctx.beginPath();
	ctx.fillStyle = 'rgb(255, 255, 255)';
	ctx.strokeStyle = ctx.fillStyle;
	ctx.arc(...pos(fraction, dot_y), dot_radius, 0, 2*Math.PI);
	// Can't get fill working well.
	ctx.fill();

	// Draw moving text.
	ctx.font = `${fontSize}px ${fontFamily}`;
	ctx.textAlign = align;
	ctx.textBaseline = baseline;
	const text = `t = ${engine.snippet_time().toFixed(2)} s`;
	ctx.fillText(text, ...pos(fraction, text_y));
}

function is_inside(mouse_pos) {
	const in_x = mouse_pos[0] > top_left[0] && mouse_pos[0] < top_left[0] + width;
	const in_y = mouse_pos[1] > top_left[1] && mouse_pos[1] < top_left[1] + height;
	return in_x && in_y;
}

function on_down(event) {
	const pos = engine.mouse_pos(event);
	console.log(pos);
	if(is_inside(pos)) {
		event.preventDefault();
		event.stopPropagation();
		pre_down_state = engine.pause_ctrl.current();
		engine.pause_ctrl.pause();
		engine.set_playback_time_rel(x_to_rel(pos[0]));
		is_dragging = true;
	}
}


function on_move(event) {
	if(is_dragging) {
		const pos = engine.mouse_pos(event);
		event.preventDefault();
		event.stopPropagation();
		engine.set_playback_time_rel(x_to_rel(pos[0]));
	}
}

function on_up(event) {
	// Important to check is_dragging before resuming, as another control
	// element may have done the pausing/unpausing.
	if(is_dragging) {
		// Only resume if it was running beforehand.
		if(!pre_down_state) {
			engine.pause_ctrl.resume();
		} 
		is_dragging = false;
	}
}

engine.renderable({ 
	"render" : (props, dt) => {
		const { context, width, height } = props;
		let sample_idx = engine.sample_idx();
		draw_spike(sample_idx, width, context);
	},
	"setup" : (props) => {
		props.canvas.addEventListener('mousedown', on_down);
		props.canvas.addEventListener('mouseup',on_up);
		props.canvas.addEventListener('mousemove', on_move);
		window.addEventListener('mouseup', on_up);
	}
});
</script>
