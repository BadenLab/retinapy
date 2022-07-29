<script>
import {renderable, width, height, snippets } from './game.js';


const OUTER_PADDING = 10;
const NUM_X_BOXES = 30;
const BOX_PADDING_FRACTION = 0.1;
const PLAYBACK_SPEED = 0.05;
const SAMPLE_RATE = 99.182;
const SNIPPET_LEN = 120;
const SNIPPET_PAD = 20;
const BOX_Y_START = 40; 

let snippets_;

snippets.subscribe( v => {snippets_ = v;});



function cur_sample(sec_since_start) {
	const snippet_time = sec_since_start * PLAYBACK_SPEED;
	const snippet_idx = Math.floor((snippet_time * SAMPLE_RATE) % SNIPPET_LEN);
	return snippet_idx;
}

function kth_box(k, c_width) {
	const available_width = (c_width - 2*OUTER_PADDING)
	const box_space = available_width / NUM_X_BOXES 
	const x_idx = k % NUM_X_BOXES
	const y_idx = Math.floor(k / NUM_X_BOXES)
	// (x,y)
	const box_space_T_left_x = OUTER_PADDING + x_idx * box_space
	const box_space_T_left_y = OUTER_PADDING + y_idx * box_space
	//const box_space_B_right_x = box_space_T_left_x + box_size
	//const box_space_B_right_y = box_space_T_left_y + box_size
	const padding = box_space * BOX_PADDING_FRACTION
	const box_TL_y = BOX_Y_START + box_space_T_left_y + padding / 2
	const box_TL_x = box_space_T_left_x + padding / 2
	const size = box_space - padding
	//const box_BR_y = box_space_B_right_y - padding
	//const box_BR_x = box_space_B_right_x - padding
	return [box_TL_x, box_TL_y, size, size]
}

function as_rgb(stimulus) {
	// 0: red, 1: green, 2: UV, 3: blue
	return [stimulus[0]*255, stimulus[1]*255, stimulus[2]*255];
}

function draw_snippet(idx, sample_idx, width, ctx) {
	let sample = snippets_[idx][sample_idx];
	let rgb = as_rgb(sample);
	ctx.fillStyle = `rgb(${rgb[0]}, ${rgb[1]}, ${rgb[2]})`
	ctx.fillRect(...kth_box(idx, width));
	ctx.strokeStyle = 'black';
	ctx.lineWidth = 2;
	ctx.strokeRect(...kth_box(idx, width));
}

function draw_spike(sample_idx, width, ctx) {
	const fraction = sample_idx / SNIPPET_LEN;
	const line_y = 20;
	const time_dot_size = 8;
	const spike_line_height = 20;
	const time_dot_y = line_y ;
	const time_dot_radius = 3;
	// Line
	ctx.strokeStyle = 'white';
	ctx.stroke();
	ctx.beginPath();
	ctx.moveTo(OUTER_PADDING, line_y);
	ctx.lineTo(width - OUTER_PADDING, line_y);
	ctx.stroke();

	// Spike point
	ctx.strokeStyle = 'rgb(255, 150, 150)';
	ctx.beginPath();
	const line_width = width - OUTER_PADDING * 2;
	const spike_point_rel = 1 - SNIPPET_PAD / SNIPPET_LEN;
	const spike_point = OUTER_PADDING + line_width * spike_point_rel;
	ctx.moveTo(spike_point, line_y - spike_line_height/2);
	ctx.lineTo(spike_point, line_y + spike_line_height/2);
	ctx.stroke();

	// Time point
	ctx.beginPath();
	ctx.fillStyle = 'rgb(255, 150, 150)';
	ctx.arc(OUTER_PADDING + fraction * line_width, time_dot_y, time_dot_radius, 0, 2*Math.PI);
	// Can't get fill working well.
	//ctx.fill();
}

renderable((props, dt) => {
	const { context, width, height, time } = props;
	let sample_idx = cur_sample(time);
	for(let i = 0; i < snippets_.length; i++) {
		draw_snippet(i, sample_idx, width, context);
	}
	draw_spike(sample_idx, width, context);
});
	
</script>
