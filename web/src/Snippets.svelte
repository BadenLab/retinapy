<script>
import * as engine from './engine.js'; 


const OUTER_PADDING = 10;
const NUM_X_BOXES = 30;
const BOX_PADDING_FRACTION = 0.1;
const BOX_Y_START = 80; 

function kth_box(k, c_width) {
	const available_width = (c_width - 2*OUTER_PADDING)
	const box_space = available_width / NUM_X_BOXES 
	const x_idx = k % NUM_X_BOXES
	const y_idx = Math.floor(k / NUM_X_BOXES)
	// (x,y)
	const box_space_T_left_x = OUTER_PADDING + x_idx * box_space
	const box_space_T_left_y = OUTER_PADDING + y_idx * box_space
	const padding = box_space * BOX_PADDING_FRACTION
	const box_TL_y = BOX_Y_START + box_space_T_left_y + padding / 2
	const box_TL_x = box_space_T_left_x + padding / 2
	const size = box_space - padding
	return [box_TL_x, box_TL_y, size, size]
}


function draw_sample(sample, idx, width, ctx) {
	const box = kth_box(idx, width);

	ctx.lineWidth = 1;
	ctx.strokeStyle = 'rgb(0, 0, 0)';
	ctx.fillStyle = 'rgb(0, 0, 0)';

	ctx.fillStyle = `rgb(${sample[2]*200}, 0, ${sample[2]*255})`;
	ctx.fillRect(...box);

	const dr = 2;
	const smaller_box = [box[0] + dr, box[1] + dr, box[2] - 2*dr, box[3] - 2*dr];
	let rgb =  [sample[0]*255, sample[1]*255, sample[3]*255];
	ctx.fillStyle = `rgb(${rgb[0]}, ${rgb[1]}, ${rgb[2]})`
	ctx.fillRect(...smaller_box);
}


engine.renderable((props, dt) => {
	const { context, width, height } = props;
	let snippets = engine.snippets();
	let sample_idx = engine.sample_idx();
	for(let i = 0; i < snippets.length; i++) {
		const sample = snippets[i][sample_idx];
		draw_sample(sample, i, width, context);
	}
});
	
</script>
