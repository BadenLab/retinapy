<script>
import * as engine from './engine.js'; 


const OUTER_PADDING = 10;
const BOX_PADDING_FRACTION = 0.1;
const BOX_Y_START = 70; 
export let num_x_boxes = 55;

function kth_box(k, c_width) {
	const available_width = (c_width - 2*OUTER_PADDING);
	const box_space = available_width / num_x_boxes; 
	const x_idx = k % num_x_boxes
	const y_idx = Math.floor(k / num_x_boxes);
	// (x,y)
	const box_space_T_left_x = OUTER_PADDING + x_idx * box_space;
	const box_space_T_left_y = OUTER_PADDING + y_idx * box_space;
	const padding = box_space * BOX_PADDING_FRACTION;
	const box_TL_y = BOX_Y_START + box_space_T_left_y + padding / 2;
	const box_TL_x = box_space_T_left_x + padding / 2;
	const size = box_space - padding;
	return [box_TL_x, box_TL_y, size, size];
}


function draw_sample(sample, idx, width, ctx) {
	const box = kth_box(idx, width);

	ctx.lineWidth = 2;
	ctx.strokeStyle = 'rgb(0, 0, 0)';
	ctx.fillStyle = 'rgb(0, 0, 0)';

	ctx.fillStyle = `rgb(${sample[3]*255}, 0, ${sample[3]*255})`;
	ctx.fillRect(...box);

	const dr =  3;
	const smaller_box = [box[0] + dr, box[1] + dr, box[2] - 2*dr, box[3] - 2*dr];
	let rgb =  [sample[0]*250, sample[1]*240, sample[2]*210];
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
