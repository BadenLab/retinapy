<script>
	import Text from './Text.svelte';
	import { time, renderable } from './engine.js';
	
	let text = '';

	let elapsed = 0;
	let frames = 0;
	let prevTime = performance.now();
	renderable((state, dt) => {
		let time = performance.now();
		frames++;
		if ( time >= prevTime + 1000 ) {
			const fps = ((frames * 1000) / (time - prevTime));
			text = `${fps.toFixed(1)} FPS`;
			//text = `${time.toFixed(1)} sec`;
			prevTime = time;
			frames = 0;
		}
	});
</script>

<Text
	{text}
	fontSize=12
	fontFamily='Courier New'
	align='left'
	baseline='top'
	x={20}
	y={5} />

<!-- The following allows this component to nest children -->
<slot></slot>
