import judd_vos_cmf from './judd_vos_cmf.json';

/**
	input is [0, 1] values.
*/
export function rgbToNumeric(rgb: number[]): number {
	const res = (rgb[0] * 255 << 16) + (rgb[1] * 255 << 8) + (rgb[2] * 255 | 0);
	return res;
}


export function numericToRGB(n: number): string {
	const colorString = "#" + n.toString(16).padStart(6, '0');
	return colorString;
}


export function XYZtoSRGB(xyz: number[]): number[] {
	// Conversion matrix
	const r = xyz[0] * 3.2406 + xyz[1] * -1.5372 + xyz[2] * -0.4986;
	const g = xyz[0] * -0.9689 + xyz[1] * 1.8758 + xyz[2] * 0.0415;
	const b = xyz[0] * 0.0557 + xyz[1] * -0.2040 + xyz[2] * 1.0570;
	let rgb = [r, g, b];

	// Gamma correction
	rgb = rgb.map((v) => {
		if (v <= 0.0031308) {
			return 12.92 * v;
		} else {
			return 1.055 * Math.pow(v, 1 / 2.4) - 0.055;
		}
	});

	// Clamp values to [0, 1]
	rgb = rgb.map((v) => Math.max(0, Math.min(1, v)));

	// Convert to 8-bit integer
	//rgb = rgb.map((v) => Math.round(v * 255));

	return rgb;
}

function toXYZ(rgbv: number[]): number[] {
	//const peaks = [420, 480, 505, 630]; // nm
	//const power = [65, 60, 65, 100]; // nW
	//const width = 10; // nm
	// The width is narrow enough that I think just one bin of the spectrum is 
	// fine.
	const xyz: number[] = [
		// "Red" LED
		judd_vos_cmf[635].map(x => x * 100/4 * rgbv[0]),
		judd_vos_cmf[630].map(x => x * 100/2 * rgbv[0]),
		judd_vos_cmf[625].map(x => x * 100/4 * rgbv[0]),
		//
		judd_vos_cmf[510].map(x => x * 65/4 * rgbv[1]),
		judd_vos_cmf[505].map(x => x * 65/2 * rgbv[1]),
		judd_vos_cmf[500].map(x => x * 65/4 * rgbv[1]),
		//
		judd_vos_cmf[485].map(x => x * 60/4 * rgbv[2]),
		judd_vos_cmf[480].map(x => x * 60/2 * rgbv[2]),
		judd_vos_cmf[475].map(x => x * 60/4 * rgbv[2]),
		// "Blue" LED
		judd_vos_cmf[425].map(x => x * 65/4 * rgbv[3]),
		judd_vos_cmf[420].map(x => x * 65/2 * rgbv[3]),
		judd_vos_cmf[415].map(x => x * 65/4 * rgbv[3])
	].reduce((acc, x) => acc.map((e, i) => e + x[i]), [0, 0, 0]);
	return xyz;
};

const xyzWhite = toXYZ([1, 1, 1, 1]);
//const xyzWhiteSum = xyzWhite.reduce((acc, x) => acc + x, 0);

export function stimToSRGB(rgbv: number[]): number[] {
	const xyz = toXYZ(rgbv);
	//const xyzNorm = xyz.map(x => x / xyzWhiteSum);
	const xyzNorm = xyz.reduce((acc, x) => acc + x, 0);
	const xyzNormed = xyz.map(x => x / xyzWhite[1]);
	const rgb = XYZtoSRGB(xyzNormed);
	//console.log("white: ", xyzWhite); [83.87, 62.33, 128.03]
	//console.log("xyz: ", xyz);
	//console.log("xyzNorm: ", xyzNorm);
	//console.log("rgbNorm: ", rgb);
	return rgb;
}
