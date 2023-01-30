export class SpikeData {
	rec_id: number;
	sample_idx: number;

	constructor(recId: number, sampleIdx: number) {
		this.rec_id = recId;
		this.sample_idx = sampleIdx;
	}
}
// TODO these actions need to queued up and done in sequence.

export class GroupData {
	id: number;
	title: string;
	spikes: SpikeData[];
	x_pos: number;
	y_pos: number;
	width: number;
	height: number;

	constructor(
		id: number,
		title: string,
		spikes: SpikeData[],
		x_pos: number,
		y_pos: number,
		width: number,
		height: number) {
		this.id = id;
		this.title = title;
		this.spikes = spikes;
		this.x_pos = x_pos;
		this.y_pos = y_pos;
		this.width = width;
		this.height = height;
	}
}

export async function createGroup(workspace_id: string,
	groupData: GroupData,
	on_success: (group_id: number) => void) {
	return fetch(`/workspaces/${workspace_id}/groups`, {
		method: 'POST',
		headers: {
			'Content-Type': 'application/json'
		},
		body: JSON.stringify({
			title: groupData.title,
			spikes: groupData.spikes,
			x_pos: groupData.x_pos,
			y_pos: groupData.y_pos,
			width: groupData.width,
			height: groupData.height
		})
	}).then(response => response.json())
		.then(data => data["id"])
		.then(group_id => on_success(group_id));
}

export async function deleteGroup(workspace_id: string, group_id: number) {
	return fetch(`/workspaces/${workspace_id}/groups/${group_id}`, {
		method: 'DELETE',
	});
}

export async function patchGroupPos(
	workspace_id: number, group_id: number, x_pos: number, y_pos: number) {
	await fetch(`/workspaces/${workspace_id}/groups/${group_id}`, {
		method: 'PATCH',
		headers: {
			'Content-Type': 'application/json'
		},
		body: JSON.stringify({
			x_pos: x_pos,
			y_pos: y_pos
		}),
	})
}

export async function getGroups(workspace_id: number): Promise<GroupData[]> {
	const gdata = await fetch(`/workspaces/${workspace_id}/groups`)
		.then(response => response.json());
	return gdata;
}

