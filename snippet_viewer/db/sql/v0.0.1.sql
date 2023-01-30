CREATE TABLE workspaces (
	id SERIAL PRIMARY KEY,
	title VARCHAR(255) NOT NULL,
	url_slug VARCHAR(255) NOT NULL
);

CREATE TABLE snippet_groups (
	id SERIAL PRIMARY KEY,
	title VARCHAR(255) NOT NULL,
	spikes jsonb NULL,
	x_pos FLOAT,
	y_pos FLOAT,
	width FLOAT,
	height FLOAT,
	workspace_id INTEGER NOT NULL,
	FOREIGN KEY (workspace_id) REFERENCES workspaces(id) ON DELETE CASCADE
);

-- CREATE TABLE spikes (
-- 	id SERIAL PRIMARY KEY,
-- 	sample_idx INTEGER,
-- 	rec_id INTEGER,
-- 	group_id INTEGER NOT NULL,
-- 	FOREIGN KEY (group_id) REFERENCES groups(id) ON DELETE CASCADE
-- );






