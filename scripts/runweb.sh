#!/bin/bash

cd /app/snippet_viewer/svelte_classic
npm install
npm run dev &
cd /app/snippet_viewer/svelte_workspace
npm install
npm run dev &
cd /app/snippet_viewer/
flask --app flaskapp --debug run --host=0.0.0.0
