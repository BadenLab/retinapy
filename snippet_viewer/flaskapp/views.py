import flask
import json
from . import models
from . import db
from . import mea_api

bp = flask.Blueprint("views", __name__)


@bp.route("/")
def index():
    """index"""
    return flask.render_template("index.html")


@bp.route("/classic")
def classic():
    """The original version of the app."""
    return flask.render_template("classic.html")


@bp.route("/workspaces", methods=["POST"])
def create_workspace():
    """
    Create a new workspace.

    Create new workspace and save to database.
    """
    ws = models.Workspace(title="Untitled")
    db.session.add(ws)
    db.session.commit()
    return flask.redirect(f"/workspaces/{ws.id}")


@bp.get("/workspaces/<int:workspace_id>")
def workspace_get(workspace_id):
    """workspace"""
    ws = db.get_or_404(models.Workspace, workspace_id)
    return flask.render_template(
        "workspace.html", workspace_id=workspace_id, workspace_title=ws.title
    )


@bp.patch("/workspaces/<int:workspace_id>")
def workspace_patch(workspace_id):
    ws = db.get_or_404(models.Workspace, workspace_id)
    if "title" in flask.request.form:
        ws.title = flask.request.form["title"]
        db.session.commit()
    return ""


@bp.route("/workspaces/<int:workspace_id>/snippets/batch_add", methods=["POST"])
def batch_add_snippets(workspace_id):
    ws = db.get_or_404(models.Workspace, workspace_id)
    error = None
    if not all(x in flask.request.form for x in ["rec_id", "cell_id"]):
        return 'Form requires "rec_id" and "cell_id"', 400
    rec_id = int(flask.request.form["rec_id"])
    cell_id = int(flask.request.form["cell_id"])
    if not rec_id in mea_api.recs_by_id:
        return "Invalid recording id", 400
    elif not cell_id in mea_api.recs_by_id[rec_id].cluster_ids:
        return "Invalid cell id", 400

    sample_rate = mea_api.recs_by_id[rec_id].sample_rate
    spikes_secs = (
        mea_api.recs_by_id[rec_id].spike_events(cell_id) / sample_rate
    )
    gid = mea_api.rec_cluster_ids[(rec_id, cell_id)]
    
    created_snippets = [
            models.Snippet(
                cell_gid=gid,
                spike_time_secs=t,
                workspace_id=workspace_id)
            ]
    for snip in created_snippets:
        db.session.add(models.Snippet(workspace=ws, data=snip))
    db.session.commit()

    pass
