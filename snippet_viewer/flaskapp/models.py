from . import db


class Snippet(db.Model):
    __tablename__ = "snippets"

    id = db.Column(db.Integer, primary_key=True)
    cell_gid = db.Column(db.Integer, nullable=False)
    spike_time_secs = db.Column(db.Float, nullable=False)
    x = db.Column(db.Float, nullable=False)
    y = db.Column(db.Float, nullable=False)
    z = db.Column(db.Float, nullable=False)
    workspace_id = db.Column(
        db.Integer, db.ForeignKey("workspaces.id"), nullable=False
    )
    group_id = db.Column(
        db.Integer, db.ForeignKey("snippet_groups.id"), nullable=False
    )
    workspace = db.relationship("Workspace", back_populates="snippets")
    group = db.relationship("SnippetGroup", back_populates="snippets")


class SnippetGroup(db.Model):
    __tablename__ = "snippet_groups"

    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(80), nullable=False, unique=True)
    workspace_id = db.Column(
        db.Integer, db.ForeignKey("workspaces.id"), nullable=False
    )
    group_id = db.Column(
        db.Integer, db.ForeignKey("snippet_groups.id"), nullable=False
    )
    workspace = db.relationship("Workspace", back_populates="snippet_groups")
    #child_groups = db.relationship("SnippetGroup", back_populates="group")
    snippets = db.relationship("Snippet", back_populates="group")


class Workspace(db.Model):
    __tablename__ = "workspaces"

    id = db.Column(db.Integer, primary_key=True)
    title = db.Column(db.String(80), nullable=False)
    snippets = db.relationship("Snippet", back_populates="workspace", lazy=True)
    snippet_groups = db.relationship(
        "SnippetGroup", back_populates="workspace", lazy=True
    )
