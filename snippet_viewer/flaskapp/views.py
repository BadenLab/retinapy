import flask
import json
import secrets
import psycopg as pg
import pydantic
from . import dbpool
from . import mea_api
from typing import List, Sequence, Optional, Iterable

bp = flask.Blueprint("views", __name__)

WORKSPACE_ID_NUM_BYTES = 8


def sql_set_placeholders(row_names: Iterable):
    """
    Create the psycopg SQL structure:
        row1=%(row1)s row2=%(row2)s

    """
    placeholders = pg.sql.SQL(", ").join(
        [
            pg.sql.Composed(
                [
                    pg.sql.Identifier(row),
                    pg.sql.SQL(" = "),
                    pg.sql.Placeholder(row),
                ]
            )
            for row in row_names
        ]
    )
    return placeholders


def sql_insert_into(table_name, row_names):
    sql = pg.sql.SQL(
        "INSERT INTO {} ({}) VALUES ({})".format(
            pg.sql.Identifier(table_name),
            pg.sql.SQL(", ").join(map(pg.sql.Identifier, row_names)),
            pg.sql.SQL(", ").join(map(pg.sql.Placeholder, row_names)),
        )
    )
    # Log using flask logger.
    flask.current_app.logger.error(sql)
    return sql


def param_or_error(req_json, param_name, err_msg=None):
    err_msg = err_msg if err_msg else ""
    res = req_json.get(param_name, None)
    if res is None:
        flask.abort(400, err_msg)
    else:
        return res


def ws_or_error(ws_slug, error_code=404, msg=""):
    # Get workspace by URL
    with dbpool().connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT (id, title) FROM workspaces " "WHERE " "url_slug = %s",
                (ws_slug,),
            )
            ws_row = cur.fetchone()
            if ws_row is None:
                flask.abort(error_code, description=f"No workspace '{ws_slug}'")
            ws_id, title = ws_row[0]
    return ws_id, title


def group_or_error(ws_slug, group_id, error_code=404, msg=""):
    ws_id, _ = ws_or_error(ws_slug, error_code=401)
    with dbpool().connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT (id, workspace_id) FROM snippet_groups WHERE id = %s",
                (group_id,),
            )
            group_row = cur.fetchone()
            if group_row is None:
                flask.abort(error_code, msg)
            group_id, ws_id2 = group_row[0]
            if ws_id2 != ws_id:
                flask.abort(error_code, msg)
    return group_id


def ws_groups(ws_id):
    with dbpool().connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT (title, spikes, x_pos, y_pos, width, height) "
                "FROM snippet_groups "
                "WHERE "
                "workspace_id = %s",
                (ws_id,),
            )
            group_rows = cur.fetchall()
    return group_rows


def workspace_as_json(title, slug, group_rows):
    res = {
        "title": title,
        "id": slug,
        "groups": group_rows,
    }
    return json.dumps(res)


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
    slug = secrets.token_urlsafe(WORKSPACE_ID_NUM_BYTES)
    DEFAULT_TITLE = "Untitled"
    with dbpool().connection() as conn:
        with conn.cursor() as cur:
            sql = "INSERT INTO workspaces (title, url_slug) VALUES (%s, %s)"
            values = (DEFAULT_TITLE, slug)
            cur.execute(sql, values)
            # ws_id = cur.fetchone()[0]
    res = {"id": slug, "title": DEFAULT_TITLE}
    return flask.redirect(f"/workspaces/{slug}")


@bp.get("/workspaces/<string:workspace_slug>")
def workspace_get(workspace_slug):
    """workspace"""
    ws_id, title = ws_or_error(workspace_slug)
    groups = ws_groups(ws_id)
    ws = workspace_as_json(title, workspace_slug, groups)
    return flask.render_template(
        "workspace.html", workspace_id=workspace_slug, workspace_title=title
    )


@bp.patch("/workspaces/<string:workspace_slug>")
def workspace_patch(workspace_slug):
    ws_id, _ = ws_or_error(workspace_slug)
    if "title" in flask.request.form:
        title = flask.request.form["title"]
        with dbpool().connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "UPDATE workspaces SET title = %s WHERE id = %s",
                    (title, ws_id),
                )
                conn.commit()
    return ""


# @bp.post("/workspaces/<string:workspace_slug>/groups")
def create_group(workspace_slug):
    ws_id, _ = ws_or_error(workspace_slug, 400, "Invalid recording")
    req_body = flask.request.get_json()
    title = param_or_error(req_body, "title", err_msg="Missing title")
    workspace_id = ws_id
    with dbpool().connection() as conn:
        with conn.cursor() as cur:
            # fmt: off
            cur.execute(
                "INSERT INTO snippet_groups "
                "(title, workspace_id) "
                "VALUES "
                "(%s, %s) "
                "RETURNING id",
                (title, workspace_id),
            )
            # fmt: on
            group_id = cur.fetchone()[0]
            if not group_id:
                flask.abort(500)
            conn.commit()
    res = json.dumps({"id": group_id})
    return res


class Spike(pydantic.BaseModel):
    sample_idx: int
    rec_id: int


class GroupRow(pydantic.BaseModel):
    id: int
    title: str
    spikes: Optional[Sequence[Spike]]
    x_pos: int
    y_pos: int
    width: int
    height: int


@bp.get("/workspaces/<string:workspace_slug>/groups")
def get_groups(workspace_slug):
    ws_id, _ = ws_or_error(workspace_slug, 400, "Invalid recording")
    with dbpool().connection() as conn:
        with conn.cursor(row_factory=pg.rows.class_row(GroupRow)) as cur:
            cur.execute(
                # fmt: off
                "SELECT * FROM "
                "snippet_groups "
                "WHERE workspace_id = %s",
                (ws_id,),
            )
            # fmt: on
            group_rows = cur.fetchall()
    res = flask.json.dumps(group_rows, default=GroupRow.__json_encoder__)
    return res


class GroupPatch(pydantic.BaseModel):
    title: Optional[str]
    spikes: Optional[Sequence[Spike]]
    x_pos: Optional[int]
    y_pos: Optional[int]
    width: Optional[int]
    height: Optional[int]

    def dict_for_sql(self, allow_optionals=True):
        update_dict = self.dict(exclude_unset=True)
        if not allow_optionals:
            target_dict = self.dict(exclude_unset=False)
            if len(target_dict.keys()) != len(update_dict.keys()):
                flask.abort(400, "Missing fields")
        # We will store spikes as json
        if "spikes" in update_dict:
            update_dict["spikes"] = pg.types.json.Jsonb(update_dict["spikes"])
        return update_dict


@bp.patch("/workspaces/<string:workspace_slug>/groups/<int:group_id>")
def patch_group(workspace_slug: str, group_id: int):
    group_id = group_or_error(workspace_slug, group_id, 400, "Invalid group id")
    group_patch = pydantic.parse_obj_as(GroupPatch, flask.request.get_json())
    update_dict = group_patch.dict_for_sql()
    with dbpool().connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                pg.sql.SQL(
                    "UPDATE snippet_groups SET {} " "WHERE id = %(id)s"
                ).format(sql_set_placeholders(update_dict.keys())),
                update_dict | {"id": group_id},
            )
    return ""


@bp.post("/workspaces/<string:workspace_slug>/groups")
def post_group(workspace_slug):
    ws_id, _ = ws_or_error(workspace_slug)
    group_patch = pydantic.parse_obj_as(GroupPatch, flask.request.get_json())
    update_dict = group_patch.dict_for_sql(allow_optionals=False)
    update_values = update_dict | {"workspace_id": ws_id}
    with dbpool().connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO snippet_groups (title, spikes, x_pos, "
                "y_pos, width, height, workspace_id) VALUES "
                "(%s, %s, %s, %s, %s, %s, %s ) RETURNING id",
                # sql_insert_into("snippet_groups", update_values.keys()),
                # update_values,
                list(update_values.values()),
            )
            group_id = cur.fetchone()[0]
    return json.dumps({"id": group_id})


@bp.delete("/workspaces/<string:ws_slug>/groups/<int:group_id>")
def delete_group(ws_slug: str, group_id: int):
    group_or_error(ws_slug, group_id)
    with dbpool().connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "DELETE FROM snippet_groups WHERE id = %s RETURNING id",
                (group_id,),
            )
            num_deleted = len(cur.fetchone())
            if num_deleted != 1:
                flask.abort(500, "Error deleting group")
    return ""
