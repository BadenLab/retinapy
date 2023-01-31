import flask
import psycopg
import psycopg_pool
import retinapy.mea as mea
import pathlib
import json
import pathlib


"""
Only the basic Flask apps can avoid having a factory function to create the
app. In this case it is `init_app()`. When using the factory function, the
decorated functions that become routes can either be defined within `init_app()`
or in a separate module. If in a separate module, we have an issue of 
circular imports to resolve. The issue is that the routes need to be decorated
with `app.route()` and `app` is not defined until `init_app()` is called. The
solution to this is to use blueprints. A blueprint object replaces `app` in
the `app.route()` decorator. Then the blueprint is registered with the app
in `init_app()`. This two step process introduces a single level of decoupling
between the `app` object and the functions to be decorated.
"""

# Database connection goes here. Not in requset or global context, both of
# which are created and deleted before and after a single request.
_dbpool = None

def create_dbpool(host, dbname):
    global _dbpool
    _dbpool = psycopg_pool.ConnectionPool(
        # It's possible to give individual parameters like host="127.0.0.1"
        # which get collated into kwargs, but I prefer to use the connection
        # string, and then be free to use named parameters.
        #conninfo="postgresql://postgres:postgres@db:5432/maindb",
        conninfo=f"postgresql://postgres:postgres@{host}:5432/{dbname}",
        min_size=1, max_size=10, max_waiting=5)

def dbpool():
    #s = _dbpool.get_stats()
    #app.logger.info(s)
    return _dbpool


def init_app():
    app = flask.Flask(__name__, instance_relative_config=True)
    # Load config.
    app.config.from_object("config.Config")

    # Create database connection pool.
    host = app.config["DB_HOST"]
    dbname = app.config["DB_NAME"]
    create_dbpool(host, dbname)
        
    # Register blueprints.
    from . import mea_api
    from . import views

    app.register_blueprint(mea_api.bp, url_prefix="/api")
    app.register_blueprint(views.bp, url_prefix="/")

    # Our hacky and temporary solution for loading data.
    mea_api.init_data()

    # We could optionally add routes here.
    # @app.route("/hello")
    # def hello():
    #    return "Hello World!"

    return app

app = init_app()
