import flask
import flask_sqlalchemy
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

db = flask_sqlalchemy.SQLAlchemy()

def init_app():
    app = flask.Flask(__name__, instance_relative_config=True)
    # Load config.
    app.config.from_object("config.Config")
    # Initialize the database.
    db.init_app(app)
    # Register blueprints.
    from . import mea_api
    from . import views
    app.register_blueprint(mea_api.bp, url_prefix="/api")
    app.register_blueprint(views.bp, url_prefix="/")

    # Our hacky and temporary solution for loading data.
    mea_api.init_data()

    # Check if the database is empty.
    from . import models
    create_database(app)



    # We could optionally add routes here.
    #@app.route("/hello")
    #def hello():
    #    return "Hello World!"

    return app

def create_database(app):
    db_path = pathlib.Path(app.config['DB_NAME'])
    if not db_path.exists():
        with app.app_context():
            db.create_all()

app = init_app()
