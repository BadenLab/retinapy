import typer
import psycopg
import pathlib
from typing import Union


DBNAME = "maindb"
app = typer.Typer()


def connect_postgres():
    conn = psycopg.connect(
        # host is the service name in docker-compose.yml
        host="db",
        port=5432,
        user="postgres",
        password="postgres",
        dbname="postgres",
    )
    return conn


def connect():
    conn = psycopg.connect(
        # host is the service name in docker-compose.yml
        host="db",
        port=5432,
        user="postgres",
        password="postgres",
        dbname="maindb",
    )
    return conn


def create_db():
    with connect_postgres() as conn:
        conn.autocommit = True
        with conn.cursor() as cur:
            cur.execute(f"CREATE DATABASE {DBNAME}")


def run_sql_file(path: Union[str, pathlib.Path]):
    with connect() as conn:
        with conn.cursor() as cur:
            with open(path, "r") as f:
                cur.execute(f.read())


@app.command()
def recreate():
    drop()
    create_db()
    run_sql_file("db/sql/v0.0.1.sql")


@app.command()
def drop():
    with connect_postgres() as conn:
        conn.autocommit = True
        with conn.cursor() as cur:
            # Drop all tables in the database.
            cur.execute(f"DROP DATABASE IF EXISTS {DBNAME}")


if __name__ == "__main__":
    app()
