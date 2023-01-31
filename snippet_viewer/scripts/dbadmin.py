import typer
import psycopg
import pathlib
from typing import Union


DBNAME = "maindb"
app = typer.Typer()


def connect_postgres(host="localhost"):
    # "db" is the host in docker-compose.yml
    conn = psycopg.connect(
        host=host,
        port=5432,
        user="postgres",
        password="postgres",
        dbname="postgres",
    )
    return conn


def connect(host="localhost"):
    conn = psycopg.connect(
        # host is the service name in docker-compose.yml
        host=host,
        port=5432,
        user="postgres",
        password="postgres",
        dbname="maindb",
    )
    return conn


@app.command()
def create(host="localhost"):
    with connect_postgres(host) as conn:
        conn.autocommit = True
        with conn.cursor() as cur:
            cur.execute(f"CREATE DATABASE {DBNAME}")


def run_sql_file(path: Union[str, pathlib.Path], host="localhost"):
    with connect(host) as conn:
        with conn.cursor() as cur:
            with open(path, "r") as f:
                cur.execute(f.read())


@app.command()
def recreate(host="localhost"):
    drop(host)
    create(host)
    run_sql_file("db/sql/v0.0.1.sql", host)


@app.command()
def drop(host="localhost"):
    with connect_postgres(host) as conn:
        conn.autocommit = True
        with conn.cursor() as cur:
            # Drop all tables in the database.
            cur.execute(f"DROP DATABASE IF EXISTS {DBNAME}")


if __name__ == "__main__":
    app()
