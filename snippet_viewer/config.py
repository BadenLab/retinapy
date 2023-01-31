"""Flack configuration."""

import os

class Config:
    TESTING = os.environ.get('FLASK_TESTING', True)
    DEBUG = os.environ.get('FLASK_DEBUG', True)
    FLASK_ENV = os.environ.get('FLASK_ENV', "development")
    DB_HOST = os.environ.get('DBHOST', "db")
    DB_NAME = "maindb"
