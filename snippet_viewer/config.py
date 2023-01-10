"""Flack configuration."""


class Config:
    TESTING = True
    DEBUG = True
    FLASK_ENV = "development"
    SECRET_KEY = "only_for_testing"
    DB_NAME = "database.db"
    SQLALCHEMY_DATABASE_URI = f"sqlite:///{DB_NAME}"
