import os
basedir = os.path.abspath(os.path.dirname(__file__))

# docker run --name chobot-db-postgres -p 5432:5432  -e POSTGRES_PASSWORD=chobotdb -d postgres
class DbConfig(object):
    SQLALCHEMY_DATABASE_URI = os.environ.get('postgresql://localhost:5432/chobotdb') or 'sqlite:///' + os.path.join(basedir, 'app.db')
    SQLALCHEMY_TRACK_MODIFICATIONS = False


# jdbc:postgresql://localhost:5432/chobotdb


