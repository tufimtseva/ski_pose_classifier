from flask import Flask

from routes.base_api import api_bp
from routes.ml_api import ml_api_bp

app = Flask(__name__)

app.register_blueprint(api_bp, url_prefix="/api")
app.register_blueprint(ml_api_bp, url_prefix="/ml-api")


if __name__ == '__main__':
    app.run()
