from flask import Flask

from backend.routes.base_api import api_bp
from backend.routes.ml_api import ml_api_bp # todo remove backend part
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

app.register_blueprint(api_bp, url_prefix="/api")
app.register_blueprint(ml_api_bp, url_prefix="/ml-api")


if __name__ == '__main__':
    app.run()
