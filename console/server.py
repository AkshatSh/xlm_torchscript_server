from typing import Tuple, Any, Dict

import flask
from flask import (
    Flask,
    jsonify,
    request,
    abort,
    Blueprint,
)
from flask_restful import Api
from flask_cors import CORS
import requests

import threading
import logging
import argparse
import urllib
import csv
import json


def get_args() -> argparse.ArgumentParser:
    '''
    Return CLI configuration for running server
    '''
    parser = argparse.ArgumentParser(description='Run the html hosting server')
    parser.add_argument(
        '--port',
        default=5000,
        type=int,
        help='the port to launch server on'
    )

    parser.add_argument(
        '--modelserver',
        default="http://localhost:8080"
    )

    parser.add_argument(
        '--debug',
        action='store_true',
        help='Launch debug version of server',
    )

    parser.add_argument(
        "--datafile",
        default="data.csv",
        help="the file to write data to"
    )
    return parser

logger = logging.getLogger(__name__)

def get_key_from_data(data, key):
    if key in data:
        return data[key]
    return None

def create_app(config_filename: str):
    api_bp = Blueprint('api', __name__)
    api = Api(api_bp)
    app = Flask(__name__, static_url_path='/static')
    CORS(app, resources=r'/api/*')

    # app.config.from_object(config_filename)
    app.register_blueprint(api_bp, url_prefix='/api')

    return app

def setup_app(app: Flask, args: object):
    @app.route('/api/coronavirus_model/', methods=['GET'])
    def get_predictions():
        query = get_key_from_data(request.args, 'query')
        payload = {'text': str(query)}
        payload = urllib.parse.urlencode(payload, quote_via=urllib.parse.quote)
        r = requests.get(args.modelserver, params=payload)
        response = json.loads(r.text)
        intent_scores = response['intent_ranking']

        intent_scores = list(filter(lambda inp: inp, intent_scores))

        def convert_to_tuple(intent_score: Dict[str, Any]) -> Tuple[str, float]:
            intent_name = intent_score['name']
            intent_score = intent_score['confidence']
            return intent_name, intent_score

        intent_scores = list(map(convert_to_tuple, intent_scores))
        intent_scores = sorted(intent_scores, reverse=True, key=lambda tup: tup[1])
        return jsonify({
            "query": query,
            "prediction": intent_scores[0],
            "raw_scores": intent_scores,
        }), 200
    
    @app.route('/api/add_coronavirus_data/', methods=['GET'])
    def upload_data():
        data_point = get_key_from_data(request.args, 'data_point')
        query,label = data_point.split(",")
        lock = threading.Lock()
        with lock:
            with open(args.datafile, 'a+') as csv_file:
                csvwriter = csv.writer(csv_file, delimiter='\t')
                csvwriter.writerow([query, label])
        
        return jsonify({
            "query": query,
            "label": label,
        }), 200
    
    @app.route('/', methods=['GET'])
    def root():
        return app.send_static_file('index.html')

def main():
    args = get_args().parse_args() 

    app = create_app("config")
    setup_app(app, args)
    try:
        app.run(
            host='0.0.0.0',
            debug=args.debug,
            port=args.port,
        )
    except KeyboardInterrupt:
        print('Recieved Keyboard interrupt, saving and exiting')

if __name__ == "__main__":
    main()
