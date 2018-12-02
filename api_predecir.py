from flask import Flask
from flask_restful import reqparse, abort, Api, Resource
from supervisado import predict

app = Flask(__name__)
api = Api(app)
# argument parsing
parser = reqparse.RequestParser()
parser.add_argument('query')


class PredictFraud(Resource):
    def get(self):
        # use parser and find the user's query
        args = parser.parse_args()
        user_query = args['query']
        prediction = predict(user_query)
#        output = {
#            'prob_fraude': str(prediction),
#            'cluster: ': str(1),
#            'anomaly': str(0)
#        }
        return str(prediction)


# Setup the Api resource routing here
# Route the URL to the resource
api.add_resource(PredictFraud, '/')


# http http://127.0.0.1:5000/ query=="{'V1': 4,'V2': 1,'V3': 1,'V4': 1,'V5': 2,'V6': 2,'V7': 2,'V8': 2,'V9': 2,'V10': 2,'V11': 2,'V12': 2,'V13': 2,'V14': 2,'V15': 2,'V16': 2,'V17': 2,'V18': 2,'V19': 2,'V20': 2,'V21': 2,'V22': 2,'V23': 2,'V24': 2,'V25': 2,'V26': 2,'V27': 2,'V28': 2,'Amount': 2}"
if __name__ == '__main__':
    app.run(debug=True)
