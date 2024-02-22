from flask import Flask, request
from flask_restx import Api, Resource, fields
import pickle
import pandas as pd
import numpy as np
import os

app = Flask(__name__)
api = Api(app, version='1.0', title='Fraud Detection API', description='Project 1 fraud detection API')
ns = api.namespace('prediction', description='Prediction operations')

predict_model = api.model('PredictModel', {
    'features': fields.List(fields.Float, required=True, description='Transaction features')
})

model_path = os.path.join(os.path.dirname(__file__), 'model.pkl')
model = pickle.load(open(model_path, 'rb'))

@ns.route('/predict')
class Predict(Resource):
    @ns.expect(predict_model)
    @ns.response(200, 'Success')
    def post(self):
        '''Predict if a transaction is fraudulent'''
        data = request.json['features']
        df = pd.DataFrame([data], columns=[f'V{i}' for i in range(1, len(data) + 1)])
        prediction = model.predict(df)
        return {'prediction': int(prediction[0])}

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
