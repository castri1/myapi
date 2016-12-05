"""
Routes and views for the flask application.
"""

from datetime import datetime
from flask import render_template, jsonify, request, send_from_directory
from MyApp import app
from Models import estimate_mack, estimate_ferguson, estimate_hovinen, estimate_mack_manual

import numpy as np
import os
# import bcrypt
import json
# from MyApp.security import generate_token, authentication_required
from MyApp.calculator_exceptions import InvalidUsage


@app.route('/favicon.ico')
def favicon():
    return send_from_directory(os.path.join(app.root_path, 'static'),
                               'favicon.ico', mimetype='image/vnd.microsoft.icon')

@app.route('/')
@app.route('/home')
def home():
    return render_template('home.html')

@app.route('/token')
def token():
    return jsonify(token = generate_token())

@app.route('/api/mack', methods=['POST'])
#@authentication_required
def calculate_mack():
    input_data = json.loads(request.data)

    if not input_data:
        raise InvalidUsage('No data provided', 400)

    data = input_data['data']
    
    if not data:
        raise InvalidUsage('Empty triangle', 400)

    data = np.array(data, dtype='float')
    
    try:
        res = estimate_mack(data)
        return jsonify(res)
    except AssertionError as ae:
        return ae.message, 400
    except Exception as e:
        return "Ha ocurrido un error calculando la proyeccion de siniestros. Por favor contacte su administrador", 400

@app.route('/api/ferguson', methods=['POST'])
#@authentication_required
def calculate_ferguson():
    input_data = json.loads(request.data)

    if not input_data:
        return 'No data provided', 400

    fields = ("data", "nus", "variances")

    if not all(key in input_data for key in fields):
        raise InvalidUsage('Missing argument(s)', status_code = 400)

    data = input_data['data']
    nus = input_data['nus']
    variances = input_data['variances']

    if not data:
        raise InvalidUsage('Empty triangle', status_code = 400)

    if not nus:
        raise InvalidUsage('Empty nus', status_code = 400)

    if not variances:
        raise InvalidUsage('Empty variances', status_code = 400)

    data = np.array(data, dtype='float')
    nus = np.array(nus, dtype='float')
    variances = np.array(variances, dtype='float')

    try:
        res = estimate_ferguson(data, nus, variances)
        return jsonify(res)
    except AssertionError as ae:
        return ae.message, 400
    except Exception as e:
        return "Ha ocurrido un error calculando la proyeccion de siniestros. Por favor contacte su administrador", 400

@app.route('/api/hovinen', methods=['POST'])
#@authentication_required
def calculate_hovinen():
    input_data = json.loads(request.data)

    if not input_data:
        return 'No data provided', 400

    fields = ("data", "nus", "variances")

    if not all(key in input_data for key in fields):
        raise InvalidUsage('Missing argument(s)', status_code = 400)

    data = input_data['data']
    nus = input_data['nus']
    variances = input_data['variances']

    if not data:
        raise InvalidUsage('Empty triangle', status_code = 400)

    if not nus:
        raise InvalidUsage('Empty nus', status_code = 400)

    if not variances:
        raise InvalidUsage('Empty variances', status_code = 400)

    data = np.array(data, dtype='float')
    nus = np.array(nus, dtype='float')
    variances = np.array(variances, dtype='float')

    try:
        res = estimate_hovinen(data, nus, variances)
        return jsonify(res)
    except AssertionError as ae:
        return ae.message, 400
    except Exception as e:
        return "Ha ocurrido un error calculando la proyeccion de siniestros. Por favor contacte su administrador", 400

@app.route('/api/manual', methods=['POST'])
def calculate_manual():
    input_data = json.loads(request.data)
    
    if not input_data:
        return 'No data provided', 400

    fields = ("data", "fjs")

    if not all(key in input_data for key in fields):
        raise InvalidUsage('Missing argument(s)', status_code = 400)

    data = input_data['data']
    fjs = input_data['fjs']

    if not data:
        raise InvalidUsage('Empty triangle', status_code = 400)

    if not fjs:
        raise InvalidUsage('Empty fjs', status_code = 400)

    data = np.array(data, dtype='float')
    fjs = np.array(fjs, dtype='float')

    try:
        res = estimate_mack_manual(data, fjs)
        return jsonify(res)
    except AssertionError as ae:
        return ae.message, 400
    except Exception as e:
        return "Ha ocurrido un error calculando la proyeccion de siniestros. Por favor contacte su administrador", 400
    

@app.errorhandler(InvalidUsage)
def handle_invalid_usage(error):
    response = jsonify(error.to_dict())
    response.status_code = error.status_code
    return response

@app.errorhandler(404)
def page_not_found(e):
    return render_template('404.html'), 404

@app.errorhandler(500)
def server_error(e):
    return render_template('500.html'), 500

@app.errorhandler(401)
def not_authorized(e):
    return Response('Not authorized', 401)