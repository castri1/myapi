# import bcrypt
# from flask import request
# from MyApp import app
# from functools import wraps

# salt = '$2b$14$r8RR01VMa2qra0ZhCnBzme' #bcrypt.gensalt(14)
# secret_key = app.config['SECRET_KEY']

# def generate_token():
#     hashed_token = bcrypt.hashpw(secret_key, salt) 
#     return hashed_token

# def token_authorized(token):
#     hashed_token = bcrypt.hashpw(secret_key, salt)
#     return token == hashed_token

# def authentication_required(f):
#     @wraps(f)
#     def decorated_function(*args, **kwargs):
#            token = str(request.headers.get('token'))
#            if not token_authorized(token):
#                return 'Not authorized', 401
#            return f(*args, **kwargs)
#     return decorated_function