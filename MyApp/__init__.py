from flask import Flask

app = Flask(__name__)
app.config['SECRET_KEY'] = 'Vaado123#'

import MyApp.views