from flask import Flask, render_template, request, json, url_for
from flask_cors import CORS
import pickle
import warnings

app = Flask(__name__)
CORS(app)

warnings.filterwarnings('ignore')

# Loading the model
MODEL_PATH = "multiple_reg.sav"
loaded_model = pickle.load(open(MODEL_PATH, 'rb'))

# Render the main page
@app.route('/')
def index():
    return json.dumps({ "message": "Okay" })

# Calculate and return the sales prediction
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        tv = float(request.form['tv'])
        radio = float(request.form['radio'])
        newspaper = float(request.form['newspaper'])

        print(tv, radio, newspaper)
        
        res = loaded_model.predict([[tv, radio, newspaper]])
        predicted_sale = "{:.2f}".format(res[0][0])
        return json.dumps({ "result": predicted_sale })

if __name__ == '__main__':
    app.run()
