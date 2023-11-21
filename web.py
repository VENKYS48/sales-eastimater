from flask import Flask, render_template, request
import pickle
import warnings

app = Flask(__name__)

warnings.filterwarnings('ignore')

# Loading the model
MODEL_PATH = "multiple_reg.sav"
loaded_model = pickle.load(open(MODEL_PATH, 'rb'))

# Render the main page
@app.route('/')
def index():
    return render_template('index.html')

# Calculate and return the sales prediction
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        tv = float(request.form['tv'])
        radio = float(request.form['radio'])
        newspaper = float(request.form['newspaper'])
        
        res = loaded_model.predict([[tv, radio, newspaper]])
        predicted_sale = "{:.2f}".format(res[0][0])
        return render_template('result.html', sale=predicted_sale)

if __name__ == '__main__':
    app.run(debug=True)
