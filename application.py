import pickle
from flask import Flask, request, render_template
import pandas as pd

app = Flask(__name__)

# Load model and scaler
ridge_model = pickle.load(open('models/ridge.pkl','rb'))
standard_scaler = pickle.load(open('models/scaler.pkl','rb'))

# Define feature names in the same order as used during training
feature_cols = ['Temperature','RH','Ws','Rain','FFMC','DMC','ISI','Classes','Region']

@app.route("/")
def index():
    return render_template('index.html')

@app.route('/predictdata', methods=['GET','POST'])
def predict_datapoint():
    if request.method == "POST":
        # Collect form data and convert to float
        input_data = [float(request.form.get(col)) for col in feature_cols]

        # Convert to DataFrame with proper column names
        input_df = pd.DataFrame([input_data], columns=feature_cols)

        # Scale
        new_data_scaled = standard_scaler.transform(input_df)

        # Predict
        result = ridge_model.predict(new_data_scaled)[0]

        # Render template with result
        return render_template('home.html', result=result)
    else:
        return render_template('home.html')

if __name__=="__main__":
    app.run(host="0.0.0.0")
