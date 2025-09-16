import joblib
from flask import Flask, request, render_template
import pandas as pd

app = Flask(__name__)

# Load the trained model
model = joblib.load("stacking_model.pkl")

dict_ethnicity = {
    'White-European': 1, 'Asian': 2, 'Unknown': 3, 'Middle Eastern ': 4,
    'Black': 5, 'South Asian': 6, 'Others': 7, 'Latino': 8,
    'Hispanic': 9, 'Pasifika': 10, 'Turkish': 11, 'others': 12
}

dict_country = {
    'United States': 1, 'Other': 2, 'New Zealand': 3, 'India': 4,
    'United Arab Emirates': 5, 'United Kingdom': 6, 'Jordan': 7,
    'Australia': 8, 'Canada': 9, 'Sri Lanka': 10, 'Afghanistan': 11,
    'France': 12, 'Netherlands': 13, 'Brazil': 14, 'Mexico': 15,
    'Iran': 16, 'Russia': 17
}

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/questionnaire')
def questionnaire():
    return render_template('questionnaire.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        a_scores = [int(request.form[f'A{i}_Score']) for i in range(1, 11)]
        age = int(request.form['age'])
        ethnicity = request.form['ethnicity']
        jaundice = 1 if request.form.get('jundice') == 'yes' else 0
        austim = 1 if request.form['austim'] == 'yes' else 0
        country = request.form['country']
        result = sum(a_scores)

        ethnicity_encoded = dict_ethnicity.get(ethnicity, 3)
        country_encoded = dict_country.get(country, 2)

        input_data = pd.DataFrame({
            'A1_Score': [a_scores[0]], 'A2_Score': [a_scores[1]], 'A3_Score': [a_scores[2]],
            'A4_Score': [a_scores[3]], 'A5_Score': [a_scores[4]], 'A6_Score': [a_scores[5]],
            'A7_Score': [a_scores[6]], 'A8_Score': [a_scores[7]], 'A9_Score': [a_scores[8]],
            'A10_Score': [a_scores[9]], 'age': [age], 'ethnicity': [ethnicity_encoded],
            'jundice': [jaundice], 'austim': [austim], 'contry_of_res': [country_encoded],
            'result': [result]
        })

        prediction = model.predict(input_data)[0]
        if prediction == 'YES':
            result_text = "Likely to have Autism Spectrum Disorder"
        else:
            result_text = "Unlikely to have Autism Spectrum Disorder"

        explanation = "This prediction is based on behavioral and demographic data. Please consult a healthcare professional for an official diagnosis."

        return render_template('result.html', prediction=result_text, explanation=explanation)

    except Exception as e:
        return render_template('result.html', prediction="Error: " + str(e), explanation="")

if __name__ == '__main__':
    app.run(debug=True)
