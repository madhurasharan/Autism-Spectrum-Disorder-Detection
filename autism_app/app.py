import joblib
from flask import Flask, request, render_template, send_file, url_for
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend to avoid GUI warning
import matplotlib.pyplot as plt
import os
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter

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

# Extract options for dropdowns
ethnicity_options = list(dict_ethnicity.keys())
country_options = list(dict_country.keys())

@app.route('/')
def home():
    return render_template('index.html', nav=True)

@app.route('/questionnaire')
def questionnaire():
    return render_template('questionnaire.html', nav=True, ethnicity_options=ethnicity_options, country_options=country_options)

@app.route('/about')
def about():
    return render_template('about.html', nav=True)

@app.route('/contributors')
def contributors():
    return render_template('contributors.html', nav=True)

@app.route('/download_pdf')
def download_pdf():
    return send_file('static/result.pdf', as_attachment=True)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        a_scores = [int(request.form[f'A{i}_Score']) for i in range(1, 11)]
        age = int(request.form['age'])
        ethnicity = request.form['ethnicity']
        jaundice = 1 if request.form.get('jaundice') == 'yes' else 0
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
        probabilities = model.predict_proba(input_data)[0]
        prob_yes = probabilities[1] * 100  # Assuming YES is index 1
        prob_no = probabilities[0] * 100

        if prediction == 'YES':
            result_text = "Likely to have Autism Spectrum Disorder"
        else:
            result_text = "Unlikely to have Autism Spectrum Disorder"

        explanation = "This prediction is based on behavioral and demographic data. Please consult a healthcare professional for an official diagnosis."

        # Generate chart
        chart_path = 'static/chart.png'
        plt.figure(figsize=(6, 4))
        plt.bar(['Unlikely', 'Likely'], [prob_no, prob_yes], color=['blue', 'red'])
        plt.ylabel('Probability (%)')
        plt.title('Prediction Probabilities')
        plt.ylim(0, 100)
        plt.savefig(os.path.join(app.root_path, chart_path))
        plt.close()

        # Generate PDF
        pdf_path = 'static/result.pdf'
        c = canvas.Canvas(os.path.join(app.root_path, pdf_path), pagesize=letter)
        c.drawString(100, 750, "Autism Screening Result")
        c.drawString(100, 730, f"Prediction: {result_text}")
        c.drawString(100, 710, f"Probability Likely: {prob_yes:.2f}%")
        c.drawString(100, 690, f"Probability Unlikely: {prob_no:.2f}%")
        c.drawString(100, 670, explanation)
        c.save()

        return render_template('result.html', prediction=result_text, explanation=explanation, probability=f"{prob_yes:.2f}%", chart_path=chart_path, pdf_path=pdf_path)

    except Exception as e:
        return render_template('result.html', prediction="Error: " + str(e), explanation="", probability=None, chart_path=None, pdf_path=None)

if __name__ == '__main__':
    app.run(debug=True)
