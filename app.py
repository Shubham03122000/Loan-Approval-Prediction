import math
from flask import Flask, render_template, request
import pickle
import numpy as np

model = pickle.load(open('model.pkl', 'rb'))

app = Flask(__name__)


@app.route('/')
def man():
    return render_template('index.html')

@app.route('/Home')
def h1():
    return render_template('Home.html')

@app.route('/Report')
def report():
    return render_template('Report.html')

@app.route('/predict', methods=['POST'])
def home():
    Gender = request.form['Gender']
    Married = request.form['Married']
    Dependents = request.form['Dependents']
    Education = request.form['Education']
    Self_Employed = request.form['Self_Employed']
    ApplicantIncome = request.form['ApplicantIncome']
    CoapplicantIncome = request.form['CoapplicantIncome']
    LoanAmount = request.form['LoanAmount']
    Loan_Amount_Term = request.form['Loan_Amount_Term']
    Credit_History = request.form['Credit_History']
    Property_Area = request.form['Property_Area']

    TotalIncome = int(ApplicantIncome) + int(CoapplicantIncome)
    LoanAmountLog = LoanAmount

    arr = np.array([[Gender,Married,Dependents,Education,Self_Employed,Loan_Amount_Term,Credit_History,Property_Area,TotalIncome,LoanAmountLog]])
    pred = model.predict(arr)
    print(arr)

    return render_template('result.html',data=pred)



if __name__ == "__main__":
    app.run(debug=True)
