import joblib
from flask import Flask, redirect, url_for, request, render_template
import numpy as np

# Define a flask app
app = Flask(__name__)

model = joblib.load(open("rock_mine_rf.pkl",'rb'))

@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')

@app.route('/predict',  methods=['GET','POST'])
def predict():
        # freq1 = request.form.get('freq1')
        # freq2 = request.form.get('freq2')
        # freq3 = request.form.get('freq3')
        # freq4 = request.form.get('freq4')
        # freq5 = request.form.get('freq5')
        # freq6 = request.form.get('freq6')
        # freq7 = request.form.get('freq7')
        # freq8 = request.form.get('freq8')
        # freq9 = request.form.get('freq9')
        # freq10 = request.form.get('freq10')
        # freq11 = request.form.get('freq11')
        # freq12 = request.form.get('freq12')
        # freq13 = request.form.get('freq13')
        # freq14 = request.form.get('freq14')
        # freq15 = request.form.get('freq15')
        # freq16 = request.form.get('freq16')
        # freq17 = request.form.get('freq17')
        # freq18 = request.form.get('freq18')
        # freq19 = request.form.get('freq19')
        # freq20 = request.form.get('freq20')
        # freq21 = request.form.get('freq21')
        # freq22 = request.form.get('freq22')
        # freq23 = request.form.get('freq23')
        # freq24 = request.form.get('freq24')
        # freq25 = request.form.get('freq25')
        # freq26 = request.form.get('freq26')
        # freq27 = request.form.get('freq27')
        # freq28 = request.form.get('freq28')
        # freq29 = request.form.get('freq29')
        # freq30 = request.form.get('freq30')
        # freq31 = request.form.get('freq31')
        # freq32 = request.form.get('freq32')
        # freq33 = request.form.get('freq33')
        # freq34 = request.form.get('freq34')
        # freq35 = request.form.get('freq35')
        # freq36 = request.form.get('freq36')
        # freq37 = request.form.get('freq37')
        # freq38 = request.form.get('freq38')
        # freq39 = request.form.get('freq39')
        # freq40 = request.form.get('freq40')
        # freq41 = request.form.get('freq41')
        # freq42 = request.form.get('freq42')
        # freq43 = request.form.get('freq43')
        # freq44 = request.form.get('freq44')
        # freq45 = request.form.get('freq45')
        # freq46 = request.form.get('freq46')
        # freq47 = request.form.get('freq47')
        # freq48 = request.form.get('freq48')
        # freq49 = request.form.get('freq49')
        # freq50 = request.form.get('freq50')
        # freq51 = request.form.get('freq51')
        # freq52 = request.form.get('freq52')
        # freq53 = request.form.get('freq53')
        # freq54 = request.form.get('freq54')
        # freq55 = request.form.get('freq55')
        # freq56 = request.form.get('freq56')
        # freq57 = request.form.get('freq57')
        # freq58 = request.form.get('freq58')
        # freq59 = request.form.get('freq59')
        # freq60 = request.form.get('freq60')


        # input_data = [freq1,freq2,freq3,freq4,freq5,freq6,freq7,freq8,freq9,freq10,freq11,freq12,freq13,freq14,freq15,freq16,freq17,freq18,freq19,freq20,freq21,freq22,freq23,freq24,freq25,freq26,freq27,freq28,freq29,freq30,
        #               freq31,freq32,freq33,freq34,freq35,freq36,freq37,freq38,freq39,freq40,freq41,freq42,freq43,freq44,freq45,freq46,freq47,freq48,freq49,freq50,freq51,freq52,freq53,freq54,freq55,freq56,freq57,freq58,freq59,freq60]
        # input_data_as_numpy_array = np.asarray(input_data)
        # input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)
        # input_data_reshaped = list(input_data_reshaped)
        # prediction = model.predict(input_data_reshaped)

        # # if prediction[0]==1:
        # #     prediction = "Diabetes"
        # #     return redirect(url_for('diabetic_retino'))
        # # else:
        # #     prediction = "You Are Safe"
        
        # return prediction
    input_data = [request.form[f'freq{i}'] for i in range(1, 61)]
    input_data_as_numpy_array = np.asarray(input_data).reshape(1, -1)
    prediction = model.predict(input_data_as_numpy_array)
    prediction = str(prediction)
    if prediction == ['M']:
        prediction =  "underwater target is Mine"
    else:
        prediction =  "underwater target is rock"
    return render_template('index.html', prediction=prediction)


if __name__ == '__main__':
    app.run(port=5001,debug=True)