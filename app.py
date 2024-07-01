from flask import Flask, request, render_template, flash
from main import predict

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def home_predict():
    prediction = None
    if request.method == 'POST':
        # Check if the POST request has a file part
        if 'file' not in request.files:
            flash('Please enter a file')

        file = request.files['file']

        # If user does not select file, browser also
        # submit an empty part without filename
        if file.filename == '':
            flash('Please select a file')

        if file:
            # Call your prediction function with the audio file
            prediction = predict(file)
        
    return render_template("predict.html", prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True, port=8000)

