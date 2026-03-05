from flask import Flask, render_template, request
import pickle

app= Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

@app.route("/")
def web_page_gen ():
  return render_template('index.html')

@app.route("/predict", methods =['POST'])
def predict():
  Practice1 = int(request.form['Practice 1'])
  Practice2 = int(request.form['Practice 2'])
  prediction = model.predict([[Practice1, Practice2]])
  output = round(prediction[0])
  return render_template('index.html',
   answer = f'Scores of {Practice1} and {Practice2} in the practice test \
     indicate a score of {output} in the final test')

if __name__ == "__main__":
    app.run()
    