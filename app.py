from flask import Flask, request, render_template
from sklearn.ensemble import RandomForestClassifier
import pickle

# Training data
X = [[1,1,1], [1,1,0], [0,0,0]]
y = ['Flu', 'Cold', 'Healthy']

# Train model
model = RandomForestClassifier()
model.fit(X, y)

# Save model
pickle.dump(model, open("model.pkl", "wb"))

# Load model
model = pickle.load(open("model.pkl", "rb"))

app = Flask(__name__)

@app.route('/', methods=['GET','POST'])
def home():
    if request.method == 'POST':
        fever = int(request.form['fever'])
        cough = int(request.form['cough'])
        fatigue = int(request.form['fatigue'])

        result = model.predict([[fever, cough, fatigue]])
        return result[0]

    return render_template('index.html')

app.run()
