from flask import Flask, render_template, request, redirect, url_for
import folium
from geopy.geocoders import Nominatim
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline

app = Flask(__name__)

# Data for various app components
educational_content = [
    {"title": "Recognizing Signs of Violence", "content": "Learn to identify behavioral signs that might indicate potential violence."},
    {"title": "Emergency Procedures", "content": "Steps to take in emergency situations involving active threats."}
]

events = [
    {"name": "Community Safety Workshop", "date": "2024-04-30"},
    {"name": "Gun Safety Course", "date": "2024-05-15"}
]

feedback = []

# Load the dataset from file
def load_dataset(file_path):
    data = []
    with open(file_path, 'r') as file:
        for line in file:
            question, label = line.strip().split(" - ")
            data.append((question, label))
    return data

# Load and prepare data for training
dataset_path = 'dataset.txt' 
training_data = load_dataset(dataset_path)
model = make_pipeline(TfidfVectorizer(), MultinomialNB())
model.fit([text for text, label in training_data], [label for text, label in training_data])


# Route to show the prediction form
@app.route('/predict_form')
def show_predict_form():
    return render_template('predict_form.html')

# Route to handle the prediction logic
@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        query = request.form.get('query')
        if query:
            prediction = model.predict([query])[0]
            return render_template('predict_form.html', query=query, prediction=prediction)
        else:
            return render_template('predict_form.html', error="Please provide a query.")
    else:
        return render_template('predict_form.html')

# Additional routes for the application
@app.route('/')
def home():
    return render_template('home.html')

@app.route('/educational')
def educational():
    return render_template('educational.html', contents=educational_content)

@app.route('/map')
def map_view():
    geolocator = Nominatim(user_agent="app")
    location = geolocator.geocode("New York")
    m = folium.Map(location=[location.latitude, location.longitude], zoom_start=12)
    folium.Marker([location.latitude, location.longitude], popup='Counseling Services').add_to(m)
    m.save('static/map.html')
    return render_template('map.html')

@app.route('/events')
def events():
    return render_template('events.html', events=events)

@app.route('/feedback', methods=['GET', 'POST'])
def user_feedback():
    if request.method == 'POST':
        user_feedback = request.form.get('feedback')
        feedback.append(user_feedback)
        return redirect(url_for('user_feedback'))
    return render_template('feedback.html', feedback=feedback)

if __name__ == '__main__':
    app.run(debug=True)
