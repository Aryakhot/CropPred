import os
print("Current Working Directory:", os.getcwd())

from flask import Flask, request, render_template, redirect, url_for, session, flash
from flask_sqlalchemy import SQLAlchemy
import numpy as np
import joblib
import bcrypt

# Use absolute paths to the model and scaler files
model_path = '/Users/apple/Desktop/mini project/crop_recommend_model.pkl'
scaler_path = '/Users/apple/Desktop/mini project/scal.pkl'

try:
    model = joblib.load('crop_recommend_model.pkl')
    scaler = joblib.load('scal.pkl')
except FileNotFoundError as e:
    print(f"Error loading files: {e}")
    raise

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///database.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.secret_key = 'your_secret_key'
db = SQLAlchemy(app)

class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    username = db.Column(db.String(100), unique=True, nullable=False)
    contact_no = db.Column(db.String(20), nullable=False)
    dob = db.Column(db.String(10), nullable=False)
    zip_code = db.Column(db.String(10), nullable=False)
    password = db.Column(db.String(100), nullable=False)

    def __init__(self, name, username, contact_no, dob, zip_code, password):
        self.name = name
        self.username = username
        self.contact_no = contact_no
        self.dob = dob
        self.zip_code = zip_code
        self.password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')

    def check_password(self, password):
        return bcrypt.checkpw(password.encode('utf-8'), self.password.encode('utf-8'))

# Initialize database
with app.app_context():
    db.create_all()


@app.route('/')
def home():
    # Check if the user is logged in
    if 'user_id' not in session:
        return redirect(url_for('login'))  # Redirect to login page if not logged in
    return render_template('home.html') 

@app.route('/registration_form', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        # Handle the registration form submission
        first_name = request.form.get('firstName')
        last_name = request.form.get('lastName')
        name = f"{first_name} {last_name}"
        username = request.form.get('username')
        contact_no = request.form.get('contactNo')
        dob = request.form.get('dob')
        zip_code = request.form.get('zipCode')
        password = request.form.get('password')
        confirm_password = request.form.get('confirmPassword')

        if password != confirm_password:
            return "Passwords do not match", 400  # handle password mismatch

        user = User(name=name, username=username, contact_no=contact_no, dob=dob, zip_code=zip_code, password=password)
        db.session.add(user)
        db.session.commit()
        return redirect(url_for('login'))  # Adjust URL as needed
    return render_template('registration_form.html')  # Render the registration form


@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')

        # Check if user exists in the database
        user = User.query.filter_by(username=username).first()

        if user and user.check_password(password):
            session['user_id'] = user.id
            flash("Login successful!", "success")
            return redirect(url_for('home'))  # Redirect to homepage
        else:
            flash("Invalid username or password", "danger")
            return redirect(url_for('login'))

    return render_template('login.html')


@app.route('/weather-updates')
def weather_updates():
    return render_template('weather_updates.html') 

@app.route('/crop-recommendation')
def crop_recommendation():
    if 'user_id' not in session:
        return redirect(url_for('login'))  # Redirect to login if not logged in
    return render_template('crop_recommendation.html')

@app.route('/logout')
def logout():
    # Remove user_id from the session, logging them out
    session.pop('user_id', None)
    flash("You have been logged out.", "success")
    return redirect(url_for('login'))  # Redirect to login page

@app.route('/contactUs')
def contact():
    return render_template('contactUs.html')  # Serve the contactUs.html page


@app.route("/predict", methods=['POST'])
def predict():
    N = request.form['Nitrogen']
    P = request.form['Phosphorus']
    K = request.form['Potassium']
    temp = request.form['Temperature']
    humidity = request.form['Humidity']
    ph = request.form['pH']
    rainfall = request.form['Rainfall']

    # Prepare the feature list
    feature_list = [N, P, K, temp, humidity, ph, rainfall]
    single_pred = np.array(feature_list).reshape(1, -1).astype(float)

    # Scale the input data
    single_pred_scaled = scaler.transform(single_pred)

    # Predict the crop
    predicted_crop = model.predict(single_pred_scaled)

    # Directly use the prediction if it returns a crop name
    result = f"{predicted_crop[0]} is the best crop to be cultivated right there."

    return render_template('crop_recommendation.html', result=result)

if __name__ == "__main__":
    app.run(debug=True)
