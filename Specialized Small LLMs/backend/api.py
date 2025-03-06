from flask import Flask, request, jsonify
from .models import db, User  # Assumes models.py is in the same folder
from flask_cors import CORS
import os

app = Flask(__name__)
basedir = os.path.abspath(os.path.dirname(__file__))
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///' + os.path.join(basedir, 'app.db')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
CORS(app)
db.init_app(app)

with app.app_context():
    db.create_all()

@app.route('/register', methods=['POST'])
def register():
    print("Register endpoint hit")  # Debug statement
    data = request.get_json()
    print(f"Received data: {data}")  # Debug statement
    
    if not data:
        print("No data received")  # Debug statement
        return jsonify({'message': 'No data received'}), 400

    # Check if a user with the same username or email already exists.
    existing_user = User.query.filter((User.username == data.get('username')) | (User.email == data.get('email'))).first()
    if existing_user:
        print("User already exists")  # Debug statement
        return jsonify({'message': 'User already exists'}), 409

    try:
        new_user = User(
            name=data.get('name'),
            username=data.get('username'),
            email=data.get('email'),
            password=data.get('password')
        )
        db.session.add(new_user)
        db.session.commit()
        print("User registered successfully")  # Debug statement
        return jsonify({'message': 'User registered successfully'}), 201
    except Exception as e:
        print(f"Error during registration: {e}")  # Debug statement
        return jsonify({'message': 'Registration failed'}), 500

@app.route('/login', methods=['POST'])
def login():
    data = request.get_json()
    user = User.query.filter((User.username == data.get('identifier')) | (User.email == data.get('identifier'))).first()
    if user and user.password == data.get('password'):
        return jsonify({'message': 'Login successful'}), 200
    return jsonify({'message': 'Invalid credentials'}), 401

if __name__ == "__main__":
    app.run(debug=True, port=5001)

