from flask import Flask, request, jsonify
from backend.models import db, User
from backend.finetune import FineTuner

app = Flask(__name__)

@app.route('/login', methods=['POST'])
def login():
    # ... existing code ...
    user = User.query.filter_by(username=data['username']).first()
    # ... existing code ...

@app.route('/finetune', methods=['POST'])
def finetune():
    # ... existing code ...
    finetuner = FineTuner(model_name, dataset_path)
    result = finetuner.run()
    # ... existing code ...

# ... existing code ... 