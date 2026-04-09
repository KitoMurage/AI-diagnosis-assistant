from flask import Flask
from extensions import db
from routes import main

def create_app():
    app = Flask(__name__)
    app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///patients.db'
    app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
    app.secret_key = 'demo_secret_key_123'
    
    # Initialize Database
    db.init_app(app)
    
    # Register Routes
    app.register_blueprint(main)
    
    with app.app_context():
        db.create_all()
        
    return app

if __name__ == '__main__':
    app = create_app()
    print("🚀 System Online. Access at http://127.0.0.1:5000")
    app.run(debug=True, port=5000)