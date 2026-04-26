from flask import Flask
from extensions import db
from routes import main
from models import Doctor
from flask_login import LoginManager

def create_app():
    print("⚙️ Initializing Enterprise AI Copilot Server...")
    app = Flask(__name__)
    app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///clinic.db'
    app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
    
    app.secret_key = 'diagnosis_assistant_secret_key'
    
    # Initialize Database
    db.init_app(app)
    
    # Initialize Security Manager
    login_manager = LoginManager()
    login_manager.login_view = 'main.login' # redirect if unauth
    login_manager.init_app(app)
    
    # load current logged in user
    @login_manager.user_loader
    def load_user(user_id):
        return Doctor.query.get(int(user_id))
    
    # Register Routes
    app.register_blueprint(main)
    
    with app.app_context():
        db.create_all()
        
    return app

if __name__ == '__main__':
    app = create_app()
    print("🚀 Enterprise System Online. Access at http://127.0.0.1:5000")
    app.run(debug=True, port=5000)