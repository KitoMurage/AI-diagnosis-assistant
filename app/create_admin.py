import getpass
from app import create_app
from extensions import db
from models import Doctor

# Initialize the Flask application context
app = create_app()

def seed_admin():
    with app.app_context():
        print("\n🛡️  Initializing Admin Creation Sequence...")
        
        # Check if an admin already exists to prevent duplicates
        existing_admin = Doctor.query.filter_by(username='admin').first()
        if existing_admin:
            print("❌ Error: An account with the username 'admin' already exists.")
            return

        # Securely prompt for a password
        print("👤 Username will be set to: admin")
        password = getpass.getpass("🔑 Enter a secure password for the Admin account: ")
        confirm_password = getpass.getpass("🔑 Confirm password: ")

        if password != confirm_password:
            print("❌ Error: Passwords do not match. Aborting.")
            return
            
        if len(password) < 6:
            print("❌ Error: Password must be at least 6 characters long.")
            return

        # Create the superuser
        admin_user = Doctor(
            username='admin',
            department='Hospital Administration'
        )
        admin_user.set_password(password)
        
        db.session.add(admin_user)
        db.session.commit()
        
        print("✅ Success! The master admin account has been securely generated.")
        print("🌐 You may now log in at http://127.0.0.1:5000/login\n")

if __name__ == '__main__':
    seed_admin()