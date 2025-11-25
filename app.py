from flask import Flask, render_template, request, redirect, url_for, flash, send_file, jsonify, session
from flask_sqlalchemy import SQLAlchemy
from auth import bcrypt, login_required, get_current_user
from models import db, User, Certificate
from certificate_predictor import predict_certificate_authenticity
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-here-change-this-in-production'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///academia.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['UPLOAD_FOLDER'] = 'static/uploads/'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Initialize extensions
db.init_app(app)
bcrypt.init_app(app)

# Create upload directory
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'pdf'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')
        
        user = User.query.filter_by(email=email).first()
        if user and bcrypt.check_password_hash(user.password, password):
            session['user_id'] = user.id
            flash('Login successful!', 'success')
            return redirect(url_for('predictor'))
        else:
            flash('Invalid email or password', 'error')
    
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form.get('username')
        email = request.form.get('email')
        password = request.form.get('password')
        
        if User.query.filter_by(email=email).first():
            flash('Email already exists', 'error')
            return render_template('register.html')
        
        hashed_password = bcrypt.generate_password_hash(password).decode('utf-8')
        user = User(username=username, email=email, password=hashed_password)
        db.session.add(user)
        db.session.commit()
        
        flash('Registration successful! Please login.', 'success')
        return redirect(url_for('login'))
    
    return render_template('register.html')

@app.route('/logout')
def logout():
    session.pop('user_id', None)
    flash('Logged out successfully!', 'success')
    return redirect(url_for('index'))

@app.route('/predictor', methods=['GET', 'POST'])
@login_required
def predictor():
    prediction = None
    if request.method == 'POST':
        if 'certificate' not in request.files:
            flash('No file selected', 'error')
            return redirect(request.url)
        
        file = request.files['certificate']
        if file.filename == '':
            flash('No file selected', 'error')
            return redirect(request.url)
        
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            # Call your DL model for prediction
            try:
                prediction = predict_certificate_authenticity(filepath)
                
                if 'error' in prediction:
                    flash(f'Analysis error: {prediction["error"]}', 'error')
                    prediction = None
                else:
                    flash('Certificate analyzed successfully!', 'success')
                    
            except Exception as e:
                flash(f'Error analyzing certificate: {str(e)}', 'error')
                prediction = None
        else:
            flash('Invalid file type. Please upload PNG, JPG, JPEG, or PDF.', 'error')
    
    return render_template('predictor.html', prediction=prediction)

@app.route('/certificates')
@login_required
def certificates():
    user = get_current_user()
    user_certificates = Certificate.query.filter_by(user_id=user.id).all()
    return render_template('certificates.html', certificates=user_certificates)

@app.route('/upload_certificate', methods=['GET', 'POST'])
@login_required
def upload_certificate():
    if request.method == 'POST':
        name = request.form.get('name')
        description = request.form.get('description')
        
        if 'certificate_file' not in request.files:
            flash('No file selected', 'error')
            return redirect(request.url)
        
        file = request.files['certificate_file']
        if file.filename == '':
            flash('No file selected', 'error')
            return redirect(request.url)
        
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            user = get_current_user()
            certificate = Certificate(
                name=name,
                description=description,
                filename=filename,
                filepath=filepath,
                user_id=user.id
            )
            db.session.add(certificate)
            db.session.commit()
            
            flash('Certificate uploaded successfully!', 'success')
            return redirect(url_for('certificates'))
        else:
            flash('Invalid file type', 'error')
    
    return render_template('upload_certificate.html')

@app.route('/download_certificate/<int:certificate_id>')
@login_required
def download_certificate(certificate_id):
    user = get_current_user()
    certificate = Certificate.query.filter_by(id=certificate_id, user_id=user.id).first_or_404()
    return send_file(certificate.filepath, as_attachment=True, download_name=certificate.filename)

@app.route('/faqs')
def faqs():
    return render_template('faqs.html')

@app.route('/about')
def about():
    return render_template('about.html')

# Add this to make get_current_user available in all templates
@app.context_processor
def utility_processor():
    return dict(get_current_user=get_current_user)

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(debug=True)