"""
app.py — SecureMail 2FA with SQLite database (Flask-SQLAlchemy)
Tables:  users | auth_attempts | keystroke_profiles | login_logs
"""
from flask import Flask, render_template, request, jsonify, send_from_directory
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime, timedelta
from dotenv import load_dotenv
import numpy as np, os, sys, json, random, string

# ── Load .env ─────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(BASE_DIR, '.env'))

sys.path.insert(0, BASE_DIR)

# ── Keys from .env ────────────────────────────────────────────────────────────
FLASK_SECRET_KEY  = os.environ.get('FLASK_SECRET_KEY',  'securemail-keystroke-2fa-secret')
GEMINI_API_KEY    = os.environ.get('GEMINI_API_KEY',    '')
GEMINI_MODEL      = os.environ.get('GEMINI_MODEL',      'gemini-3-flash-preview')
TWILIO_SID        = os.environ.get('TWILIO_SID',        '')
TWILIO_AUTH       = os.environ.get('TWILIO_AUTH',       '')
TWILIO_VERIFY_SID = os.environ.get('TWILIO_VERIFY_SID', '')

# ── In-memory OTP store ───────────────────────────────────────────────────────
otp_store = {}

from data.generator import simulate_keystrokes, extract_features, USERS, FIXED_PASSWORD
from models.classical_ml import predict_user, FEATURE_COLS
from models.anomaly_detection import is_impostor

app = Flask(__name__)

# ── Database config ─────────────────────────────────────────────────────────
app.config['SQLALCHEMY_DATABASE_URI'] = f"sqlite:///{os.path.join(BASE_DIR, 'securemail.db')}"
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['SECRET_KEY'] = FLASK_SECRET_KEY

db = SQLAlchemy(app)

# ── Models ───────────────────────────────────────────────────────────────────

class User(db.Model):
    __tablename__ = 'users'
    id            = db.Column(db.Integer, primary_key=True)
    email         = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(256), nullable=False)
    created_at    = db.Column(db.DateTime, default=datetime.utcnow)

    attempts  = db.relationship('AuthAttempt',      backref='user', lazy=True)
    profiles  = db.relationship('KeystrokeProfile', backref='user', lazy=True)

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)

    def to_dict(self):
        return {'id': self.id, 'email': self.email, 'created_at': self.created_at.isoformat()}


class AuthAttempt(db.Model):
    __tablename__ = 'auth_attempts'
    id               = db.Column(db.Integer, primary_key=True)
    user_id          = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=True)
    email            = db.Column(db.String(120))
    timestamp        = db.Column(db.DateTime, default=datetime.utcnow)
    claimed_user     = db.Column(db.String(100))
    predicted_user   = db.Column(db.String(100))
    identity_match   = db.Column(db.Boolean)
    anomaly_verdict  = db.Column(db.String(20))
    authenticated    = db.Column(db.Boolean)
    features_json    = db.Column(db.Text)

    def to_dict(self):
        return {
            'id':              self.id,
            'email':           self.email,
            'timestamp':       self.timestamp.isoformat(),
            'claimed_user':    self.claimed_user,
            'predicted_user':  self.predicted_user,
            'identity_match':  self.identity_match,
            'anomaly_verdict': self.anomaly_verdict,
            'authenticated':   self.authenticated,
        }


class KeystrokeProfile(db.Model):
    __tablename__ = 'keystroke_profiles'
    id           = db.Column(db.Integer, primary_key=True)
    user_id      = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=True)
    email        = db.Column(db.String(120))
    recorded_at  = db.Column(db.DateTime, default=datetime.utcnow)
    features_json= db.Column(db.Text)

    def to_dict(self):
        return {
            'id':          self.id,
            'email':       self.email,
            'recorded_at': self.recorded_at.isoformat(),
            'features':    json.loads(self.features_json) if self.features_json else {},
        }


class LoginLog(db.Model):
    __tablename__ = 'login_logs'
    id           = db.Column(db.Integer, primary_key=True)
    email        = db.Column(db.String(120), nullable=False)
    attempted_at = db.Column(db.DateTime, default=datetime.utcnow)
    success      = db.Column(db.Boolean, nullable=False)

    def to_dict(self):
        return {
            'id':           self.id,
            'email':        self.email,
            'attempted_at': self.attempted_at.isoformat(),
            'success':      self.success,
        }


# ── Create tables on first run ────────────────────────────────────────────────
with app.app_context():
    db.create_all()
    print("✅  Database ready →", os.path.join(BASE_DIR, 'securemail.db'))


# ── CORS ──────────────────────────────────────────────────────────────────────
@app.after_request
def add_cors(response):
    response.headers['Access-Control-Allow-Origin']  = '*'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type'
    response.headers['Access-Control-Allow-Methods'] = 'GET, POST, OPTIONS'
    return response


# ── Auth endpoint ─────────────────────────────────────────────────────────────
@app.route('/api/authenticate', methods=['POST', 'OPTIONS'])
def authenticate():
    if request.method == 'OPTIONS':
        return '', 204

    data         = request.get_json()
    events       = data.get('events', [])
    claimed_user = data.get('claimed_user', '')
    email        = data.get('email', '')

    if len(events) < 3:
        return jsonify({'error': 'Too few keystrokes'}), 400

    features = extract_features(events)
    if not features:
        return jsonify({'error': 'Feature extraction failed'}), 400

    try:
        classical = predict_user(features, save_dir='results')
    except Exception as e:
        classical = {'error': str(e)}

    try:
        anomaly = is_impostor(features, save_dir='results')
    except Exception as e:
        anomaly = {'error': str(e)}

    if isinstance(classical, dict) and 'error' not in classical:
        votes    = [v['predicted_user'] for v in classical.values()]
        majority = max(set(votes), key=votes.count)
        match    = majority == claimed_user
    else:
        majority = 'unknown'
        match    = False

    authenticated = match and anomaly.get('verdict') == 'GENUINE'

    user = User.query.filter_by(email=email).first()
    attempt = AuthAttempt(
        user_id         = user.id if user else None,
        email           = email,
        claimed_user    = claimed_user,
        predicted_user  = majority,
        identity_match  = match,
        anomaly_verdict = anomaly.get('verdict', 'UNKNOWN'),
        authenticated   = authenticated,
        features_json   = json.dumps(features),
    )
    db.session.add(attempt)
    db.session.commit()

    return jsonify({
        'features':             {k: round(v, 3) if isinstance(v, float) else v for k, v in features.items()},
        'classical_predictions': classical,
        'anomaly_detection':    anomaly,
        'verdict': {
            'predicted_user':  majority,
            'claimed_user':    claimed_user,
            'identity_match':  match,
            'anomaly_verdict': anomaly.get('verdict', 'UNKNOWN'),
            'authenticated':   authenticated,
        },
    })


# ── Register endpoint ─────────────────────────────────────────────────────────
@app.route('/api/register', methods=['POST', 'OPTIONS'])
def register():
    if request.method == 'OPTIONS':
        return '', 204

    data     = request.get_json()
    email    = data.get('email', '').strip().lower()
    password = data.get('password', '')

    if not email or not password:
        return jsonify({'error': 'Email and password required'}), 400

    if User.query.filter_by(email=email).first():
        return jsonify({'error': 'Email already registered'}), 409

    user = User(email=email)
    user.set_password(password)
    db.session.add(user)
    db.session.commit()

    return jsonify({'message': 'User registered successfully', 'user': user.to_dict()}), 201


# ── Login endpoint ────────────────────────────────────────────────────────────
@app.route('/api/login', methods=['POST', 'OPTIONS'])
def login():
    if request.method == 'OPTIONS':
        return '', 204

    data     = request.get_json()
    email    = data.get('email', '').strip().lower()
    password = data.get('password', '')

    user = User.query.filter_by(email=email).first()
    ok   = bool(user and user.check_password(password))

    db.session.add(LoginLog(email=email, success=ok))
    db.session.commit()

    if not ok:
        return jsonify({'error': 'Invalid email or password'}), 401

    return jsonify({'message': 'Password verified — proceed to keystroke step',
                    'user': user.to_dict()})


# ── History endpoint ──────────────────────────────────────────────────────────
# ── Send OTP ──────────────────────────────────────────────────────────────────
@app.route('/api/send-otp', methods=['POST', 'OPTIONS'])
def send_otp():
    if request.method == 'OPTIONS':
        return '', 204

    data       = request.get_json()
    method     = data.get('method', '')      # 'email' or 'sms'
    identifier = data.get('identifier', '').strip()

    if not identifier or method != 'sms':
        return jsonify({'error': 'Provide method (sms) and identifier'}), 400

    if method == 'sms':
        try:
            from twilio.rest import Client
            Client(TWILIO_SID, TWILIO_AUTH).verify \
                .v2.services(TWILIO_VERIFY_SID) \
                .verifications.create(to=identifier, channel='sms')
            masked = identifier[:3] + '****' + identifier[-4:]
            return jsonify({'message': f'OTP sent to {masked}', 'verify': True}), 200
        except Exception as e:
            return jsonify({'error': f'SMS send failed: {str(e)}'}), 500


# ── Verify OTP ────────────────────────────────────────────────────────────────
@app.route('/api/verify-otp', methods=['POST', 'OPTIONS'])
def verify_otp():
    if request.method == 'OPTIONS':
        return '', 204

    data       = request.get_json()
    identifier = data.get('identifier', '').strip()
    otp        = data.get('otp', '').strip()
    # SMS via Twilio Verify
    try:
        from twilio.rest import Client
        result = Client(TWILIO_SID, TWILIO_AUTH).verify \
            .v2.services(TWILIO_VERIFY_SID) \
            .verification_checks.create(to=identifier, code=otp)
        if result.status == 'approved':
            return jsonify({'message': 'OTP verified successfully'}), 200
        else:
            return jsonify({'error': 'Incorrect OTP. Try again.'}), 401
    except Exception as e:
        return jsonify({'error': f'Verification failed: {str(e)}'}), 500


@app.route('/api/history', methods=['GET'])
def history():
    attempts = AuthAttempt.query.order_by(AuthAttempt.timestamp.desc()).limit(50).all()
    return jsonify([a.to_dict() for a in attempts])


@app.route('/api/config', methods=['GET'])
def get_config():
    """Serve non-secret frontend config (Gemini key, model name)."""
    return jsonify({
        'gemini_api_key': GEMINI_API_KEY,
        'gemini_model':   GEMINI_MODEL,
    })


@app.route('/api/login-logs', methods=['GET'])
def login_logs():
    logs = LoginLog.query.order_by(LoginLog.attempted_at.desc()).limit(100).all()
    return jsonify([l.to_dict() for l in logs])


# ── Existing endpoints ────────────────────────────────────────────────────────
@app.route('/api/users')
def get_users():
    return jsonify([{'user_id': u.user_id, 'typing_speed': u.typing_speed} for u in USERS])


@app.route('/api/results')
def get_results():
    path = 'results/summary.json'
    if not os.path.exists(path):
        return jsonify({'error': 'Run train.py first'}), 404
    with open(path) as f:
        return jsonify(json.load(f))


# ── Serve SecureMail as homepage ──────────────────────────────────────────────
@app.route('/')
@app.route('/securemail')
def index():
    if os.path.exists('templates/SecureMail_2FA_AI.html'):
        return render_template('SecureMail_2FA_AI.html')
    return send_from_directory('.', 'SecureMail_2FA_AI.html')


@app.route('/keyauth')
def keyauth():
    return render_template('index.html')


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5050))
    app.run(host='0.0.0.0', port=port, debug=True)
