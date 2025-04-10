from flask import Flask, jsonify, request
from datetime import datetime, timedelta
import threading

app = Flask(__name__)
lock = threading.Lock()

# In-memory storage (replace with database in production)
sessions = {}
players = {}

# Security settings
API_KEY = "Secret"
ALLOWED_ORIGINS = [
    "http://localhost",
    "http://127.0.0.1",
]

@app.before_request
def validate_request():
    if request.headers.get('X-API-Key') != API_KEY:
        return jsonify({"error": "Unauthorized"}), 401
    
    origin = request.origin or "null"
    if origin not in ALLOWED_ORIGINS + ["null"]:
        return jsonify({"error": f"Origin {origin} not allowed"}), 403

@app.after_request
def add_cors_headers(response):
    # Filter out None values and join origins
    allowed_origins = [origin for origin in ALLOWED_ORIGINS if origin is not None]
    response.headers['Access-Control-Allow-Origin'] = ','.join(allowed_origins)
    response.headers['Access-Control-Allow-Headers'] = 'X-API-Key, Content-Type'
    response.headers['Access-Control-Allow-Methods'] = 'POST, GET'
    return response

@app.route('/register', methods=['POST'])
def register_player():
    with lock:
        data = request.json
        session_id = data['session_id']
        player_id = data.get('player_id', str(uuid.uuid4()))  # Generate unique player ID
        
        # Create session if not exists
        if session_id not in sessions:
            sessions[session_id] = {
                'total_players': data['total_players'],
                'ready_players': set(),
                'start_time': None
            }
        
        # Register player
        players[player_id] = {
            'session_id': session_id,
            'ready': False,
            'last_seen': datetime.now().isoformat()
        }
        
        return jsonify({"player_id": player_id, "status": "registered"})

@app.route('/ready', methods=['POST'])
def mark_ready():
    with lock:
        data = request.json
        player_id = data['player_id']
        
        if player_id not in players:
            return jsonify({"error": "Player not registered"}), 404
            
        session_id = players[player_id]['session_id']
        players[player_id]['ready'] = True
        players[player_id]['last_seen'] = datetime.now().isoformat()
        
        # Add to ready players
        sessions[session_id]['ready_players'].add(player_id)
        
        # Update start time when first player is ready
        if len(sessions[session_id]['ready_players']) == 1:
            sessions[session_id]['start_time'] = datetime.now().isoformat()
            
        return jsonify({
            "ready_count": len(sessions[session_id]['ready_players']),
            "total_players": sessions[session_id]['total_players']
        })

@app.route('/status/<session_id>')
def get_status(session_id):
    with lock:
        if session_id not in sessions:
            return jsonify({"error": "Session not found"}), 404
            
        return jsonify({
            "total_players": sessions[session_id]['total_players'],
            "ready_players": sessions[session_id]['ready_count'],
            "start_time": sessions[session_id].get('start_time')  # Safe get
        })

if __name__ == '__main__':
    app.run(host='0.0.0.0', threaded=True, port=8044, debug=True)
