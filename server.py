from flask import Flask, jsonify, request
from datetime import datetime, timedelta
import threading
import uuid
import time

app = Flask(__name__)
lock = threading.Lock()

# Server configuration
SERVER_CONFIG = {
    "session_timeout": 3600,  # 1 hour
    "player_timeout": 30,    # 5 minutes
    "move_delay": 2,          # seconds between player movements
    "default_game_link": "",
    "enable_networking": True
}

# In-memory storage
sessions = {}
players = {}

# Security settings
API_KEY = "Secret"
ALLOWED_ORIGINS = ["http://localhost", "http://127.0.0.1"]

def clean_expired_sessions():
    while True:
        with lock:
            now = datetime.now()
            # Clean expired sessions
            expired_sessions = [
                sid for sid, session in sessions.items()
                if (now - datetime.fromisoformat(session['last_activity'])) > 
                   timedelta(seconds=SERVER_CONFIG['session_timeout'])
            ]
            
            for sid in expired_sessions:
                del sessions[sid]
            
            # Clean inactive players
            for pid in list(players.keys()):
                if (now - datetime.fromisoformat(players[pid]['last_seen'])) > \
                   timedelta(seconds=SERVER_CONFIG['player_timeout']):
                    session_id = players[pid]['session_id']
                    if session_id in sessions:
                        sessions[session_id]['players'].discard(pid)
                        sessions[session_id]['disconnected_players'].add(pid)
                    del players[pid]
            
            # Check for auto-start conditions
            for sid, session in sessions.items():
                if not session.get('started', False) and \
                   (now - datetime.fromisoformat(session['created_at'])) > \
                   timedelta(seconds=SERVER_CONFIG['auto_start_time']):
                    session['started'] = True
                    session['ready_players'] = set(session['players'])  # Mark all as ready
        
        time.sleep(60)

# Start cleaner thread
cleaner_thread = threading.Thread(target=clean_expired_sessions, daemon=True)
cleaner_thread.start()

@app.before_request
def validate_request():
    if not SERVER_CONFIG['enable_networking']:
        return jsonify({"error": "Networking disabled"}), 503
        
    if request.headers.get('X-API-Key') != API_KEY:
        return jsonify({"error": "Unauthorized"}), 401
    
    origin = request.origin or "null"
    if origin not in ALLOWED_ORIGINS + ["null"]:
        return jsonify({"error": f"Origin {origin} not allowed"}), 403

@app.route('/register', methods=['POST'])
def register_player():
    with lock:
        try:
            data = request.json
            session_id = data['session_id']
            
            # Check if session exists and has space
            if session_id in sessions:
                if len(sessions[session_id]['players']) >= SERVER_CONFIG['max_players']:
                    return jsonify({"error": "Party is full"}), 400
            
            # Create new session if needed
            if session_id not in sessions:
                sessions[session_id] = {
                    'players': set(),
                    'ready_players': set(),
                    'disconnected_players': set(),
                    'created_at': datetime.now().isoformat(),
                    'last_activity': datetime.now().isoformat(),
                    'started': False,
                    'current_move_order': 0,
                    'current_sequence': 0
                }
            
            # Create new player
            player_id = str(uuid.uuid4())
            players[player_id] = {
                'session_id': session_id,
                'last_seen': datetime.now().isoformat()
            }
            sessions[session_id]['players'].add(player_id)
            sessions[session_id]['last_activity'] = datetime.now().isoformat()
            
            return jsonify({
                "player_id": player_id,
                "session_id": session_id,
                "total_players": len(sessions[session_id]['players']),
                "max_players": SERVER_CONFIG['max_players']
            })
            
        except Exception as e:
            return jsonify({"error": str(e)}), 500

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
        sessions[session_id]['last_activity'] = datetime.now().isoformat()
        sessions[session_id]['ready_players'].add(player_id)
        
        return jsonify({
            "ready_count": len(sessions[session_id]['ready_players']),
            "total_players": sessions[session_id]['total_players']
        })

@app.route('/loaded', methods=['POST'])
def mark_loaded():
    with lock:
        data = request.json
        player_id = data['player_id']
        
        if player_id not in players:
            return jsonify({"error": "Player not registered"}), 404
            
        players[player_id]['loaded'] = True
        players[player_id]['last_seen'] = datetime.now().isoformat()
        session_id = players[player_id]['session_id']
        sessions[session_id]['last_activity'] = datetime.now().isoformat()
        
        # Check if all players are loaded
        all_loaded = all(
            players[p]['loaded'] 
            for p in sessions[session_id]['players_order'] 
            if not players[p]['disconnected']
        )
        
        return jsonify({
            "all_loaded": all_loaded,
            "move_sequence": sessions[session_id]['move_sequence']
        })

@app.route('/ping', methods=['POST'])
def player_ping():
    with lock:
        data = request.json
        player_id = data['player_id']
        
        if player_id not in players:
            return jsonify({"error": "Player not found"}), 404
            
        players[player_id]['last_seen'] = datetime.now().isoformat()
        session_id = players[player_id]['session_id']
        sessions[session_id]['last_activity'] = datetime.now().isoformat()
        
        return jsonify({"status": "ok"})

@app.route('/disconnect', methods=['POST'])
def mark_disconnected():
    with lock:
        data = request.json
        player_id = data['player_id']
        
        if player_id not in players:
            return jsonify({"error": "Player not registered"}), 404
            
        session_id = players[player_id]['session_id']
        players[player_id]['disconnected'] = True
        sessions[session_id]['disconnected_players'].add(player_id)
        sessions[session_id]['last_activity'] = datetime.now().isoformat()
        
        # Notify other players
        return jsonify({
            "should_restart": True,
            "disconnected_player": player_id
        })

@app.route('/status/<session_id>')
def get_status(session_id):
    with lock:
        if session_id not in sessions:
            return jsonify({"error": "Session not found"}), 404
            
        session = sessions[session_id]
        return jsonify({
            "total_players": len(session['players']),
            "ready_players": len(session['ready_players']),
            "disconnected_players": list(session['disconnected_players']),
            "started": session['started'],
            "current_move_order": session['current_move_order'],
            "current_sequence": session['current_sequence']
        })

@app.route('/next_move', methods=['POST'])
def next_move():
    with lock:
        data = request.json
        player_id = data['player_id']
        
        if player_id not in players:
            return jsonify({"error": "Player not registered"}), 404
            
        session_id = players[player_id]['session_id']
        sessions[session_id]['move_sequence'] += 1
        sessions[session_id]['last_activity'] = datetime.now().isoformat()
        
        return jsonify({
            "move_order": players[player_id]['order'],
            "current_sequence": sessions[session_id]['move_sequence']
        })
        
@app.route('/complete_move', methods=['POST'])
def complete_move():
    with lock:
        data = request.json
        player_id = data['player_id']
        
        if player_id not in players:
            return jsonify({"error": "Player not registered"}), 404
            
        session_id = players[player_id]['session_id']
        
        # Update the move sequence
        sessions[session_id]['current_move_order'] = (sessions[session_id].get('current_move_order', 0)) + 1
        if sessions[session_id]['current_move_order'] >= sessions[session_id]['total_players']:
            sessions[session_id]['current_move_order'] = 0
            sessions[session_id]['current_sequence'] = sessions[session_id].get('current_sequence', 0) + 1
            
        return jsonify({
            "status": "move_completed",
            "next_move_order": sessions[session_id]['current_move_order'],
            "current_sequence": sessions[session_id]['current_sequence']
        })

@app.route('/get_move_status', methods=['GET'])
def get_move_status():
    with lock:
        session_id = request.args.get('session_id')
        
        if session_id not in sessions:
            return jsonify({"error": "Session not found"}), 404
            
        return jsonify({
            "current_move_order": sessions[session_id].get('current_move_order', 0),
            "current_sequence": sessions[session_id].get('current_sequence', 0),
            "move_delay": sessions[session_id].get('move_delay', 2)
        })

if __name__ == '__main__':
    app.run(host='0.0.0.0', threaded=True, port=8044, debug=True)