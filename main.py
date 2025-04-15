import time
import pydirectinput
import keyboard
import pyautogui
import cv2
import numpy as np
import pygetwindow as gw
import requests
import os
import json
import platform
import ctypes
import logging
import mss
import easyocr
import re
from logging.handlers import RotatingFileHandler
from io import BytesIO
from datetime import datetime
from collections import defaultdict

DEFAULT_CONFIG = {
    "scan_interval": 5,
    "start_key": "F6",
    "pause_key": "F7",
    "stop_key": "F8",
    "auto_start": False,
    "webhook_url": "",
    "mode": "auto",
    "money_mode": False,
    "easyocr_gpu": False,
    "wave_upgrade_delay_start": 120,
    "wave_upgrade_rounds": [1, 6],
    "button_delay": 0.2,
    "high_priority": [
        "regen",
        "boss",
        "discount",
        "atk",
        "health",
        "units"
    ],
    "low_priority": [
        "luck",
        "speed",
        "heal",
        "jade",
        "enemy"
    ],
    "debug": False
}

CONFIG_PATH = "config.json"

def load_config():
    if not os.path.exists(CONFIG_PATH):
        with open(CONFIG_PATH, "w") as f:
            json.dump(DEFAULT_CONFIG, f, indent=4)
        print(f"[INFO] Created default config at {CONFIG_PATH}")

    with open(CONFIG_PATH, "r") as f:
        config = json.load(f)

    return config

def add_png_suffix(items):
    return [f"{item}.png" if not item.endswith('.png') else item for item in items]

config = load_config()
AUTO_START = config.get("auto_start", False)
SCAN_INTERVAL = config.get("scan_interval", 5)
BUTTON_DELAY = config.get("button_delay", 0.2)
START_KEY = config.get("start_key", "f6")
PAUSE_KEY = config.get("pause_key", "f7")
STOP_KEY = config.get("stop_key", "f8")
DISCORD_WEBHOOK_URL = config.get("webhook_url", "")
DEBUG = config.get("debug", False)
HIGH_PRIORITY = add_png_suffix(config.get("high_priority", DEFAULT_CONFIG["high_priority"]))
LOW_PRIORITY = add_png_suffix(config.get("low_priority", DEFAULT_CONFIG["low_priority"]))
UPGRADE_PRIORITY = HIGH_PRIORITY + LOW_PRIORITY
MONEY_MODE = config.get("money_mode", False)
DELAY_START = config.get("wave_upgrade_delay_start", 150)
ALLOWED_ROUNDS = config.get("wave_upgrade_rounds", [1, 6])

MODE = config.get("mode", "auto")

IMAGE_FOLDER = "images"
CONFIDENCE_THRESHOLD = 0.8
UI_TOGGLE_KEY = '\\'
DEBOUNCE_TIME = 0.3
KEY_HOLD_TIME = 0.3

# State variables
last_victory_time = 0
run_start_time = 0
victory_detected = False
upgrades_purchased = defaultdict(lambda: defaultdict(int))
is_running = False
is_paused = False
last_key_press_time = defaultdict(float)
key_hold_state = defaultdict(bool)
last_upgraded_wave = None
last_known_wave = None

IMAGE_FOLDER = "images"

VICTORY_TEMPLATE = "victory.png"
DISCONNECT_TEMPLATE = "disconnected.png"
MENU_TEMPLATE = "menu.png"

RARITY_COLORS = {
    (71, 99, 189): "Common",     # Blue
    (190, 60, 238): "Epic",      # Purple
    (238, 208, 60): "Legendary", # Yellow
    (238, 60, 60): "Mythic"      # Red
}

RARITY_ORDER = ["Unknown", "Common", "Epic", "Legendary", "Mythic"]

if platform.system() == "Windows":

    # Constants
    ES_CONTINUOUS = 0x80000000
    ES_SYSTEM_REQUIRED = 0x00000001
    ES_DISPLAY_REQUIRED = 0x00000002

    def prevent_sleep():
        ctypes.windll.kernel32.SetThreadExecutionState(
            ES_CONTINUOUS | ES_SYSTEM_REQUIRED | ES_DISPLAY_REQUIRED
        )

    def allow_sleep():
        ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS)
else:
    def prevent_sleep():
        print("[WARN] Sleep prevention not implemented for this OS")

    def allow_sleep():
        pass

def setup_logging():
    # Clear previous log file on startup
    with open('afm_macro.log', 'w'):
        pass
    
    # Create logger
    logger = logging.getLogger('AFM')
    logger.setLevel(logging.DEBUG)
    
    # File handler (rotates when reaches 1MB)
    file_handler = RotatingFileHandler(
        'afm_macro.log', 
        maxBytes=1024*1024, 
        backupCount=3,
        encoding='utf-8'
    )
    file_handler.setFormatter(logging.Formatter(
        '[%(asctime)s] %(levelname)s: %(message)s',
        datefmt='%H:%M:%S'
    ))
    
    # Console handler (for real-time output)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter(
        '[%(asctime)s] %(message)s',
        datefmt='%H:%M:%S'
    ))
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    return logger

log = setup_logging()

def get_template_path(filename):
    """Helper function to get correct template path"""
    return os.path.join(IMAGE_FOLDER, filename) if IMAGE_FOLDER else filename

def focus_roblox_window():
    try:
        roblox_windows = gw.getWindowsWithTitle("Roblox")
        if roblox_windows:
            window = roblox_windows[0]
            if not window.isActive:
                window.activate()
                time.sleep(0.2)
            return window
        if DEBUG:
            log.warning("Roblox window not found")
    except Exception as e:
        if DEBUG:
            log.error(f"Window focus error: {str(e)}")
    return None

def get_roblox_window():
    """Get Roblox window dimensions without focusing, with multi-monitor support"""
    try:
        roblox_windows = gw.getWindowsWithTitle("Roblox")
        if not roblox_windows:
            if DEBUG:
                log.warning("Roblox window not found")
            return None
            
        window = roblox_windows[0]
        
        # Refresh window dimensions
        window.left = window._rect.left
        window.top = window._rect.top
        window.width = window._rect.width
        window.height = window._rect.height
        
        # Verify window is visible and has reasonable dimensions
        if window.width < 100 or window.height < 100:
            if DEBUG:
                log.warning(f"Window too small: {window.width}x{window.height}")
            return None
            
        return window
        
    except Exception as e:
        if DEBUG:
            log.error(f"Window detection error: {str(e)}")
        return None

def get_window_screenshot(window):
    """Capture ONLY the Roblox window using mss"""
    if not window:
        return None
        
    try:
        with mss.mss() as sct:
            # Get exact window coordinates including borders
            monitor = {
                "left": window.left,
                "top": window.top,
                "width": window.width,
                "height": window.height,
            }
            
            # Grab the image
            sct_img = sct.grab(monitor)
            img = np.array(sct_img)
            
            # Convert color space and return
            return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
            
    except Exception as e:
        log.error(f"Window capture failed: {str(e)}")
        return None

USE_GPU = config.get("easyocr_gpu", False)
reader = easyocr.Reader(['en'], gpu=USE_GPU)

def get_wave_region(window):
    return {
        "top": window.top + int(window.height * 0),
        "left": window.left + int(window.width * 0.32), # Not ideal since only works in 1080p
        "width": int(window.width * 0.07),
        "height": int(window.height * 0.09)
    }

def preprocess_wave_image(image):
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Resize (scaling up helps OCR)
    scale_factor = 5
    gray = cv2.resize(gray, (image.shape[1]*scale_factor, image.shape[0]*scale_factor))
    
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Adaptive thresholding to highlight digits
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 15, 7)

    return thresh

def extract_wave_number(window):
    global last_known_wave
    region = get_wave_region(window)
    with mss.mss() as sct:
        wave_img = np.array(sct.grab(region))
        processed = preprocess_wave_image(wave_img)

        results = reader.readtext(processed, detail=1)
        best_guess = None

        for _, text, conf in results:
            clean = text.replace(" ", "").upper()
            match = re.search(r'\d{2,4}', clean)
            if match:
                digits = match.group(0)
                if digits.isdigit():
                    detected_wave = int(digits)
                    # Heuristic: assume valid if it's near expected range
                    if last_known_wave is None or detected_wave >= last_known_wave:
                        best_guess = detected_wave
                        break

        if best_guess is not None:
            last_known_wave = best_guess
            return best_guess

        # If OCR failed, assume it's the next round
        if last_known_wave is not None:
            last_known_wave += 1
            if DEBUG:
                log.debug(f"Assuming wave {last_known_wave} based on last known wave")
            return last_known_wave

        log.warning("No wave detected and no fallback available.")
        return None

def resize_to_template(screenshot, template):
    h, w = screenshot.shape[:2]
    th, tw = template.shape[:2]
    scale = min(th/h, tw/w)
    return cv2.resize(screenshot, (int(w*scale), int(h*scale)))

def scale_to_window(image, window_height):
    """Scale templates relative to window size"""
    # Base template height (assuming templates made at 1080p)
    BASE_HEIGHT = 1080
    scale_factor = window_height / BASE_HEIGHT
    
    # Calculate new dimensions
    width = int(image.shape[1] * scale_factor)
    height = int(image.shape[0] * scale_factor)
    
    # Scale the image
    return cv2.resize(image, (width, height))

def match_template_in_window(screenshot, template_name):
    """Helper function for template matching that accounts for window size"""
    template_path = get_template_path(template_name)
    template = cv2.imread(template_path)
    if template is None:
        return 0.0, (0, 0)
        
    # Scale template relative to window size
    window_height = screenshot.shape[0]
    scale_factor = window_height / 1080  # Assuming 1080p base resolution
    new_width = int(template.shape[1] * scale_factor)
    new_height = int(template.shape[0] * scale_factor)
    template = cv2.resize(template, (new_width, new_height))
    
    # Perform matching
    res = cv2.matchTemplate(screenshot, template, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, max_loc = cv2.minMaxLoc(res)
    
    # Debug visualization
    if DEBUG and max_val > 0.5:
        debug_img = screenshot.copy()
        cv2.rectangle(debug_img,
                     max_loc,
                     (max_loc[0] + new_width, max_loc[1] + new_height),
                     (0, 255, 0), 2)
        cv2.putText(debug_img, f"{template_name} {max_val:.2f}",
                   (max_loc[0], max_loc[1] - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                   (0, 255, 0), 1)
    
    return max_val, max_loc

def scan_for_upgrades(max_attempts=3):
    attempts = 0
    last_results = []
    
    while attempts < max_attempts:
        try:
            window = get_roblox_window()
            if not window:
                time.sleep(1)
                attempts += 1
                continue
                
            # Get window screenshot
            screenshot = get_window_screenshot(window)
            if screenshot is None:
                attempts += 1
                continue
                
            window_height = screenshot.shape[0]
            window_width = screenshot.shape[1]
            
            # Dynamic card dimensions
            card_width = int(window_width * 0.18)
            gap = int(window_width * 0.03)
            first_card_left = (window_width // 2) - int((card_width + gap) * 1.45)
            
            # Define card regions
            regions = [
                (first_card_left, 0, card_width, window_height),
                (first_card_left + card_width + gap, 0, card_width, window_height),
                (first_card_left + 2*(card_width + gap), 0, card_width, window_height)
            ]
            
            found_upgrades = []
            
            for position, (x, y, w, h) in enumerate(regions):
                try:
                    # Crop card region
                    card_img = screenshot[y:y+h, x:x+w]
                    
                    # Check each upgrade type
                    for upgrade in UPGRADE_PRIORITY:
                        template = cv2.imread(get_template_path(upgrade))
                        if template is None:
                            continue
                            
                        # Scale template to window size
                        scaled_template = scale_to_window(template, window_height)
                        
                        # Match template
                        res = cv2.matchTemplate(card_img, scaled_template, cv2.TM_CCOEFF_NORMED)
                        _, confidence, _, _ = cv2.minMaxLoc(res)
                        
                        if confidence >= CONFIDENCE_THRESHOLD:
                            # Found a match!
                            rarity = get_rarity_from_color(card_img, position)
                            is_percent = is_percent_upgrade(card_img, position)
                            
                            found_upgrades.append({
                                'upgrade': upgrade.replace('.png', ''),  # Clean name
                                'original_upgrade': upgrade,             # Original filename
                                'position': position,
                                'rarity': rarity,
                                'confidence': confidence,
                                'is_percent': is_percent
                            })
                            break
                            
                except Exception as e:
                    log.error(f"Card scan error: {str(e)}")
            
            if found_upgrades:
                return found_upgrades
                
            attempts += 1
            time.sleep(0.5)
            
        except Exception as e:
            log.error(f"Scan error: {str(e)}")
            attempts += 1
            
    return []

def get_rarity_from_color(card_img, position):
    try:
        height, width = card_img.shape[:2]
        
        # Dynamic rarity color detection
        scan_x = int(width * 0.9)
        y_base = int(height * 0.57)
        
        # Sample width 
        sample_width = 10
        
        # Ensure we don't go out of bounds
        if scan_x + sample_width > width:
            scan_x = width - sample_width - 1
        
        # Define the scan region (5px tall, sample_width px wide)
        sample_region = card_img[
            y_base - 2 : y_base + 3,
            scan_x : scan_x + sample_width
        ]
        
        # Get average color (BGR → RGB for comparison)
        avg_color = np.mean(sample_region, axis=(0, 1)).astype(int)
        r, g, b = avg_color[2], avg_color[1], avg_color[0]  # Convert to (R, G, B)
        rgb_color = (r, g, b)
        
        if DEBUG:
            log.debug(f"Position {position} | Scanned at X={scan_x} | Color: {rgb_color}")
        
        # Find the closest rarity match
        closest_match = "Unknown"
        min_distance = float('inf')
        
        for ref_color, rarity in RARITY_COLORS.items():
            distance = sum((c1 - c2) ** 2 for c1, c2 in zip(ref_color, rgb_color)) ** 0.5
            if distance < min_distance:
                min_distance = distance
                closest_match = rarity
                if distance < 30:  # Early exit if very close match
                    break
        
        return closest_match if min_distance < 50 else "Unknown"
        
    except Exception as e:
        log.error(f"Rarity detection error: {str(e)}")
        return "Unknown"

def is_percent_upgrade(card_img, position):
    try:
        template = cv2.imread(get_template_path("percent.png"))
        if template is None:
            return False
            
        # Scale template
        scaled_template = scale_to_window(template, card_img.shape[0])
        
        # Match template
        res = cv2.matchTemplate(card_img, scaled_template, cv2.TM_CCOEFF_NORMED)
        _, confidence, _, _ = cv2.minMaxLoc(res)
        
        # Debug
        if DEBUG and confidence > 0.5:
            debug_img = card_img.copy()
            h, w = scaled_template.shape[:2]
            cv2.rectangle(debug_img,
                         (0, 0),
                         (w, h),
                         (0, 255, 0) if confidence > 0.7 else (0, 0, 255), 2)
            cv2.imwrite(f"debug/percent_{position}.png", debug_img)
            
        return confidence > 0.7
    except Exception as e:
        log.error(f"Percent check error: {str(e)}")
        return False

def select_best_upgrade(upgrades):
    if not upgrades:
        log.debug("No upgrades detected")
        return False

    # First filter out unwanted upgrades
    valid_upgrades = []
    for upgrade in upgrades:
        # Skip Common unit upgrades
        if "unit" in upgrade['upgrade'].lower() and upgrade['rarity'] == "Common":
            if DEBUG:
                log.debug(f"Skipping Common unit upgrade: {upgrade['upgrade']} at pos {upgrade['position']}")
            continue
            
        # Skip flat ATK/HP upgrades (non-percent)
        if (upgrade['original_upgrade'] in ['atk.png', 'health.png'] and 
            not upgrade.get('is_percent', False)):
            if DEBUG:
                log.debug(f"Skipping flat {upgrade['original_upgrade']} upgrade at pos {upgrade['position']}")
            continue
            
        valid_upgrades.append(upgrade)
    
    # Fail-safe if no valid upgrades
    if not valid_upgrades:
        log.debug("No valid upgrades after filtering - using fail-safe")
        for upgrade in upgrades:
            if upgrade['position'] == 0:
                log.debug(f"Fail-safe: Selecting position 0 ({upgrade['upgrade']})")
                if navigate_to(0):
                    record_upgrade_purchase(upgrade['upgrade'], upgrade['rarity'])
                    return True
        return False

    if MONEY_MODE:
        # Split into boss and non-boss upgrades
        boss_upgrades = []
        other_upgrades = []
        
        for upgrade in valid_upgrades:
            if upgrade['original_upgrade'] == 'boss.png':
                boss_upgrades.append(upgrade)
            else:
                other_upgrades.append(upgrade)
        
        # Sort boss upgrades by highest rarity first, then original priority
        boss_upgrades.sort(key=lambda x: (
            -RARITY_ORDER.index(x['rarity']),
            UPGRADE_PRIORITY.index(x['original_upgrade'])
        ))
        
        # Sort other upgrades normally
        other_upgrades.sort(key=lambda x: (
            0 if x['original_upgrade'] in HIGH_PRIORITY else 1,
            -RARITY_ORDER.index(x['rarity']),
            UPGRADE_PRIORITY.index(x['original_upgrade'])
        ))
        
        valid_upgrades = boss_upgrades + other_upgrades
    else:
        # Original sorting logic
        def get_sort_key(upgrade):
            if upgrade['original_upgrade'] in HIGH_PRIORITY:
                group = 0
            else:
                group = 1
            return (
                group,
                -RARITY_ORDER.index(upgrade['rarity']),
                UPGRADE_PRIORITY.index(upgrade['original_upgrade'])
            )
        valid_upgrades.sort(key=get_sort_key)

    if not MONEY_MODE:
        def get_sort_key(upgrade):
            # Check if the upgrade is in the high priority group
            if upgrade['original_upgrade'] in HIGH_PRIORITY:
                group = 0  # High priority
            else:
                group = 1  # Low priority

            return (
                group,
                -RARITY_ORDER.index(upgrade['rarity']),
                UPGRADE_PRIORITY.index(upgrade['original_upgrade'])
            )

        valid_upgrades.sort(key=get_sort_key)

    if DEBUG:
        log.debug("Valid upgrades sorted:")
        for idx, upgrade in enumerate(valid_upgrades, 1):
            perc = " (PERCENT)" if upgrade.get('is_percent', False) else ""
            # Determine group for debug display
            group = "HIGH" if UPGRADE_PRIORITY.index(upgrade['original_upgrade']) < 6 else "LOW"
            log.debug(f"{idx}. [{group}] {upgrade['upgrade']}{perc} ({upgrade['rarity']}) at pos {upgrade['position']}")

    # Try to select best upgrade
    for upgrade in valid_upgrades:
        perc = " (PERCENT)" if upgrade.get('is_percent', False) else ""
        if DEBUG:
            log.debug(f"Attempting: {upgrade['upgrade']}{perc} at position {upgrade['position']}")
        if navigate_to(upgrade['position']):
            record_upgrade_purchase(upgrade['upgrade'], upgrade['rarity'])
            clean_name = upgrade['upgrade'].replace('.png', '').replace("IMAGES\\","").upper()
            log.debug(f"Selected upgrade: {clean_name} ({upgrade['rarity']})")
            return True

    return False

def navigate_to(position_index):
    try:
        if DEBUG:
            log.debug(f"Attempting to navigate to position {position_index}")
        
        pydirectinput.keyDown(UI_TOGGLE_KEY)
        time.sleep(BUTTON_DELAY/2)
        pydirectinput.keyUp(UI_TOGGLE_KEY)
        time.sleep(BUTTON_DELAY)
        
        pydirectinput.keyDown('left')
        time.sleep(BUTTON_DELAY/2)
        pydirectinput.keyUp('left')
        time.sleep(BUTTON_DELAY)
        
        pydirectinput.keyDown('down')
        time.sleep(BUTTON_DELAY/2)
        pydirectinput.keyUp('down')
        time.sleep(BUTTON_DELAY)
        
        for _ in range(position_index):
            pydirectinput.keyDown('right')
            time.sleep(BUTTON_DELAY/2)
            pydirectinput.keyUp('right')
            time.sleep(BUTTON_DELAY)
        
        pydirectinput.keyDown('enter')
        time.sleep(BUTTON_DELAY/2)
        pydirectinput.keyUp('enter')
        time.sleep(BUTTON_DELAY)
        
        pydirectinput.keyDown(UI_TOGGLE_KEY)
        time.sleep(BUTTON_DELAY/2)
        pydirectinput.keyUp(UI_TOGGLE_KEY)
        time.sleep(BUTTON_DELAY)
        
        return True
        
    except Exception as e:
        log.error(f"Navigation error: {str(e)}")
        return False

def record_upgrade_purchase(upgrade_name, rarity):
    global upgrades_purchased
    upgrades_purchased[upgrade_name.replace('.png', '')][rarity] += 1
    if DEBUG:
        log.debug(f"Recorded purchase: {upgrade_name} ({rarity})")

def detect_victory():
    global last_victory_time, victory_detected, run_start_time
    
    try:
        current_time = time.time()
        if current_time - last_victory_time < 30:
            return False
            
        window = get_roblox_window()
        if not window:
            return False
        
        # Get screenshot from correct monitor
        screenshot = get_window_screenshot(window)
        
        template = cv2.imread(get_template_path(VICTORY_TEMPLATE))
        if template is None:
            return False
        
        resized_screen = resize_to_template(screenshot, template)
        res = cv2.matchTemplate(resized_screen, template, cv2.TM_CCOEFF_NORMED)
            
        #res = cv2.matchTemplate(screenshot, template, cv2.TM_CCOEFF_NORMED)
        _, confidence, _, _ = cv2.minMaxLoc(res)
        
        time.sleep(2) # Wait for results to load
        
        screenshot = get_window_screenshot(window)
        
        if confidence > CONFIDENCE_THRESHOLD:
            run_time = 0
            if run_start_time > 0:
                run_time = current_time - run_start_time
                run_start_time = 0
            
            if DEBUG:
                log.debug(f"Victory detected! (confidence: {confidence:.2f})")
            if DISCORD_WEBHOOK_URL:
                log.debug("Uploading screenshot and stats to Discord...")
                if not upload_to_discord(screenshot, run_time):
                    log.error("Failed to upload to Discord")
            else:
                if DEBUG:
                    log.debug("Discord webhook URL not set, skipping upload")
            
            last_victory_time = current_time
            victory_detected = True
            return True
            
        return False
        
    except Exception as e:
        log.error(f"Victory detection error: {str(e)}")
        return False

def generate_upgrade_summary():
    """Generate a formatted summary of upgrades purchased this run with custom header and static column widths."""
    if not upgrades_purchased:
        return "No upgrades purchased this run"

    # Custom header line with fixed width
    header_line = "Upgrade         |        🔴        |        🟡         |        🟣        | 🔵 "

    RARITY_EMOJIS = {
        "Mythic": "🔴",
        "Legendary": "🟡",
        "Epic": "🟣",
        "Common": "🔵"
    }

    rarities_order = ["Mythic", "Legendary", "Epic", "Common"]
    
    # Prepare the rows for upgrades
    rows = []
    for upgrade in sorted(upgrades_purchased.keys()):
        row = [upgrade.capitalize()]
        for rarity in rarities_order:
            row.append(str(upgrades_purchased[upgrade].get(rarity, 0)))
        rows.append(row)

    # Calculate the totals
    totals = ["TOTAL"]
    for i in range(len(rarities_order)):
        totals.append(str(sum(int(row[i+1]) for row in rows)))
    rows.append(totals)

    # Define the fixed column widths (manual width setting)
    col_widths = [15, 2, 2, 2, 2]

    # Function to format a row with the fixed column widths
    def format_row(row):
        return " | ".join(f"{str(cell):<{col_widths[i]}}" for i, cell in enumerate(row))

    # Output the table
    output = ["```"]
    output.append(header_line)  # Use the manually set header line
    output.append("----------------+----+----+----+---") # Separator line
    for row in rows:
        output.append(format_row(row))  # Format and add the data rows
    output.append("```")

    return "\n".join(output)

def upload_to_discord(screenshot, run_time):
    try:
        _, img_encoded = cv2.imencode('.png', screenshot)
        img_bytes = BytesIO(img_encoded.tobytes())
        
        minutes, seconds = divmod(run_time, 60)
        time_str = f"{int(minutes):02d}:{int(seconds):02d}"
        
        summary = generate_upgrade_summary()
        
        files = {
            'file': ('victory.png', img_bytes, 'image/png')
        }
        
        payload = {
            "content": f"Run completed in {time_str}\n{summary}",
            "username": "AFM"
        }
        
        response = requests.post(
            DISCORD_WEBHOOK_URL,
            data=payload,
            files=files
        )
        
        if response.status_code == 204:
            if DEBUG:
                log.debug("Screenshot and stats uploaded to Discord successfully")
            return True
        else:
            log.error(f"Discord upload failed: {response.text}")
            return False
            
    except Exception as e:
        log.error(f"Discord upload error: {str(e)}")
        return False

def restart_run():
    global run_start_time, victory_detected, upgrades_purchased
    
    log.debug("Attempting to restart run")
    try:
        time.sleep(BUTTON_DELAY)
        pydirectinput.keyDown(UI_TOGGLE_KEY)
        time.sleep(BUTTON_DELAY/2)
        pydirectinput.keyUp(UI_TOGGLE_KEY)
        time.sleep(BUTTON_DELAY)
        
        for _ in range(3):
            pydirectinput.keyDown('down')
            time.sleep(BUTTON_DELAY/2)
            pydirectinput.keyUp('down')
            time.sleep(BUTTON_DELAY)
            
        pydirectinput.keyDown('enter')
        time.sleep(BUTTON_DELAY/2)
        pydirectinput.keyUp('enter')
        time.sleep(BUTTON_DELAY)
        
        pydirectinput.keyDown(UI_TOGGLE_KEY)
        time.sleep(BUTTON_DELAY/2)
        pydirectinput.keyUp(UI_TOGGLE_KEY)
        time.sleep(BUTTON_DELAY)
        
        run_start_time = time.time()
        victory_detected = False
        upgrades_purchased = defaultdict(lambda: defaultdict(int))
        return True
        
    except Exception as e:
        log.error(f"Run restart error: {str(e)}")
        return False

def is_key_pressed(key, check_hold=False):
    """Improved key press detection with debouncing and hold detection"""
    global last_key_press_time, key_hold_state
    
    current_time = time.time()
    
    # If key isn't physically pressed, reset its state
    if not keyboard.is_pressed(key):
        key_hold_state[key] = False
        return False
    
    # Check if key is being held down
    if check_hold and key_hold_state[key]:
        if current_time - last_key_press_time[key] > KEY_HOLD_TIME:
            return True
        return False
    
    # Standard debounce check
    if current_time - last_key_press_time[key] < DEBOUNCE_TIME:
        return False
    
    # Valid key press detected
    last_key_press_time[key] = current_time
    key_hold_state[key] = True
    return True

def manual_mode_loop():
    global is_running, is_paused
    log.info("=== Manual Mode ===")
    log.info(f"Press {START_KEY} to scan/select | {STOP_KEY} to exit")
    
    while True:
        if is_key_pressed(STOP_KEY):
            log.info("=== Stopped ===")
            is_running = False
            allow_sleep()
            break
            
        if is_key_pressed(PAUSE_KEY):  # Pause not used in manual mode
            time.sleep(0.5)  # Debounce
            
        if is_key_pressed(START_KEY):
            log.debug("Manual scan triggered")
            # Perform single scan/select cycle
            window = get_roblox_window()
            if window:
                upgrades = scan_for_upgrades()
                if upgrades:
                    if focus_roblox_window():
                        select_best_upgrade(upgrades)
                else:
                    log.debug("No upgrades found")
            # Wait for key release
            while is_key_pressed(START_KEY):
                time.sleep(0.1)
            time.sleep(0.2)  # Debounce
            
        time.sleep(0.05)

def main_loop():
    global run_start_time, victory_detected, is_running, is_paused, last_upgraded_wave, last_known_wave
    
    log.info("=== AFK Endless Macro ===")
    log.info(f"Mode: {MODE.capitalize()} Scan | Press {START_KEY} to begin." if not AUTO_START else "Auto-start: Enabled.")
    log.info(f"Press {PAUSE_KEY} to pause/resume | {STOP_KEY} to stop")
    log.info(f"Money Mode: {'ENABLED' if MONEY_MODE else 'disabled'}")
    log.info(f"High priority upgrades: {', '.join([upgrade.replace('.png', '') for upgrade in HIGH_PRIORITY])}")
    log.info(f"Low priority upgrades: {', '.join([upgrade.replace('.png', '') for upgrade in LOW_PRIORITY])}")
    
    last_scan = time.time()
    last_victory_check = time.time()
    run_start_time = time.time()
    victory_detected = False
    
    if MODE == "manual":
        manual_mode_loop()
        return
    
    # Auto-start if configured
    if AUTO_START:
        is_running = True
    
    while True:
        # Check for start/pause/stop keys
        if is_key_pressed(STOP_KEY):
            log.info("=== Stopped ===")
            is_running = False
            allow_sleep()
            break
            
        if is_key_pressed(START_KEY):
            if not is_running:
                log.info("=== Started ===")
                is_running = True
                run_start_time = time.time()  # Reset timer on manual start
            time.sleep(0.5)  # Debounce
            
        if is_key_pressed(PAUSE_KEY):
            is_paused = not is_paused
            log.info(f"=== {'Paused' if is_paused else 'Resumed'} ===")
            time.sleep(0.5)  # Debounce
        
        # Only run logic when active and not paused
        if is_running and not is_paused:
            prevent_sleep()   
            current_time = time.time()
            
            # Victory check
            if current_time - last_victory_check > SCAN_INTERVAL:
                if detect_victory():
                    if victory_detected:
                        restart_run()
                last_victory_check = current_time
                
            # Upgrade scanning
            if current_time - last_scan > SCAN_INTERVAL:
                window = get_roblox_window()
                if window:
                    upgrades = scan_for_upgrades()
                    if upgrades:
                        current_wave = extract_wave_number(window)
                        if current_wave:
                            if DEBUG:
                                log.info(f"Current wave: {current_wave}")

                            # Check upgrade condition based on wave
                            if current_wave < DELAY_START or (current_wave % 10) in ALLOWED_ROUNDS:
                                if last_upgraded_wave != current_wave:
                                    if focus_roblox_window():
                                        if select_best_upgrade(upgrades):
                                            last_upgraded_wave = current_wave  # Record this wave
                                            log.info(f"Upgrade executed on wave {current_wave}")
                                        else:
                                            log.info("No upgrade selected.")
                                        last_scan = time.time()
                                    else:
                                        last_scan = time.time() + 2
                                else:
                                    if DEBUG:
                                        log.info(f"Already upgraded on wave {current_wave}. Skipping.")
                                    last_scan = time.time()
                            else:
                                if DEBUG:
                                    log.info(f"Upgrade delay active for wave {current_wave}")
                                last_scan = time.time()
                        else:
                            log.debug("Could not read wave number — skipping upgrade.")
                            last_scan = time.time()
                    else:
                        if DEBUG:
                            log.info("No upgrades found.")
                        last_scan = time.time()
                else:
                    last_scan = time.time() + 1
        else:
            allow_sleep()  # Allow system sleep when paused/stopped
            time.sleep(0.1)  # Reduce CPU usage

if __name__ == "__main__":
    main_loop()
