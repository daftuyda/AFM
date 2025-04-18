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
import threading
import tempfile
import subprocess
import sys
from logging.handlers import RotatingFileHandler
from io import BytesIO
from datetime import datetime
from collections import defaultdict

__version__ = "1.3.1"

DEFAULT_CONFIG = {
    "ultrawide_mode": False,
    "maximize_window": True,
    "auto_reconnect": True,
    "scan_interval": 5,
    "start_key": "F6",
    "pause_key": "F7",
    "stop_key": "F8",
    "auto_start": False,
    "webhook_url": "",
    "mode": "auto",
    "money_mode": False,
    "button_delay": 0.2,
    "high_priority": ["regen","boss","discount","atk","health","units"],
    "low_priority": ["luck","speed","heal","jade","enemy"],
    "auto_updates": False,
    "debug": False
}

CONFIG_PATH = "config.json"

def prompt_config():
    config = DEFAULT_CONFIG.copy()
    print("\n=== Configuration Setup ===")
    print("Configure settings below (press Enter to use default):\n")
    
    response = input(f"Ultrawide mode? [y/N]: ").strip().lower()
    config["ultrawide_mode"] = response == "y"
    
    response = input(f"Maximize window? [Y/n]: ").strip().lower()
    config["maximize_window"] = False if response == "n" else True
    
    response = input(f"Enable auto-reconnect? [Y/n]: ").strip().lower()
    config["auto_reconnect"] = False if response == "n" else True
    
    response = input(f"Scan interval (default: {DEFAULT_CONFIG['scan_interval']}): ").strip()
    if response:
        try:
            config["scan_interval"] = int(response)
        except ValueError:
            print("Invalid input. Using default.")
    
    response = input(f"Start key (default: {DEFAULT_CONFIG['start_key']}): ").strip().upper()
    if response:
        config["start_key"] = response
    
    response = input(f"Pause key (default: {DEFAULT_CONFIG['pause_key']}): ").strip().upper()
    if response:
        config["pause_key"] = response
    
    response = input(f"Stop key (default: {DEFAULT_CONFIG['stop_key']}): ").strip().upper()
    if response:
        config["stop_key"] = response
    
    response = input(f"Auto-start? [y/N]: ").strip().lower()
    config["auto_start"] = response == "y"
    
    response = input(f"Discord webhook URL (optional): ").strip()
    if response:
        config["webhook_url"] = response
    
    response = input(f"Mode [auto]/manual: ").strip().lower()
    config["mode"] = response if response in ["auto", "manual"] else "auto"
    
    response = input(f"Money mode? [y/N]: ").strip().lower()
    config["money_mode"] = response == "y"
    
    response = input(f"Button delay (default: {DEFAULT_CONFIG['button_delay']}): ").strip()
    if response:
        try:
            config["button_delay"] = float(response)
        except ValueError:
            print("Invalid input. Using default.")
    
    response = input(f"Debug mode? [y/N]: ").strip().lower()
    config["debug"] = response == "y"
    
    response = input(f"Automatically update? [y/N]: ").strip().lower()
    config["auto_updates"] = response == "y"
    
    print("\n=== Configuration Complete ===")
    print("High/low priority upgrades use defaults. Edit the code to modify them.\n")
    return config

if os.path.exists(CONFIG_PATH):
    with open(CONFIG_PATH, 'r') as f:
        config = json.load(f)
else:
    config = prompt_config()
    with open(CONFIG_PATH, 'w') as f:
        json.dump(config, f, indent=4)

def add_png_suffix(items):
    return [f"{item}.png" if not item.endswith('.png') else item for item in items]

ULTRAWIDE_MODE = config.get("ultrawide_mode", False)
MAXIMIZE_WINDOW = config.get("maximize_window", True)
AUTO_RECONNECT = config.get("auto_reconnect", True)
AUTO_START = config.get("auto_start", False)
SCAN_INTERVAL = config.get("scan_interval", 5)
BUTTON_DELAY = config.get("button_delay", 0.2)
START_KEY = config.get("start_key", "f6").lower()
PAUSE_KEY = config.get("pause_key", "f7").lower()
STOP_KEY = config.get("stop_key", "f8").lower()
DISCORD_WEBHOOK_URL = config.get("webhook_url", "")
MONEY_MODE = config.get("money_mode", False)
MODE = config.get("mode", "auto")
HIGH_PRIORITY = add_png_suffix(config.get("high_priority", DEFAULT_CONFIG["high_priority"]))
LOW_PRIORITY = add_png_suffix(config.get("low_priority", DEFAULT_CONFIG["low_priority"]))
UPGRADE_PRIORITY = HIGH_PRIORITY + LOW_PRIORITY
DEBUG = config.get("debug", False)
AUTO_UPDATES = config.get("auto_updates", True)

IMAGE_FOLDER = "ultrawide" if ULTRAWIDE_MODE else "images"
CONFIDENCE_THRESHOLD = 0.8
UI_TOGGLE_KEY = '\\'
DEBOUNCE_TIME = 0.3
KEY_HOLD_TIME = 0.1

# State variables
last_victory_time = 0
run_start_time = 0
last_disconnect_check = 0
victory_detected = False
upgrades_purchased = defaultdict(lambda: defaultdict(int))
is_running = False
is_paused = False
last_key_press_time = defaultdict(float)
key_hold_state = defaultdict(bool)

RARITY_COLORS = {(71, 99, 189): "Common", (190, 60, 238): "Epic", (238, 208, 60): "Legendary", (238, 60, 60): "Mythic"}

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

class UpgradeScannerThread(threading.Thread):
    def __init__(self):
        super().__init__(daemon=True)
        self.upgrades = None
        self.running = False
        self.lock = threading.Lock()

    def run(self):
        self.running = True
        while self.running:
            if is_running and not is_paused:
                upgrades = scan_for_upgrades()
                with self.lock:
                    self.upgrades = upgrades
            time.sleep(SCAN_INTERVAL)

    def get_upgrades(self):
        with self.lock:
            upgrades = self.upgrades
            self.upgrades = None
        return upgrades

    def stop(self):
        self.running = False

class VictoryCheckerThread(threading.Thread):
    def __init__(self):
        super().__init__(daemon=True)
        self.victory = False
        self.running = False
        self.lock = threading.Lock()

    def run(self):
        self.running = True
        while self.running:
            if is_running and not is_paused:
                if detect_victory():
                    with self.lock:
                        self.victory = True
            time.sleep(SCAN_INTERVAL)

    def get_victory(self):
        with self.lock:
            v = self.victory
            self.victory = False
        return v

    def stop(self):
        self.running = False

class DisconnectionWatcherThread(threading.Thread):
    def __init__(self):
        super().__init__(daemon=True)
        self.running = False
        self.rejoining = False

    def run(self):
        self.running = True
        while self.running:
            try:
                time.sleep(300)

                if not is_running or is_paused or self.rejoining:
                    continue

                if detect_disconnection_button():
                    log.warning("Disconnection detected! Attempting to rejoin.")
                    self.rejoining = True
                    try:
                        if rejoin_game():
                            log.info("Rejoin successful")
                        else:
                            log.error("Rejoin failed")
                    finally:
                        self.rejoining = False
            except Exception as e:
                log.error(f"Disconnection watcher error: {str(e)}")
                self.rejoining = False
                
    def stop(self):
        self.running = False

def setup_logging():
    with open('afm_macro.log', 'w'):
        pass
    
    logger = logging.getLogger('AFM')
    logger.setLevel(logging.DEBUG)
    
    file_handler = RotatingFileHandler('afm_macro.log', maxBytes=2048*2048, backupCount=3, encoding='utf-8')
    file_handler.setFormatter(logging.Formatter('[%(asctime)s] %(levelname)s: %(message)s', datefmt='%H:%M:%S'))
    
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter('[%(asctime)s] %(message)s', datefmt='%H:%M:%S'))
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    return logger

log = setup_logging()

def get_template_path(filename):
    return os.path.join(IMAGE_FOLDER, filename) if IMAGE_FOLDER else filename

def focus_roblox_window():
    try:
        roblox_windows = gw.getWindowsWithTitle("Roblox")
        if roblox_windows:
            window = roblox_windows[0]
            if not window.isActive:
                window.activate()
                time.sleep(0.2)
            if MAXIMIZE_WINDOW and not window.isMaximized:
                window.maximize()
                time.sleep(1)  # Allow time for window to maximize
            return window
        if DEBUG:
            log.warning("Roblox window not found")
    except Exception as e:
        if DEBUG:
            log.error(f"Window focus error: {str(e)}")
    return None

def get_roblox_window():
    try:
        roblox_windows = gw.getWindowsWithTitle("Roblox")
        if not roblox_windows:
            if DEBUG:
                log.warning("Roblox window not found")
            return None
            
        window = roblox_windows[0]
        
        window.left = window._rect.left
        window.top = window._rect.top
        window.width = window._rect.width
        window.height = window._rect.height
        
        if window.width < 100 or window.height < 100: # Verify window is visible and has reasonable dimensions
            if DEBUG:
                log.warning(f"Window too small: {window.width}x{window.height}")
            return None
            
        return window
        
    except Exception as e:
        if DEBUG:
            log.error(f"Window detection error: {str(e)}")
        return None

def get_window_screenshot(window):
    if not window:
        return None
        
    try:
        with mss.mss() as sct:
            monitor = { "left": window.left, "top": window.top, "width": window.width, "height": window.height}
            
            sct_img = sct.grab(monitor)
            img = np.array(sct_img)
            
            return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
            
    except Exception as e:
        log.error(f"Window capture failed: {str(e)}")
        return None

def resize_to_template(screenshot, template):
    h, w = screenshot.shape[:2]
    th, tw = template.shape[:2]
    scale = min(th/h, tw/w)
    return cv2.resize(screenshot, (int(w*scale), int(h*scale)))

def scale_to_window(image, window_height):
    BASE_HEIGHT = 1080
    scale_factor = window_height / BASE_HEIGHT
    
    width = int(image.shape[1] * scale_factor)
    height = int(image.shape[0] * scale_factor)
    
    return cv2.resize(image, (width, height))

def match_template_in_window(screenshot, template_name):
    template_path = get_template_path(template_name)
    template = cv2.imread(template_path)
    if template is None:
        return 0.0, (0, 0)
        
    window_height = screenshot.shape[0]
    scale_factor = window_height / 1080
    new_width = int(template.shape[1] * scale_factor)
    new_height = int(template.shape[0] * scale_factor)
    template = cv2.resize(template, (new_width, new_height))

    res = cv2.matchTemplate(screenshot, template, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, max_loc = cv2.minMaxLoc(res)
    
    return max_val, max_loc

def scan_for_upgrades(max_attempts=3):
    attempts = 0
    
    while attempts < max_attempts:
        try:
            window = get_roblox_window()
            if not window:
                time.sleep(1)
                attempts += 1
                continue
                
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
                    card_img = screenshot[y:y+h, x:x+w]
                    
                    for upgrade in UPGRADE_PRIORITY:
                        template = cv2.imread(get_template_path(upgrade))
                        if template is None:
                            continue
                            
                        scaled_template = scale_to_window(template, window_height)
                        
                        res = cv2.matchTemplate(card_img, scaled_template, cv2.TM_CCOEFF_NORMED)
                        _, confidence, _, _ = cv2.minMaxLoc(res)
                        
                        if confidence >= CONFIDENCE_THRESHOLD:
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
        y_base = int(height * 0.56)
        
        # Sample width 
        sample_width = 10
        
        if scan_x + sample_width > width:
            scan_x = width - sample_width - 1
        
        sample_region = card_img[ y_base - 2 : y_base + 3, scan_x : scan_x + sample_width]
        
        # Get average color (BGR → RGB for comparison)
        avg_color = np.mean(sample_region, axis=(0, 1)).astype(int)
        r, g, b = avg_color[2], avg_color[1], avg_color[0]  # Convert to (R, G, B)
        rgb_color = (r, g, b)
        
        if DEBUG:
            log.debug(f"Position {position} | Scanned at X={scan_x} | Color: {rgb_color}")
        
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
        # Check first percent template
        template1_path = get_template_path("percent.png")
        if os.path.exists(template1_path):
            template1 = cv2.imread(template1_path)
            if template1 is not None:
                scaled_template1 = scale_to_window(template1, card_img.shape[0])
                res1 = cv2.matchTemplate(card_img, scaled_template1, cv2.TM_CCOEFF_NORMED)
                _, confidence1, _, _ = cv2.minMaxLoc(res1)
                if confidence1 > 0.7:
                    if DEBUG:
                        log.debug(f"Percent detected (template1) at position {position} with confidence {confidence1:.2f}")
                    return True

        # Check second percent template only if file exists
        template2_path = get_template_path("percent2.png")
        if os.path.exists(template2_path):
            template2 = cv2.imread(template2_path)
            if template2 is not None:
                scaled_template2 = scale_to_window(template2, card_img.shape[0])
                res2 = cv2.matchTemplate(card_img, scaled_template2, cv2.TM_CCOEFF_NORMED)
                _, confidence2, _, _ = cv2.minMaxLoc(res2)
                if confidence2 > 0.7:
                    if DEBUG:
                        log.debug(f"Percent detected (template2) at position {position} with confidence {confidence2:.2f}")
                    return True
        elif DEBUG:
            log.debug("percent2.png not found, skipping check")

        if DEBUG:
            conf1 = confidence1 if 'confidence1' in locals() else 'N/A'
            conf2 = confidence2 if 'confidence2' in locals() else 'N/A'
            log.debug(f"Percent check results - Template1: {conf1}, Template2: {conf2}")

        return False

    except Exception as e:
        log.error(f"Percent check error: {str(e)}")
        return False

def select_best_upgrade(upgrades):
    if not upgrades:
        log.debug("No upgrades detected")
        return False

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
        if len(valid_upgrades) > 1:
            valid_upgrades.sort(key=get_sort_key)

    if not MONEY_MODE:
        def get_sort_key(upgrade):
            if upgrade['original_upgrade'] in HIGH_PRIORITY:
                group = 0  # High priority
            else:
                group = 1  # Low priority

            return (
                group,
                -RARITY_ORDER.index(upgrade['rarity']),
                UPGRADE_PRIORITY.index(upgrade['original_upgrade'])
            )

        if len(valid_upgrades) > 1:
            valid_upgrades.sort(key=get_sort_key)

    if DEBUG:
        log.debug("Valid upgrades sorted:")
        for idx, upgrade in enumerate(valid_upgrades, 1):
            perc = " (PERCENT)" if upgrade.get('is_percent', False) else ""
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

def toggle_ui_and_confirm(window=None, max_attempts=3):
    if not window:
        window = get_roblox_window()
        if not window:
            log.warning("No Roblox window found for UI confirmation.")
            return False

    template_path = get_template_path("navbox.png")
    if not os.path.exists(template_path):
        log.warning("navbox.png not found — skipping UI toggle check.")
        return True

    template = cv2.imread(template_path)
    if template is None:
        log.error("Failed to load navbox.png")
        return False

    for attempt in range(max_attempts):
        pydirectinput.keyDown(UI_TOGGLE_KEY) # Press the UI toggle key
        time.sleep(BUTTON_DELAY / 2)
        pydirectinput.keyUp(UI_TOGGLE_KEY)
        time.sleep(1.2 if attempt == 0 else 1) # Wait longer on first attempt

        screenshot = get_window_screenshot(window)
        if screenshot is None:
            continue

        scaled_template = scale_to_window(template, screenshot.shape[0])
        res = cv2.matchTemplate(screenshot, scaled_template, cv2.TM_CCOEFF_NORMED)
        _, confidence, _, _ = cv2.minMaxLoc(res)

        if DEBUG:
            log.debug(f"UI box detection attempt {attempt + 1}: confidence={confidence:.2f}")

        if confidence >= 0.6:
            return True

    log.warning("UI nav box not detected after toggling UI key.")
    return False

def navigate_to(position_index):
    try:
        if DEBUG:
            log.debug(f"Attempting to navigate to position {position_index}")
        
        toggle_ui_and_confirm()
        
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
        
        screenshot = get_window_screenshot(window)
        
        template = cv2.imread(get_template_path("victory.png"))
        if template is None:
            return False
        
        template = scale_to_window(template, screenshot.shape[0])  # Use same scaling as upgrades
        res = cv2.matchTemplate(screenshot, template, cv2.TM_CCOEFF_NORMED)
        
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
    
    rows = []
    for upgrade in sorted(upgrades_purchased.keys()):
        row = [upgrade.capitalize()]
        for rarity in rarities_order:
            row.append(str(upgrades_purchased[upgrade].get(rarity, 0)))
        rows.append(row)

    totals = ["TOTAL"]
    for i in range(len(rarities_order)):
        totals.append(str(sum(int(row[i+1]) for row in rows)))
    rows.append(totals)

    col_widths = [15, 2, 2, 2, 2]

    def format_row(row):
        return " | ".join(f"{str(cell):<{col_widths[i]}}" for i, cell in enumerate(row))

    output = ["```"]
    output.append(header_line)  # Use the manually set header line
    output.append("----------------+----+----+----+---") # Separator line
    for row in rows:
        output.append(format_row(row))  # Format and add the data rows
    output.append("```")

    return "\n".join(output)

def upload_to_discord(screenshot, run_time):
    try:
        _, img_encoded = cv2.imencode('.jpg', screenshot, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        img_bytes = BytesIO(img_encoded.tobytes())
        
        minutes, seconds = divmod(run_time, 60)
        time_str = f"{int(minutes):02d}:{int(seconds):02d}"
        
        summary = generate_upgrade_summary()
        
        files = {'file': ('victory.png', img_bytes, 'image/png')}
        
        payload = {"content": f"Run completed in {time_str}\n{summary}", "username": "AFM"}
        
        response = requests.post(DISCORD_WEBHOOK_URL, data=payload, files=files)
        
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

def template_match(screenshot, template_path, confidence=0.7):
    try:
        template = cv2.imread(template_path)
        if template is None:
            log.warning(f"Template not found at {template_path}")
            return False
        
        template = scale_to_window(template, screenshot.shape[0])
        result = cv2.matchTemplate(screenshot, template, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, _ = cv2.minMaxLoc(result)
        
        if DEBUG:
            log.debug(f"Template match confidence for {os.path.basename(template_path)}: {max_val:.2f}")
            
        return max_val > confidence
    except Exception as e:
        log.error(f"Template match error: {str(e)}")
        return False

def teleport_to_endless():
    try:
        if DEBUG:
            log.debug(f"Teleporting to Endless Area")
        
        pydirectinput.keyDown(UI_TOGGLE_KEY)
        time.sleep(BUTTON_DELAY/2)
        pydirectinput.keyUp(UI_TOGGLE_KEY)
        time.sleep(BUTTON_DELAY)
        
        pydirectinput.keyDown('down')
        time.sleep(BUTTON_DELAY/2)
        pydirectinput.keyUp('down')
        time.sleep(BUTTON_DELAY)
        
        pydirectinput.keyDown('enter')
        time.sleep(BUTTON_DELAY/2)
        pydirectinput.keyUp('enter')
        time.sleep(BUTTON_DELAY)
        
        for _ in range(2):
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
        
        return True
        
    except Exception as e:
        log.error(f"Navigation error: {str(e)}")
        return False

def move_to_endless():
    try:
        if DEBUG:
            log.debug("Moving to Endless Area")
            
        time.sleep(0.5)
        pydirectinput.keyDown('q')
        time.sleep(BUTTON_DELAY/2)
        pydirectinput.keyUp('q')
        time.sleep(1)
        
        return True
        
    except Exception as e:
        log.error(f"Move to Endless error: {str(e)}")
        return False
    
def toggle_troops():
    try:
        if DEBUG:
            log.debug("Attempting to enable Auto Troops")
            
        # Wait for auto.png to appear (up to 10 seconds)
        start_time = time.time()
        auto_detected = False
        while time.time() - start_time < 10:
            window = get_roblox_window()
            if window:
                screenshot = get_window_screenshot(window)
                if screenshot is not None:
                    # Check for auto.png with higher confidence threshold
                    if template_match(screenshot, get_template_path("auto.png"), confidence=0.8):
                        auto_detected = True
                        if DEBUG:
                            log.debug("Auto troops button detected")
                        break
                    elif DEBUG:
                        log.debug("Auto troops button not yet visible")
            time.sleep(1)  # Check every second
            
                # First open the menu
        time.sleep(1)
        pydirectinput.keyDown(UI_TOGGLE_KEY)
        time.sleep(BUTTON_DELAY/2)
        pydirectinput.keyUp(UI_TOGGLE_KEY)
        time.sleep(BUTTON_DELAY)

        # Navigate to auto troops option
        for _ in range(7):
            pydirectinput.keyDown('down')
            time.sleep(BUTTON_DELAY/2)
            pydirectinput.keyUp('down')
            time.sleep(BUTTON_DELAY)

        if not auto_detected:
            log.error("Auto troops button not found after 10 seconds")
            # Close menu
            pydirectinput.keyDown(UI_TOGGLE_KEY)
            time.sleep(BUTTON_DELAY/2)
            pydirectinput.keyUp(UI_TOGGLE_KEY)
            time.sleep(BUTTON_DELAY)
            return False

        # Press enter to enable auto troops
        pydirectinput.keyDown('enter')
        time.sleep(BUTTON_DELAY/2)
        pydirectinput.keyUp('enter')
        time.sleep(BUTTON_DELAY)

        # Close the menu
        pydirectinput.keyDown(UI_TOGGLE_KEY)
        time.sleep(BUTTON_DELAY/2)
        pydirectinput.keyUp(UI_TOGGLE_KEY)
        time.sleep(BUTTON_DELAY)
        
        log.info("Auto troops successfully enabled")
        return True
        
    except Exception as e:
        log.error(f"Auto Troops Error: {str(e)}")
        # Ensure menu is closed if error occurs
        pydirectinput.keyDown(UI_TOGGLE_KEY)
        time.sleep(BUTTON_DELAY/2)
        pydirectinput.keyUp(UI_TOGGLE_KEY)
        time.sleep(BUTTON_DELAY)
        return False

def detect_disconnection_button():
    try:
        window = get_roblox_window()
        if not window:
            return False

        screenshot = get_window_screenshot(window)
        if screenshot is None:
            return False

        template_path = get_template_path("reconnect.png")
        template = cv2.imread(template_path)
        if template is None:
            return False

        template = scale_to_window(template, screenshot.shape[0])
        res = cv2.matchTemplate(screenshot, template, cv2.TM_CCOEFF_NORMED)
        _, confidence, _, _ = cv2.minMaxLoc(res)

        if DEBUG:
            log.debug(f"Disconnection button match confidence: {confidence:.2f}")

        return confidence >= 0.7

    except Exception as e:
        log.error(f"Disconnection detection error: {str(e)}")
        return False

def click_on_template(template_name, confidence=0.7):
    try:
        window = get_roblox_window()
        if not window:
            log.warning("Roblox window not found for clicking.")
            return False

        screenshot = get_window_screenshot(window)
        if screenshot is None:
            return False

        template_path = get_template_path(template_name)
        template = cv2.imread(template_path)
        if template is None:
            log.warning(f"Template '{template_name}' not found.")
            return False

        template = scale_to_window(template, screenshot.shape[0])
        res = cv2.matchTemplate(screenshot, template, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(res)

        if DEBUG:
            log.debug(f"'{template_name}' match confidence: {max_val:.2f}")

        if max_val >= confidence:
            template_h, template_w = template.shape[:2]
            
            # Get virtual desktop dimensions
            virtual_width = ctypes.windll.user32.GetSystemMetrics(78)  # SM_CXVIRTUALSCREEN
            virtual_height = ctypes.windll.user32.GetSystemMetrics(79) # SM_CYVIRTUALSCREEN
            virtual_left = ctypes.windll.user32.GetSystemMetrics(76)   # SM_XVIRTUALSCREEN
            virtual_top = ctypes.windll.user32.GetSystemMetrics(77)    # SM_YVIRTUALSCREEN

            # Calculate absolute coordinates within virtual desktop
            target_x = window.left + max_loc[0] + template_w // 2
            target_y = window.top + max_loc[1] + template_h // 2

            # Convert to absolute coordinates (0-65535)
            abs_x = int((target_x - virtual_left) / virtual_width * 65535)
            abs_y = int((target_y - virtual_top) / virtual_height * 65535)

            # Get current mouse position
            current_pos = pyautogui.position()
            current_abs_x = int((current_pos.x - virtual_left) / virtual_width * 65535)
            current_abs_y = int((current_pos.y - virtual_top) / virtual_height * 65535)

            # Move mouse in small steps to the target
            steps = 5
            for i in range(1, steps + 1):
                intermediate_x = current_abs_x + (abs_x - current_abs_x) * i / steps
                intermediate_y = current_abs_y + (abs_y - current_abs_y) * i / steps
                ctypes.windll.user32.mouse_event(0x8000 | 0x0001 | 0x4000,  # MOUSEEVENTF_ABSOLUTE|MOUSEEVENTF_MOVE|MOUSEEVENTF_VIRTUALDESK
                                                int(intermediate_x), 
                                                int(intermediate_y), 
                                                0, 0)
                time.sleep(0.05)  # Small delay between movements

            # Perform click sequence with slight movement
            for _ in range(2):  # Try twice for reliability
                # Small movement before click
                ctypes.windll.user32.mouse_event(0x8000 | 0x0001 | 0x4000, 
                                                abs_x - 5, 
                                                abs_y - 5, 
                                                0, 0)
                time.sleep(0.05)
                
                # Move to exact position
                ctypes.windll.user32.mouse_event(0x8000 | 0x0001 | 0x4000, 
                                                abs_x, 
                                                abs_y, 
                                                0, 0)
                time.sleep(0.05)
                
                # Click
                ctypes.windll.user32.mouse_event(0x0002, 0, 0, 0, 0)  # LEFT DOWN
                time.sleep(0.1)  # Slightly longer hold
                ctypes.windll.user32.mouse_event(0x0004, 0, 0, 0, 0)  # LEFT UP
                time.sleep(0.1)

                # Verify if click was successful by checking if the button disappeared
                time.sleep(0.5)
                new_screenshot = get_window_screenshot(window)
                if new_screenshot is not None:
                    new_res = cv2.matchTemplate(new_screenshot, template, cv2.TM_CCOEFF_NORMED)
                    _, new_max_val, _, _ = cv2.minMaxLoc(new_res)
                    if new_max_val < confidence * 0.8:  # Button disappeared or changed
                        break  # Assume click was successful

            if DEBUG:
                log.debug(f"Clicked at virtual coordinates ({target_x}, {target_y}) with movement")
            return True
        else:
            log.warning(f"'{template_name}' not matched with sufficient confidence.")
            return False

    except Exception as e:
        log.error(f"Error clicking template '{template_name}': {str(e)}")
        return False

def rejoin_game():
    global run_start_time, victory_detected, upgrades_purchased

    log.info("Attempting to rejoin and restart farming...")

    if not focus_roblox_window():
        log.error("Roblox window not found, skipping rejoin.")
        return False

    try:
        # Click reconnect button just once
        if not click_on_template("reconnect.png", confidence=0.6):
            log.error("Failed to click reconnect button")
            return False

        log.info("Reconnect button clicked successfully, waiting for game to load...")
        
        # Wait for reconnect button to disappear (game is loading)
        start_time = time.time()
        timeout = 30
        last_reconnect_check = 0
        reconnect_check_interval = 2
        
        while time.time() - start_time < timeout:
            current_time = time.time()
            
            if current_time - last_reconnect_check > reconnect_check_interval:
                window = get_roblox_window()
                if window:
                    screenshot = get_window_screenshot(window)
                    if screenshot is not None:
                        # Check if reconnect button is still visible
                        if not template_match(screenshot, get_template_path("reconnect.png"), confidence=0.5):
                            log.info("Reconnect button disappeared, game is loading")
                            break
                last_reconnect_check = current_time
            
            time.sleep(0.5)

        # Now wait for menu to appear
        menu_timeout = 60
        menu_start_time = time.time()
        
        while time.time() - menu_start_time < menu_timeout:
            window = get_roblox_window()
            if window:
                screenshot = get_window_screenshot(window)
                if screenshot is not None:
                    # Check if menu is visible
                    if template_match(screenshot, get_template_path("menu.png")):
                        log.info("Menu detected, proceeding with rejoining process")
                        time.sleep(5)  # Additional wait
                        break
                    
                    # Check if we got disconnected again
                    if template_match(screenshot, get_template_path("reconnect.png")):
                        log.warning("Disconnected again during loading")
                        return False
            
            time.sleep(1)

        else:
            log.error("Menu didn't appear after rejoin (timeout).")
            return False

        # Proceed with rejoining process
        if not teleport_to_endless():
            log.error("Failed to teleport to endless area.")
            return False
        time.sleep(1)

        if not move_to_endless():
            log.error("Failed to move to endless area.")
            return False
        time.sleep(1)

        if not toggle_troops():
            log.error("Failed to enable auto troops.")
            return False

        # Reset run stats
        run_start_time = time.time()
        victory_detected = False
        upgrades_purchased = defaultdict(lambda: defaultdict(int))
        log.info("Rejoined successfully and restarted run.")
        return True

    except Exception as e:
        log.error(f"Rejoin error: {str(e)}")
        return False

def is_key_pressed(key, check_hold=False):
    global last_key_press_time, key_hold_state
    
    current_time = time.time()
    
    if not keyboard.is_pressed(key):
        key_hold_state[key] = False
        return False
    
    if check_hold and key_hold_state[key]:
        if current_time - last_key_press_time[key] > KEY_HOLD_TIME:
            return True
        return False
    
    if current_time - last_key_press_time[key] < DEBOUNCE_TIME:
        return False
    
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
            time.sleep(0.5)
            
        if is_key_pressed(START_KEY): # Perform single scan/select cycle
            log.debug("Manual scan triggered")
            window = get_roblox_window()
            if window:
                upgrades = scan_for_upgrades()
                if upgrades:
                    if focus_roblox_window():
                        select_best_upgrade(upgrades)
                else:
                    log.debug("No upgrades found")
            while is_key_pressed(START_KEY): # Wait for key release
                time.sleep(0.1)
            time.sleep(0.2)
            
        time.sleep(0.1)

def check_for_update():
    try:
        VERSION_URL = "https://raw.githubusercontent.com/daftuyda/AFM/refs/heads/main/version.txt"
        EXE_URL = "https://github.com/daftuyda/AFM/releases/latest/download/AFM.exe"
        CURRENT_VERSION = __version__
        CURRENT_EXE = sys.argv[0]

        res = requests.get(VERSION_URL)
        if res.status_code != 200:
            log.warning("Could not fetch version info.")
            return

        latest_version = res.text.strip()
        if latest_version == CURRENT_VERSION:
            log.debug("You are on the latest version.")
            return

        if not AUTO_UPDATES:
            log.info("Update available, but auto_updates is disabled.")
            return

        print(f"\n🚨 Update found! {CURRENT_VERSION} → {latest_version}")
        print("Download update? (Y/n), defaulting to skip in 5 seconds...\n")

        answer = None
        def get_input():
            nonlocal answer
            answer = input("Update now? [y/N]: ").strip().lower()

        thread = threading.Thread(target=get_input)
        thread.start()
        thread.join(timeout=5)

        if answer != 'y':
            print("⏩ Skipping update.")
            return

        print("⬇️ Downloading update...")

        update = requests.get(EXE_URL, stream=True)
        if update.status_code != 200:
            print("❌ Failed to download update.")
            return

        # Save to temp file
        temp_exe = tempfile.NamedTemporaryFile(delete=False, suffix=".exe")
        for chunk in update.iter_content(chunk_size=8192):
            temp_exe.write(chunk)
        temp_exe.close()

        print("✅ Update downloaded. Restarting...")

        # Launch updater
        subprocess.Popen([
            "cmd", "/c",
            f"timeout 1 > NUL && move /Y \"{temp_exe.name}\" \"{CURRENT_EXE}\" && start \"\" \"{CURRENT_EXE}\""
        ], shell=True)

        exit()

    except Exception as e:
        log.error(f"Update check failed: {e}")

def main_loop():
    global run_start_time, victory_detected, is_running, is_paused
    
    log.info("=== AFK Endless Macro ===")
    log.info(f"Mode: {MODE.capitalize()} Scan | Press {START_KEY} to begin." if not AUTO_START else "Auto-start: Enabled.")
    log.info(f"Press {PAUSE_KEY} to pause/resume | {STOP_KEY} to stop")
    log.info(f"Money Mode: {'ENABLED' if MONEY_MODE else 'disabled'}")
    log.info(f"High priority upgrades: {', '.join([upgrade.replace('.png', '') for upgrade in HIGH_PRIORITY])}")
    log.info(f"Low priority upgrades: {', '.join([upgrade.replace('.png', '') for upgrade in LOW_PRIORITY])}")
    
    run_start_time = time.time()
    victory_detected = False
    
    if MODE == "manual":
        manual_mode_loop()
        return
    
    if AUTO_START:
        is_running = True
        
    if MAXIMIZE_WINDOW:
        focus_roblox_window()
        
    # === Start Threads ===
    upgrade_thread = UpgradeScannerThread()
    victory_thread = VictoryCheckerThread()
    upgrade_thread.start()
    victory_thread.start()
    
    disconnect_thread = None
    if AUTO_RECONNECT:
        disconnect_thread = DisconnectionWatcherThread()
        disconnect_thread.start()
    
    try:
        while True:
            if is_key_pressed(STOP_KEY):
                log.info("=== Stopped ===")
                is_running = False
                allow_sleep()
                break

            if is_key_pressed(START_KEY):
                if not is_running:
                    log.info("=== Started ===")
                    is_running = True
                    run_start_time = time.time()
                time.sleep(0.5)

            if is_key_pressed(PAUSE_KEY):
                is_paused = not is_paused
                log.info(f"=== {'Paused' if is_paused else 'Resumed'} ===")
                time.sleep(0.5)

            if is_running and not is_paused: # Pull victory result
                if victory_thread.get_victory():
                    if victory_detected:
                        restart_run()

                upgrades = upgrade_thread.get_upgrades() # Pull upgrade scan result
                if upgrades:
                    if focus_roblox_window():
                        select_best_upgrade(upgrades)

            time.sleep(0.05)

    finally: # Ensure threads are stopped when loop ends
        upgrade_thread.stop()
        victory_thread.stop()
        
        upgrade_thread.join(timeout=5)
        victory_thread.join(timeout=5)
        
        if disconnect_thread:
            disconnect_thread.stop()
            disconnect_thread.join(timeout=5)

if __name__ == "__main__":
    main_loop()
