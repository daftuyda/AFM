import time
import pydirectinput
pydirectinput.FAILSAFE = False
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
import random
import gc
from logging.handlers import RotatingFileHandler
from io import BytesIO
from datetime import datetime
from collections import defaultdict

__version__ = "1.5.1"

DEFAULT_CONFIG = {
    "ultrawide_mode": False,
    "maximize_window": True,
    "auto_reconnect": True,
    "afk_prevention": True,
    "gold_wave_tracking": False,
    "scan_interval": 5,
    "start_key": "F6",
    "pause_key": "F7",
    "stop_key": "F8",
    "auto_start": False,
    "webhook_url": "",
    "mode": "auto",
    "money_mode": False,
    "wave_threshold": 120,
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
    
    response = input(f"Enable AFK prevention (jump every 10-15 mins)? [Y/n]: ").strip().lower()
    config["afk_prevention"] = False if response == "n" else True
    
    response = input(f"Enable Gold and Wave Tracking? [y/N]: ").strip().lower()
    config["gold_wave_tracking"] = response == "y"
    
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
    
    response = input(f"Wave threshold (default: {DEFAULT_CONFIG['wave_threshold']}): ").strip()
    if response:
        try:
            config["wave_threshold"] = int(response)
        except ValueError:
            print("Invalid input. Using default.")
    
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
AFK_PREVENTION = config.get("afk_prevention", True)
GOLD_WAVE_TRACKING = config.get("gold_wave_tracking", False)
AUTO_START = config.get("auto_start", False)
SCAN_INTERVAL = config.get("scan_interval", 5)
BUTTON_DELAY = config.get("button_delay", 0.2)
START_KEY = config.get("start_key", "f6").lower()
PAUSE_KEY = config.get("pause_key", "f7").lower()
STOP_KEY = config.get("stop_key", "f8").lower()
DISCORD_WEBHOOK_URL = config.get("webhook_url", "")
MONEY_MODE = config.get("money_mode", False)
WAVE_THRESHOLD = config.get("wave_threshold", 120)
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
last_upgrade_time = 0
last_disconnect_check = 0
last_wave_number = 0
last_wave_gold_check = 0
WAVE_GOLD_INTERVAL = SCAN_INTERVAL * 3
last_wave_reset_time = 0
WAVE_RESET_INTERVAL = 3 * 60
last_ui_check = 0
UI_CHECK_INTERVAL = 2
victory_detected = False
upgrades_purchased = defaultdict(lambda: defaultdict(int))
is_running = False
is_paused = False
last_key_press_time = defaultdict(float)
key_hold_state = defaultdict(bool)
upgrade_allowed = True
post_boss_missed_upgrade = False
grand_total_gold = 0
gold_start = None
gold_last = None
gold_log = []

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
                    
                    detect_ui_elements_and_respond()
            time.sleep(SCAN_INTERVAL)

    def get_upgrades(self):
        with self.lock:
            upgrades = self.upgrades
            self.upgrades = None
        return upgrades

    def stop(self):
        self.running = False

class UIDetectorThread(threading.Thread):
    def __init__(self, interval=3):
        super().__init__(daemon=True)
        self.running = False
        self.interval = interval

    def run(self):
        self.running = True
        while self.running:
            if is_running and not is_paused:
                try:
                    detect_ui_elements_and_respond()
                except Exception as e:
                    log.error(f"[UI Detection] Error: {e}")
            time.sleep(self.interval)

    def stop(self):
        self.running = False

class GoldWaveScannerThread(threading.Thread):
    def __init__(self):
        super().__init__(daemon=True)
        self.running = False

    def run(self):
        global last_wave_number, last_wave_gold_check, last_wave_reset_time, gold_start, gold_last

        self.running = True
        while self.running:
            if is_running and not is_paused:
                now = time.time()

                # Wave regression reset logic
                if now - last_wave_reset_time > WAVE_RESET_INTERVAL:
                    if DEBUG:
                        log.debug("[Wave] Resetting last_wave_number due to 3-min timeout.")
                    last_wave_number = 0
                    last_wave_reset_time = now

                # Perform wave scan
                scanned_wave = get_current_wave_number(last_wave=last_wave_number)
                if scanned_wave is not None:
                    if scanned_wave >= last_wave_number:
                        last_wave_number = scanned_wave
                        if DEBUG:
                            log.debug(f"[WaveScan] Updated wave: {last_wave_number}")
                else:
                    if DEBUG:
                        log.debug("[WaveScan] No valid wave read.")

                # Perform gold scan
                current_gold = get_current_gold_amount()
                if current_gold is not None:
                    if gold_start is None:
                        gold_start = current_gold
                    gold_last = current_gold
                    if DEBUG:
                        log.debug(f"[GoldScan] Current gold: {current_gold}")
                                
            time.sleep(15)

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

class AFKPreventionThread(threading.Thread):
    def __init__(self):
        super().__init__(daemon=True)
        self.running = False

    def run(self):
        self.running = True
        while self.running:
            if is_running and not is_paused:
                log.debug("[AFK] Simulating jump")
                pydirectinput.keyDown('space')
                time.sleep(0.1)
                pydirectinput.keyUp('space')
            time.sleep(random.randint(600, 900))  # 10 to 15 mins

    def stop(self):
        self.running = False

class KeyListenerThread(threading.Thread):
    def __init__(self):
        super().__init__(daemon=True)
        self.running = True
        self.last_pressed = {}

    def check_key(self, key, action):
        if keyboard.is_pressed(key):
            if not self.last_pressed.get(key, False):
                action()
                self.last_pressed[key] = True
        else:
            self.last_pressed[key] = False

    def run(self):
        global is_running, is_paused

        while self.running:
            self.check_key(START_KEY, self.start_script)
            self.check_key(PAUSE_KEY, self.toggle_pause)
            self.check_key(STOP_KEY, self.stop_script)
            time.sleep(0.05)

    def start_script(self):
        global is_running
        if not is_running:
            is_running = True
            log.info("[Keybind] Script started")

    def toggle_pause(self):
        global is_paused
        is_paused = not is_paused
        log.info(f"[Keybind] Script {'paused' if is_paused else 'resumed'}")

    def stop_script(self):
        global is_running
        is_running = False
        self.running = False
        log.info("[Keybind] Script stopped via keybind")
        os._exit(0)

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
        candidates = gw.getWindowsWithTitle("Roblox")
        best_window = None

        for window in candidates:
            w, h = window.width, window.height

            try:
                if window.isMinimized or window.width < 500 or window.height < 400:
                    continue
            except Exception as e:
                log.debug(f"Skipping window due to error: {e}")
                continue
            if w < 500 or h < 400:  # Arbitrary sanity threshold
                if DEBUG:
                    log.debug(f"Skipping small window: {w}x{h} ({window.title})")
                continue

            best_window = window
            break

        if not best_window:
            if DEBUG:
                log.warning("No valid Roblox window found.")
            return None

        return best_window

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
                unknown_count = sum(1 for u in found_upgrades if u['rarity'] == "Unknown")
                if len(found_upgrades) < 3 or unknown_count > 0:
                    if DEBUG:
                        log.debug(f"Retrying scan: {len(found_upgrades)} upgrades found, {unknown_count} unknown rarities")
                    attempts += 1
                    time.sleep(0.6)
                    continue
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
        
        return closest_match if min_distance < 60 else "Unknown" # Increase threshold for more leniency
        
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
    
    if not focus_roblox_window():
        log.warning("Failed to focus Roblox window before upgrade.")
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
        time.sleep(0.1)
        pydirectinput.keyUp(UI_TOGGLE_KEY)
        time.sleep(1 if attempt == 0 else 0.8) # Wait longer on first attempt

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
        time.sleep(0.1)
        pydirectinput.keyUp('left')
        time.sleep(BUTTON_DELAY)
        
        pydirectinput.keyDown('down')
        time.sleep(0.1)
        pydirectinput.keyUp('down')
        time.sleep(BUTTON_DELAY)
        
        for _ in range(position_index):
            pydirectinput.keyDown('right')
            time.sleep(0.1)
            pydirectinput.keyUp('right')
            time.sleep(BUTTON_DELAY)
        
        pydirectinput.keyDown('enter')
        time.sleep(0.1)
        pydirectinput.keyUp('enter')
        time.sleep(BUTTON_DELAY)
        
        pydirectinput.keyDown(UI_TOGGLE_KEY)
        time.sleep(0.1)
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

def get_current_wave_number(last_wave=None):
    window = get_roblox_window()
    if not window:
        return None

    screenshot = get_window_screenshot(window)
    if screenshot is None:
        return None

    gray = cv2.cvtColor(screenshot, cv2.COLOR_BGR2GRAY)

    h, w = gray.shape
    x = int(w * 680 / 1920)
    y = int(h * 45 / 1080)
    roi_w = int(w * 70 / 1920)
    roi_h = int(h * 35 / 1080)
    roi = gray[y:y+roi_h, x:x+roi_w]
    
    # Check for E prefix in wave ROI
    e_template = cv2.imread(os.path.join("numbers", "E.png"), cv2.IMREAD_GRAYSCALE)
    if e_template is not None:
        scale_factor = roi.shape[0] / e_template.shape[0]
        e_template = cv2.resize(e_template, (int(e_template.shape[1] * scale_factor), roi.shape[0]))

        res = cv2.matchTemplate(roi, e_template, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(res)

        if DEBUG:
            log.debug(f"[Wave] E detection confidence: {max_val:.2f}")

        if max_val >= 0.7:
            # Crop out E region from the ROI
            e_width = e_template.shape[1]
            roi = roi[:, e_width:]  # Strip E from the left side
            if DEBUG:
                log.debug("[Wave] E prefix detected and removed from ROI.")

    roi = cv2.equalizeHist(roi)
    roi = cv2.GaussianBlur(roi, (3, 3), 0)
    _, roi = cv2.threshold(roi, 160, 255, cv2.THRESH_BINARY)

    # Strengthen digits (especially thin ones like 1, 7)
    kernel = np.ones((2, 2), np.uint8)
    roi = cv2.dilate(roi, kernel, iterations=1)

    # Resize ROI slightly larger to match templates from higher wave counts
    roi = cv2.resize(roi, None, fx=1.2, fy=1.2, interpolation=cv2.INTER_LINEAR)
        
    # Digit matching logic
    h_roi, w_roi = roi.shape
    heatmap = np.zeros((h_roi, w_roi), dtype=np.float32)
    digit_map = np.full((h_roi, w_roi), -1, dtype=np.int32)

    for digit in reversed(range(10)):  # Try 9 down to 0
        template_path = os.path.join("numbers", f"{digit}.png")
        template = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)
        if template is None:
            continue

        th, tw = template.shape
        res = cv2.matchTemplate(roi, template, cv2.TM_CCOEFF_NORMED)

        for y_res in range(res.shape[0]):
            for x_res in range(res.shape[1]):
                score = res[y_res, x_res]
                if score > heatmap[y_res, x_res]:
                    heatmap[y_res, x_res] = score
                    digit_map[y_res, x_res] = digit

    # Collect best matches above threshold
    threshold = 0.7
    matches = []
    for y_scan in range(digit_map.shape[0]):
        for x_scan in range(digit_map.shape[1]):
            digit = digit_map[y_scan, x_scan]
            score = heatmap[y_scan, x_scan]

            if digit != -1 and score >= threshold:
                matches.append((x_scan, digit, score))

    if not matches:
        return None

    matches.sort(key=lambda m: m[0])

    # Filter overlapping digits
    filtered = []
    last_x = -999
    for x_pos, digit, score in matches:
        if x_pos - last_x > 10:
            filtered.append((x_pos, digit, score))
            last_x = x_pos

    digits = [str(d) for _, d, _ in filtered]

    if DEBUG:
        digits_str = "".join([str(d) for (_, d, _) in filtered])
        log.debug(f"Matched digits: {digits_str}")

    try:
        wave = int("".join(digits))

        if last_wave is not None and wave < last_wave:
            if DEBUG:
                log.debug(f"Ignoring regressed wave number: {wave} < {last_wave}")
            return None
        return wave

    except ValueError:
        return None

def is_boss_round(wave):
    return wave % 5 == 0

def get_current_gold_amount():
    window = get_roblox_window()
    if not window:
        return None

    screenshot = get_window_screenshot(window)
    if screenshot is None:
        return None

    gray = cv2.cvtColor(screenshot, cv2.COLOR_BGR2GRAY)

    # Estimate ROI for gold (top-right)
    h, w = gray.shape
    x = int(w * 70 / 1920)
    y = int(h * 660 / 1080)
    roi_w = int(w * 180 / 1920)
    roi_h = int(h * 50 / 1080)
    roi = gray[y:y+roi_h, x:x+roi_w]

    # Preprocess ROI for better OCR
    roi = cv2.equalizeHist(roi)
    roi = cv2.GaussianBlur(roi, (3, 3), 0)
    _, roi = cv2.threshold(roi, 80, 255, cv2.THRESH_BINARY_INV)

    h_roi, w_roi = roi.shape
    heatmap = np.zeros((h_roi, w_roi), dtype=np.float32)
    digit_map = np.full((h_roi, w_roi), -1, dtype=np.int32)

    for digit in reversed(range(10)):
        template_path = os.path.join("numbers", f"gold_{digit}.png")
        template = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)
        if template is None:
            continue

        th, tw = template.shape
        res = cv2.matchTemplate(roi, template, cv2.TM_CCOEFF_NORMED)

        for y_res in range(res.shape[0]):
            for x_res in range(res.shape[1]):
                score = res[y_res, x_res]
                if score > heatmap[y_res, x_res]:
                    heatmap[y_res, x_res] = score
                    digit_map[y_res, x_res] = digit

    threshold = 0.7
    matches = []
    for y_scan in range(digit_map.shape[0]):
        for x_scan in range(digit_map.shape[1]):
            digit = digit_map[y_scan, x_scan]
            score = heatmap[y_scan, x_scan]
            if digit != -1 and score >= threshold:
                matches.append((x_scan, digit, score))

    if not matches:
        return None

    matches.sort(key=lambda m: m[0])
    filtered = []
    last_x = -999
    for x_pos, digit, score in matches:
        if x_pos - last_x > 10:
            filtered.append((x_pos, digit, score))
            last_x = x_pos

    digits = [str(d) for _, d, _ in filtered]

    try:
        return int("".join(digits))
    except ValueError:
        return None

def detect_victory():
    global last_victory_time, victory_detected, run_start_time, gold_start, gold_last, grand_total_gold
    
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
        
        time.sleep(1) # Wait for results to load
        
        screenshot = get_window_screenshot(window)
        
        if confidence > 0.7:
            run_time = 0
            if run_start_time > 0:
                run_time = current_time - run_start_time
                run_start_time = 0
                
            last_victory_time = current_time
            victory_detected = True
            if gold_start is not None and gold_last is not None:
                gold_gained = gold_last - gold_start
                global grand_total_gold
                grand_total_gold += gold_gained
                log.info(f"[Victory] Gold gained: {gold_gained} | Grand total: {grand_total_gold}")
                
            if DEBUG:
                log.debug(f"Victory detected! (confidence: {confidence:.2f})")
            if DISCORD_WEBHOOK_URL:
                log.debug("Uploading screenshot and stats to Discord...")
                if not upload_to_discord(screenshot, run_time):
                    log.error("Failed to upload to Discord")
            else:
                if DEBUG:
                    log.debug("Discord webhook URL not set, skipping upload")
            
            gold_start = None
            gold_last = None
            gold_log = []
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
        _, img_encoded = cv2.imencode('.png', screenshot)
        img_bytes = BytesIO(img_encoded.tobytes())
        
        minutes, seconds = divmod(run_time, 60)
        time_str = f"{int(minutes):02d}:{int(seconds):02d}"

        # Final wave
        final_wave = last_wave_number

        # Gold stats
        gold_this_run = gold_last - gold_start if gold_start is not None and gold_last is not None else 0
        gold_per_min = (gold_this_run / run_time) * 60 if run_time > 0 else 0
        gold_per_hr = (gold_this_run / run_time) * 3600 if run_time > 0 else 0

        summary = generate_upgrade_summary()

        gold_report = (
            f"**Run Summary**\n"
            f"• Wave Reached: {final_wave}\n"
            f"• Time: {time_str}\n"
            f"• Gold This Run: {gold_this_run:,}\n"
            f"• Gold/min: {gold_per_min:,.2f}\n"
            f"• Gold/hr: {gold_per_hr:,.2f}\n"
            f"• Grand Total Gold: {grand_total_gold:,}\n\n"
        )

        payload = {
            "content": f"{gold_report}{summary}",
            "username": "AFM"
        }
        
        files = {'file': ('victory.png', img_bytes, 'image/png')}
        
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
    global run_start_time, victory_detected, upgrades_purchased, last_wave_number
    
    log.debug("Attempting to restart run")
    try:
        time.sleep(BUTTON_DELAY)
        pydirectinput.keyDown(UI_TOGGLE_KEY)
        time.sleep(0.1)
        pydirectinput.keyUp(UI_TOGGLE_KEY)
        time.sleep(BUTTON_DELAY)
        
        for _ in range(3):
            pydirectinput.keyDown('down')
            time.sleep(0.1)
            pydirectinput.keyUp('down')
            time.sleep(BUTTON_DELAY)
            
        pydirectinput.keyDown('enter')
        time.sleep(0.1)
        pydirectinput.keyUp('enter')
        time.sleep(BUTTON_DELAY)
        
        pydirectinput.keyDown(UI_TOGGLE_KEY)
        time.sleep(0.1)
        pydirectinput.keyUp(UI_TOGGLE_KEY)
        time.sleep(BUTTON_DELAY)
        
        run_start_time = time.time()
        victory_detected = False
        last_wave_number = 0
        upgrades_purchased = defaultdict(lambda: defaultdict(int))
        return True
        
    except Exception as e:
        log.error(f"Run restart error: {str(e)}")
        return False

def detect_ui_elements_and_respond():
    try:
        window = get_roblox_window()
        if not window:
            return

        screenshot = get_window_screenshot(window)
        if screenshot is None:
            return

        settings_detected = template_match(screenshot, get_template_path("settings.png"), confidence=0.7)
        lobby_detected = template_match(screenshot, get_template_path("lobby.png"), confidence=0.7)

        if settings_detected:
            log.info("Detected settings UI. Toggling UI.")
            pydirectinput.press(UI_TOGGLE_KEY)
            time.sleep(BUTTON_DELAY)
            for _ in range(3):
                pydirectinput.press('left')
                time.sleep(BUTTON_DELAY)
            
            pydirectinput.press('up')
            time.sleep(BUTTON_DELAY)
            pydirectinput.press('enter')
            time.sleep(BUTTON_DELAY)

        if lobby_detected:
            log.info("Detected lobby UI. Attempting to escape.")
            pydirectinput.press(UI_TOGGLE_KEY)
            time.sleep(BUTTON_DELAY)
            for _ in range(5):
                pydirectinput.press('up')
                time.sleep(BUTTON_DELAY)
            
            pydirectinput.press('enter')
            time.sleep(BUTTON_DELAY)
            
            time.sleep(1)
            # Check again
            screenshot = get_window_screenshot(window)
            if template_match(screenshot, get_template_path("lobby.png"), confidence=0.7):
                pydirectinput.press('down')
                time.sleep(BUTTON_DELAY)
                pydirectinput.press('enter')
                time.sleep(BUTTON_DELAY)
                
            pydirectinput.press(UI_TOGGLE_KEY)
            time.sleep(BUTTON_DELAY)

    except Exception as e:
        log.error(f"UI detection error: {str(e)}")

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
        time.sleep(0.1)
        pydirectinput.keyUp(UI_TOGGLE_KEY)
        time.sleep(BUTTON_DELAY)
        
        pydirectinput.keyDown('down')
        time.sleep(0.1)
        pydirectinput.keyUp('down')
        time.sleep(BUTTON_DELAY)
        
        pydirectinput.keyDown('enter')
        time.sleep(0.1)
        pydirectinput.keyUp('enter')
        time.sleep(BUTTON_DELAY)
        
        for _ in range(2):
            pydirectinput.keyDown('down')
            time.sleep(0.1)
            pydirectinput.keyUp('down')
            time.sleep(BUTTON_DELAY)
        
        pydirectinput.keyDown('enter')
        time.sleep(0.1)
        pydirectinput.keyUp('enter')
        time.sleep(BUTTON_DELAY)
        
        pydirectinput.keyDown(UI_TOGGLE_KEY)
        time.sleep(0.1)
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
        time.sleep(0.1)
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
        time.sleep(0.1)
        pydirectinput.keyUp(UI_TOGGLE_KEY)
        time.sleep(BUTTON_DELAY)

        # Navigate to auto troops option
        for _ in range(7):
            pydirectinput.keyDown('down')
            time.sleep(0.1)
            pydirectinput.keyUp('down')
            time.sleep(BUTTON_DELAY)

        if not auto_detected:
            log.error("Auto troops button not found after 10 seconds")
            # Close menu
            pydirectinput.keyDown(UI_TOGGLE_KEY)
            time.sleep(0.1)
            pydirectinput.keyUp(UI_TOGGLE_KEY)
            time.sleep(BUTTON_DELAY)
            return False

        # Press enter to enable auto troops
        pydirectinput.keyDown('enter')
        time.sleep(0.1)
        pydirectinput.keyUp('enter')
        time.sleep(BUTTON_DELAY)

        # Close the menu
        pydirectinput.keyDown(UI_TOGGLE_KEY)
        time.sleep(0.1)
        pydirectinput.keyUp(UI_TOGGLE_KEY)
        time.sleep(BUTTON_DELAY)
        
        log.info("Auto troops successfully enabled")
        return True
        
    except Exception as e:
        log.error(f"Auto Troops Error: {str(e)}")
        # Ensure menu is closed if error occurs
        pydirectinput.keyDown(UI_TOGGLE_KEY)
        time.sleep(0.1)
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

        return confidence >= 0.6

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

    while True:
        if not focus_roblox_window():
            log.error("Roblox window not found, retrying in 5 seconds...")
            time.sleep(5)
            continue

        try:
            # Click reconnect button
            if not click_on_template("reconnect.png", confidence=0.6):
                log.error("Failed to click reconnect button, retrying in 5 seconds...")
                time.sleep(5)
                continue

            log.info("Reconnect clicked, waiting for game to load...")

            # Wait for reconnect button to disappear
            start_time = time.time()
            while time.time() - start_time < 30:
                screenshot = get_window_screenshot(get_roblox_window())
                if screenshot is None:
                    continue
                if not template_match(screenshot, get_template_path("reconnect.png"), confidence=0.5):
                    break
                time.sleep(1)
            else:
                log.warning("Reconnect button still visible after timeout, retrying...")
                time.sleep(3)
                continue

            # Wait for menu to appear
            start_time = time.time()
            while time.time() - start_time < 60:
                screenshot = get_window_screenshot(get_roblox_window())
                if screenshot is None:
                    continue
                if template_match(screenshot, get_template_path("menu.png")):
                    log.info("Menu detected, proceeding.")
                    break
                if template_match(screenshot, get_template_path("reconnect.png")):
                    log.warning("Disconnected again during loading, retrying...")
                    time.sleep(3)
                    continue
                time.sleep(1)
            else:
                log.warning("Menu not detected in time. Retrying...")
                time.sleep(3)
                continue

            # Perform rejoin steps
            if not teleport_to_endless():
                log.error("Teleport failed, retrying...")
                time.sleep(3)
                continue
            if not move_to_endless():
                log.error("Move failed, retrying...")
                time.sleep(3)
                continue
            if not toggle_troops():
                log.error("Toggle auto troops failed, retrying...")
                time.sleep(3)
                continue

            # Reset run state
            run_start_time = time.time()
            victory_detected = False
            upgrades_purchased = defaultdict(lambda: defaultdict(int))
            log.info("Rejoined and restarted successfully.")
            return True

        except Exception as e:
            log.error(f"Rejoin error: {str(e)} — retrying in 5 seconds...")
            time.sleep(5)

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
        cmd = f'timeout 1 > NUL && move /Y "{temp_exe.name}" "{CURRENT_EXE}" && start "" "{CURRENT_EXE}"'
        subprocess.Popen(cmd, shell=True)

        sys.exit()

    except Exception as e:
        log.error(f"Update check failed: {e}")

def main_loop():
    global run_start_time, victory_detected, is_running, is_paused, upgrade_allowed, last_wave_number, last_wave_reset_time, post_boss_missed_upgrade
    global last_upgrade_time, gold_start, gold_last, gold_log, grand_grand_total_gold, wave_scan_block_until, last_wave_gold_check 
    
    log.info("=== AFK Endless Macro ===")
    log.info(f"Mode: {MODE.capitalize()} Scan | Press {START_KEY} to begin." if not AUTO_START else "Auto-start: Enabled.")
    log.info(f"Press {PAUSE_KEY} to pause/resume | {STOP_KEY} to stop")
    log.info(f"Money Mode: {'ENABLED' if MONEY_MODE else 'disabled'}")
    log.info(f"High priority upgrades: {', '.join([upgrade.replace('.png', '') for upgrade in HIGH_PRIORITY])}")
    log.info(f"Low priority upgrades: {', '.join([upgrade.replace('.png', '') for upgrade in LOW_PRIORITY])}")
    
    run_start_time = time.time()
    victory_detected = False
    wave_scan_block_until = 0
    
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
    ui_thread = UIDetectorThread(interval=5)
    key_thread = KeyListenerThread()
    upgrade_thread.start()
    victory_thread.start()
    ui_thread.start()
    key_thread.start()
    
    gold_wave_thread = None
    if GOLD_WAVE_TRACKING:
        gold_wave_thread = GoldWaveScannerThread()
        gold_wave_thread.start()
    
    afk_thread = None
    if AFK_PREVENTION:
        afk_thread = AFKPreventionThread()
        afk_thread.start()
    
    disconnect_thread = None
    if AUTO_RECONNECT:
        disconnect_thread = DisconnectionWatcherThread()
        disconnect_thread.start()
    
    try:
        while True:
            time.sleep(0.05)

            if not is_running or is_paused:
                continue

            if victory_thread.get_victory():
                if victory_detected:
                    restart_run()
                    wave_scan_block_until = time.time() + 5
                    last_wave_number = 0

            upgrades = upgrade_thread.get_upgrades()
            wave = last_wave_number

            if upgrades:
                if wave < WAVE_THRESHOLD:
                    if select_best_upgrade(upgrades):
                        last_upgrade_time = time.time()
                        current_gold = gold_last
                        if current_gold is not None:
                            if gold_start is None:
                                gold_start = current_gold
                            gold_diff = current_gold - gold_start
                            time_diff = time.time() - run_start_time if run_start_time else 1
                            gold_per_hour = (gold_diff / time_diff) * 3600

                            if DEBUG:
                                log.debug(f"[Upgrade] Gold: {current_gold} | Gained: {gold_diff} | Rate: {gold_per_hour:.2f}/hr")

                else:
                    # Post-threshold wave upgrade logic
                    if wave % 5 == 1 or wave % 5 == 6:
                        if upgrade_allowed:
                            log.info(f"Post-boss wave {wave}: upgrading once.")
                            if select_best_upgrade(upgrades):
                                last_upgrade_time = time.time()
                                upgrade_allowed = False
                                post_boss_missed_upgrade = False
                    elif post_boss_missed_upgrade:
                        log.info(f"Missed post-boss. Performing safety upgrade at wave {wave}.")
                        if select_best_upgrade(upgrades):
                            last_upgrade_time = time.time()
                            post_boss_missed_upgrade = False
                            upgrade_allowed = False
                    elif wave % 5 in [0, 5]:
                        upgrade_allowed = True
                        post_boss_missed_upgrade = True
                    else:
                        if DEBUG:
                            log.debug(f"Holding upgrade at wave {wave} (waiting for next post-boss)")
            
            if int(time.time()) % 30 == 0:
                gc.collect()

    except KeyboardInterrupt:
        log.info("Shutting down...")

    finally:
        upgrade_thread.stop()
        victory_thread.stop()
        ui_thread.stop()
        key_thread.running = False

        upgrade_thread.join(timeout=2)
        victory_thread.join(timeout=2)
        ui_thread.join(timeout=2)
        key_thread.join(timeout=2)
        
        if gold_wave_thread:
            gold_wave_thread.stop()
            gold_wave_thread.join(timeout=2)
        
        if afk_thread:
            afk_thread.stop()
            afk_thread.join(timeout=2)
        
        if disconnect_thread:
            disconnect_thread.stop()
            disconnect_thread.join(timeout=2)

if __name__ == "__main__":
    check_for_update()
    main_loop()
