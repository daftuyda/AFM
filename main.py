import time
import pydirectinput
import keyboard
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
import random
import gc
from logging.handlers import RotatingFileHandler
from io import BytesIO
from collections import defaultdict

pydirectinput.FAILSAFE = False

DEFAULT_CONFIG = {
    "ultrawide_mode": False,
    "scan_interval": 5,
    "start_key": "F6",
    "pause_key": "F7",
    "stop_key": "F8",
    "ui_detection_enabled": False,
    "afk_prevention": True,
    "auto_start": False,
    "money_mode": False,
    "wave_threshold": 120,
    "capacity_upgrades_delay": 15,
    "max_total_cap_upgrades": 17,
    "team_change_interval": 60,
    "enable_ult_clicker": True,
    "button_delay": 0.2,
    "high_priority": [
        "boss",
        "atk",
        "health",
        "units",
        "enemy"
    ],
    "low_priority": [
        "heal",
        "speed",
        "luck",
        "regen",
        "discount",
        "jade"
    ],
    "multi_instance_support": False,
    "webhook_url": "",
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
ULTRAWIDE_MODE = config.get("ultrawide_mode", False)
AUTO_START = config.get("auto_start", False)
SCAN_INTERVAL = config.get("scan_interval", 5)
BUTTON_DELAY = config.get("button_delay", 0.2)
START_KEY = config.get("start_key", "f6")
PAUSE_KEY = config.get("pause_key", "f7")
STOP_KEY = config.get("stop_key", "f8")
UI_DETECTION_ENABLED = config.get("ui_detection_enabled", False)
AFK_PREVENTION = config.get("afk_prevention", True)
DISCORD_WEBHOOK_URL = config.get("webhook_url", "")
DEBUG = config.get("debug", False)
HIGH_PRIORITY = add_png_suffix(config.get("high_priority", DEFAULT_CONFIG["high_priority"]))
LOW_PRIORITY = add_png_suffix(config.get("low_priority", DEFAULT_CONFIG["low_priority"]))
UPGRADE_PRIORITY = HIGH_PRIORITY + LOW_PRIORITY
MONEY_MODE = config.get("money_mode", False)
WAVE_THRESHOLD = config.get("wave_threshold", 120)
MULTI_INSTANCE = config.get("multi_instance_support", False)
CAPACITY_UPGRADES_DELAY = config.get("capacity_upgrades_delay", 15)
MAX_TOTAL_CAP_UPGRADES = config.get("max_total_cap_upgrades", 17)
TEAM_CHANGE_INTERVAL = config.get("team_change_interval", 60)
ENABLE_ULT_CLICKER = config.get("enable_ult_clicker", True)

IMAGE_FOLDER = "images"
CONFIDENCE_THRESHOLD = 0.7
UI_TOGGLE_KEY = '\\'

main_cap_upgrades_done = 0
alt_cap_upgrades_done = 0
main_card_upgrades = 0
alt_card_upgrades = 0
last_victory_time = 0
run_start_time = 0
victory_detected = False
upgrades_purchased = defaultdict(lambda: defaultdict(int))
is_running = False
is_paused = False

IMAGE_FOLDER = "ultrawide" if ULTRAWIDE_MODE else "images"

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

class AFKPreventionThread(threading.Thread):
    def __init__(self):
        super().__init__(daemon=True)
        self.running = False

    def run(self):
        self.running = True
        while self.running:
            if is_running and not is_paused:
                log.debug("[AFK] Simulating jump")

                def _do_jump():
                    pydirectinput.keyDown('space')
                    time.sleep(0.1)
                    pydirectinput.keyUp('space')

                _do_jump()
                if MULTI_INSTANCE:
                    alt_tab()
                    time.sleep(0.5)
                    _do_jump()
                    alt_tab()

            time.sleep(random.randint(600, 900))

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
        log.info("[Keybind] Script stopped")
        os._exit(0)

class UltClickerThread(threading.Thread):
    def __init__(self):
        super().__init__(daemon=True)
        self.running = True
    
    @staticmethod
    def rdp_safe_click(x, y):
        ctypes.windll.user32.SetCursorPos(x, y)
        time.sleep(0.02)  # tiny delay to settle
        ctypes.windll.user32.mouse_event(2, 0, 0, 0, 0)  # left down
        ctypes.windll.user32.mouse_event(4, 0, 0, 0, 0)  # left up

    def run(self):
        while self.running:
            region = detect_yellow_border_region()
            if region:
                x, y, w, h = region

                # Skip invalid or too-small regions
                if w < 5 or h < 5:
                    log.warning("[ULT-CLICK] Detected yellow region too small to click safely")
                    time.sleep(1)
                    continue

                # Safe bounds for random point
                click_x = x + np.random.randint(int(w * 0.2), int(w * 0.8))
                click_y = y + np.random.randint(int(h * 0.2), int(h * 0.8))

                log.info(f"[ULT-CLICK] Clicking yellow ult at ({click_x}, {click_y})")
                self.rdp_safe_click(click_x, click_y)

                time.sleep(1.5)  # Short cooldown to avoid rapid spam
            else:
                time.sleep(0.5)
    
    def stop(self):
        self.running = False

def setup_logging():
    with open('afm_macro.log', 'w'):
        pass
    
    logger = logging.getLogger('AFM')
    logger.setLevel(logging.DEBUG)
    
    file_handler = RotatingFileHandler('afm_macro.log', encoding='utf-8')
    file_handler.setFormatter(logging.Formatter('[%(asctime)s] %(levelname)s: %(message)s', datefmt='%H:%M:%S'))
    
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter('[%(asctime)s] %(message)s', datefmt='%H:%M:%S'))
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    return logger

log = setup_logging()

def get_template_path(filename):
    return os.path.join(IMAGE_FOLDER, filename) if IMAGE_FOLDER else filename

def get_full_display_screenshot():
    try:
        with mss.mss() as sct:
            monitor = sct.monitors[1]  # Full primary display
            sct_img = sct.grab(monitor)
            img = np.array(sct_img)
            return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    except Exception as e:
        log.error(f"Display capture failed: {str(e)}")
        return None

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
            monitor = {
                "left": window.left,
                "top": window.top,
                "width": window.width,
                "height": window.height,
            }
            
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
        log.warning(f"Template not found: {template_path}")
        return 0.0, (0, 0)

    window_height, window_width = screenshot.shape[:2]

    BASE_WIDTH = 1920
    BASE_HEIGHT = 1080

    scale_w = window_width / BASE_WIDTH
    scale_h = window_height / BASE_HEIGHT
    scale = min(scale_w, scale_h)  # Keep proportions consistent

    new_width = int(template.shape[1] * scale)
    new_height = int(template.shape[0] * scale)
    resized_template = cv2.resize(template, (new_width, new_height))

    res = cv2.matchTemplate(screenshot, resized_template, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, max_loc = cv2.minMaxLoc(res)

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
            
            if MULTI_INSTANCE:
                screenshot = get_full_display_screenshot()
            else:
                screenshot = get_window_screenshot(window)
            if screenshot is None:
                attempts += 1
                continue
                
            window_height = screenshot.shape[0]
            window_width = screenshot.shape[1]
            
            card_width = int(window_width * 0.18)
            gap = int(window_width * 0.03)
            first_card_left = (window_width // 2) - int((card_width + gap) * 1.45)
            
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
        
        scan_x = int(width * 0.9)
        y_base = int(height * 0.57)
        
        sample_width = 10
        
        if scan_x + sample_width > width:
            scan_x = width - sample_width - 1
        
        sample_region = card_img[
            y_base - 2 : y_base + 3,
            scan_x : scan_x + sample_width
        ]
        
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
        if "unit" in upgrade['upgrade'].lower() and upgrade['rarity'] == "Common":
            if DEBUG:
                log.debug(f"Skipping Common unit upgrade: {upgrade['upgrade']} at pos {upgrade['position']}")
            continue
            
        if (upgrade['original_upgrade'] in ['atk.png', 'health.png'] and 
            not upgrade.get('is_percent', False)):
            if DEBUG:
                log.debug(f"Skipping flat {upgrade['original_upgrade']} upgrade at pos {upgrade['position']}")
            continue
            
        valid_upgrades.append(upgrade)
    
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
        
        boss_upgrades.sort(key=lambda x: (-RARITY_ORDER.index(x['rarity']), UPGRADE_PRIORITY.index(x['original_upgrade'])))
        
        other_upgrades.sort(key=lambda x: (0 if x['original_upgrade'] in HIGH_PRIORITY else 1, -RARITY_ORDER.index(x['rarity']), UPGRADE_PRIORITY.index(x['original_upgrade'])))
        
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
        valid_upgrades.sort(key=get_sort_key)

    if not MONEY_MODE:
        def get_sort_key(upgrade):
            if upgrade['original_upgrade'] in HIGH_PRIORITY:
                group = 0  # High priority
            else:
                group = 1  # Low priority

            return (group, -RARITY_ORDER.index(upgrade['rarity']), UPGRADE_PRIORITY.index(upgrade['original_upgrade']))

        valid_upgrades.sort(key=get_sort_key)

    if DEBUG:
        log.debug("Valid upgrades sorted:")
        for idx, upgrade in enumerate(valid_upgrades, 1):
            perc = " (PERCENT)" if upgrade.get('is_percent', False) else ""
            group = "HIGH" if UPGRADE_PRIORITY.index(upgrade['original_upgrade']) < 6 else "LOW"
            log.debug(f"{idx}. [{group}] {upgrade['upgrade']}{perc} ({upgrade['rarity']}) at pos {upgrade['position']}")

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

def toggle_ui_and_confirm(window=None, max_attempts=2):
    if not UI_DETECTION_ENABLED: 
        pydirectinput.keyDown(UI_TOGGLE_KEY)
        time.sleep(BUTTON_DELAY)
        pydirectinput.keyUp(UI_TOGGLE_KEY)
        time.sleep(BUTTON_DELAY)
        return True
    
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

    success = False
    for attempt in range(max_attempts):
        pydirectinput.keyDown(UI_TOGGLE_KEY)
        time.sleep(BUTTON_DELAY)
        pydirectinput.keyUp(UI_TOGGLE_KEY)
        time.sleep(1.2 if attempt == 0 else 1)

        screenshot = get_window_screenshot(window)
        if screenshot is None:
            continue

        scaled_template = scale_to_window(template, screenshot.shape[0])
        res = cv2.matchTemplate(screenshot, scaled_template, cv2.TM_CCOEFF_NORMED)
        _, confidence, _, _ = cv2.minMaxLoc(res)

        if DEBUG:
            log.debug(f"UI box detection attempt {attempt + 1}: confidence={confidence:.2f}")

        if confidence >= 0.6:
            success = True
            break

    if not success:
        log.warning("UI nav box not detected after all attempts — sending one last UI toggle just in case.")
        pydirectinput.keyDown(UI_TOGGLE_KEY)
        time.sleep(BUTTON_DELAY)
        pydirectinput.keyUp(UI_TOGGLE_KEY)
        time.sleep(BUTTON_DELAY)

    return True

def navigate_to(position_index):
    try:
        if DEBUG:
            log.debug(f"Attempting to navigate to position {position_index}")
        
        toggle_ui_and_confirm()
        
        pydirectinput.keyDown('left')
        time.sleep(BUTTON_DELAY)
        pydirectinput.keyUp('left')
        time.sleep(BUTTON_DELAY)
        
        pydirectinput.keyDown('down')
        time.sleep(BUTTON_DELAY)
        pydirectinput.keyUp('down')
        time.sleep(BUTTON_DELAY)
        
        for _ in range(position_index):
            pydirectinput.keyDown('right')
            time.sleep(BUTTON_DELAY)
            pydirectinput.keyUp('right')
            time.sleep(BUTTON_DELAY)
        
        pydirectinput.keyDown('enter')
        time.sleep(BUTTON_DELAY)
        pydirectinput.keyUp('enter')
        time.sleep(BUTTON_DELAY)
        
        pydirectinput.keyDown(UI_TOGGLE_KEY)
        time.sleep(BUTTON_DELAY)
        pydirectinput.keyUp(UI_TOGGLE_KEY)
        time.sleep(BUTTON_DELAY)
        
        return True
        
    except Exception as e:
        log.error(f"Navigation error: {str(e)}")
        return False

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

    roi = cv2.equalizeHist(roi)
    roi = cv2.GaussianBlur(roi, (3, 3), 0)
    _, roi = cv2.threshold(roi, 150, 255, cv2.THRESH_BINARY)

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

    threshold = 0.65
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

    if DEBUG:
        digits_str = "".join([str(d) for (_, d, _) in filtered])
        log.debug(f"Matched digits: {digits_str}")

    try:
        wave = int("".join(digits))       
        return wave
    
    except ValueError:
        return None

def record_upgrade_purchase(upgrade_name, rarity):
    global upgrades_purchased
    upgrades_purchased[upgrade_name.replace('.png', '')][rarity] += 1
    if DEBUG:
        log.debug(f"Recorded purchase: {upgrade_name} ({rarity})")

def red_bar_present():
    try:
        window = get_roblox_window()
        if not window:
            return False

        screenshot = get_window_screenshot(window)
        if screenshot is None:
            return False

        h, w = screenshot.shape[:2]
        x_start = int(w * 0.45)
        y_start = int(h * 0.11)
        x_end = int(w * 0.55)
        y_end = int(h * 0.1)

        x_start = max(0, min(w - 1, x_start))
        x_end   = max(x_start + 1, min(w, x_end))
        y_start = max(0, min(h - 1, y_start))
        y_end   = max(y_start + 1, min(h, y_end))

        bar_region = screenshot[y_start:y_end, x_start:x_end]
        avg_color = np.mean(bar_region, axis=(0, 1))  # BGR
        b, g, r = avg_color.astype(int)

        if DEBUG:
            log.debug(f"[ULT] Avg RGB: R={r}, G={g}, B={b}")

        return r > 150 and r > g + 50 and r > b + 50

    except Exception as e:
        log.error(f"[ULT] Red bar detection error: {str(e)}")
        return False

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
        
        template = cv2.imread(get_template_path(VICTORY_TEMPLATE))
        if template is None:
            return False
        
        template = scale_to_window(template, screenshot.shape[0])
        res = cv2.matchTemplate(screenshot, template, cv2.TM_CCOEFF_NORMED)
            
        _, confidence, _, _ = cv2.minMaxLoc(res)
        
        time.sleep(1) # Wait for results to load
        
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

def perform_cap_upgrade():
    log.info("Performing cap upgrade")

    try:
        toggle_ui_and_confirm()
        pydirectinput.press('down', presses=3, interval=BUTTON_DELAY)
        pydirectinput.press('enter')
        time.sleep(BUTTON_DELAY)
        pydirectinput.press('up', presses=3, interval=BUTTON_DELAY)
        pydirectinput.press('left', presses=2, interval=BUTTON_DELAY)
        pydirectinput.press('down')
        time.sleep(BUTTON_DELAY)
        pydirectinput.press('right')
        time.sleep(BUTTON_DELAY)
        pydirectinput.press('up')
        time.sleep(BUTTON_DELAY)
        pydirectinput.press('enter')
        time.sleep(BUTTON_DELAY)
        pydirectinput.press(UI_TOGGLE_KEY)
        time.sleep(BUTTON_DELAY)
        pydirectinput.press(UI_TOGGLE_KEY)
        time.sleep(BUTTON_DELAY)
        pydirectinput.press('down', presses=3, interval=BUTTON_DELAY)
        pydirectinput.press('enter')
        time.sleep(BUTTON_DELAY)
        pydirectinput.press(UI_TOGGLE_KEY)
        time.sleep(BUTTON_DELAY)
    except Exception as e: 
        log.error(f"Cap upgrade error: {str(e)}")

def change_team_to_1():
    log.info("Changing to Team 1")

    def _do_team_change():
        toggle_ui_and_confirm()
        pydirectinput.press('left', presses=2, interval=BUTTON_DELAY)
        time.sleep(BUTTON_DELAY)
        pydirectinput.press('down', presses=2, interval=BUTTON_DELAY)
        pydirectinput.press('right')
        time.sleep(BUTTON_DELAY)
        pydirectinput.press('enter')
        time.sleep(BUTTON_DELAY)
        pydirectinput.press(UI_TOGGLE_KEY)
        time.sleep(BUTTON_DELAY)

    try:
        _do_team_change()
        if MULTI_INSTANCE:
            alt_tab()
            time.sleep(0.5)
            _do_team_change()
            alt_tab()
    except Exception as e:
        log.error(f"Team change error: {str(e)}")

def detect_yellow_border_region(
    debug=False,
    y_start_ratio=0.85,
    y_end_ratio=0.97,
    x_start_ratio=0.3,
    x_end_ratio=0.7
):
    window = get_roblox_window()
    if not window:
        return None

    screenshot = get_window_screenshot(window)
    if screenshot is None:
        return None

    h, w = screenshot.shape[:2]

    # Calculate scan region
    x_start = int(w * x_start_ratio)
    x_end   = int(w * x_end_ratio)
    y_start = int(h * y_start_ratio)
    y_end   = int(h * y_end_ratio)

    # Extract ROI
    roi = screenshot[y_start:y_end, x_start:x_end]

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    lower_yellow = np.array([25, 200, 200])
    upper_yellow = np.array([35, 255, 255])
    mask = cv2.inRange(hsv, lower_yellow, upper_yellow)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if debug:
        debug_img = screenshot.copy()

        # Draw the scanning region in cyan
        cv2.rectangle(debug_img, (x_start, y_start), (x_end, y_end), (255, 255, 0), 2)

        # Draw detected contours in yellow (offset to full screen coords)
        for cnt in contours:
            cx, cy, cw, ch = cv2.boundingRect(cnt)
            abs_x = cx + x_start
            abs_y = cy + y_start
            cv2.rectangle(debug_img, (abs_x, abs_y), (abs_x + cw, abs_y + ch), (0, 255, 255), 2)

        cv2.imwrite("debug_yellow_scan.png", debug_img)
        log.info("[DEBUG] Saved yellow scan visualization to debug_yellow_scan.png")

    if contours:
        largest = max(contours, key=cv2.contourArea)
        cx, cy, cw, ch = cv2.boundingRect(largest)
        return (cx + x_start, cy + y_start, cw, ch)

    return None
        
def detect_ui_elements_and_respond():
    try:
        window = get_roblox_window()
        if not window:
            return

        screenshot = get_window_screenshot(window)
        if screenshot is None:
            return

        settings_conf, _ = match_template_in_window(screenshot, "settings.png")
        settings_detected = settings_conf >= CONFIDENCE_THRESHOLD

        lobby_conf, _ = match_template_in_window(screenshot, "lobby.png")
        lobby_detected = lobby_conf >= CONFIDENCE_THRESHOLD

        if settings_detected:
            log.info("Detected settings UI. Toggling UI.")
            toggle_ui_and_confirm()
            time.sleep(BUTTON_DELAY)
            for _ in range(3):
                pydirectinput.press('left')
                time.sleep(BUTTON_DELAY)
            
            pydirectinput.press('up')
            time.sleep(BUTTON_DELAY)
            pydirectinput.press('enter')
            time.sleep(BUTTON_DELAY)
            pydirectinput.press(UI_TOGGLE_KEY)
            time.sleep(BUTTON_DELAY)

        if lobby_detected:
            log.info("Detected lobby UI. Attempting to escape.")
            toggle_ui_and_confirm()
            time.sleep(BUTTON_DELAY)
            for _ in range(3):
                pydirectinput.press('left')
                time.sleep(BUTTON_DELAY)
            
            pydirectinput.press('down')
            time.sleep(BUTTON_DELAY)
            pydirectinput.press('down')
            time.sleep(BUTTON_DELAY)
            pydirectinput.press('enter')
            time.sleep(BUTTON_DELAY)
            
            time.sleep(1)
            screenshot = get_window_screenshot(window)
            if screenshot is not None:
                lobby_conf, _ = match_template_in_window(screenshot, "lobby.png")
                if lobby_conf >= CONFIDENCE_THRESHOLD:
                    pydirectinput.press('right')
                    time.sleep(BUTTON_DELAY)
                    pydirectinput.press('enter')
                    time.sleep(BUTTON_DELAY)
            pydirectinput.press(UI_TOGGLE_KEY)
            time.sleep(BUTTON_DELAY)
        
    except Exception as e:
        log.error(f"UI detection error: {str(e)}")

def restart_run():
    global run_start_time, victory_detected, upgrades_purchased
    global cap_upgrades_done, upgrades_since_last_cap

    log.debug("Resetting run")
    try:
        upgrades_since_last_cap = 0
        cap_upgrades_done = 0
        run_start_time = time.time()
        victory_detected = False
        upgrades_purchased = defaultdict(lambda: defaultdict(int))
        return True
    except Exception as e:
        log.error(f"Run restart error: {str(e)}")
        return False

def alt_tab():
    pydirectinput.keyDown('alt')
    time.sleep(0.1)
    pydirectinput.press('tab')
    time.sleep(0.1)
    pydirectinput.keyUp('alt')
    time.sleep(0.5)  # Allow window switch time

def main_loop():
    global run_start_time, victory_detected, is_running, is_paused, upgrade_allowed, last_wave_number, main_card_upgrades, alt_card_upgrades, main_cap_upgrades_done, alt_cap_upgrades_done, cap_upgrades_done 
    
    log.info("=== AFK Endless Macro ===")
    log.info(f"Press {START_KEY} to begin." if not AUTO_START else "Auto-start: Enabled.")
    log.info(f"Press {PAUSE_KEY} to pause/resume | {STOP_KEY} to stop")
    log.info(f"Money Mode: {'ENABLED' if MONEY_MODE else 'Disabled'}")
    log.info(f"High priority upgrades: {', '.join([upgrade.replace('.png', '') for upgrade in HIGH_PRIORITY])}")
    log.info(f"Low priority upgrades: {', '.join([upgrade.replace('.png', '') for upgrade in LOW_PRIORITY])}")
    
    last_scan = time.time()
    last_victory_check = time.time()
    run_start_time = time.time()
    last_team_change = time.time()
    victory_detected = False
    last_wave_number = None
    
    key_thread = KeyListenerThread()
    key_thread.start()
    
    ult_clicker = None
    if ENABLE_ULT_CLICKER:
        ult_clicker = UltClickerThread()
        ult_clicker.start()
        log.info("[ULT-CLICK] Ult clicker thread started")
    else:
        log.info("[ULT-CLICK] Ult clicker thread disabled via config")
    
    if AUTO_START:
        is_running = True
    
    afk_thread = None
    if AFK_PREVENTION:
        afk_thread = AFKPreventionThread()
        afk_thread.start()
    
    try:
        while True:
            if is_running and not is_paused:
                prevent_sleep()   
                current_time = time.time()
                
                if current_time - last_team_change >= TEAM_CHANGE_INTERVAL * 60:
                    log.info("[TeamChange] Interval reached — changing team")
                    change_team_to_1()
                    last_team_change = current_time
                
                if current_time - last_victory_check > SCAN_INTERVAL:
                    if detect_victory():
                        if victory_detected:
                            restart_run()
                    detect_ui_elements_and_respond()
                    last_victory_check = current_time
                    
                if current_time - last_scan > SCAN_INTERVAL:
                    window = get_roblox_window()
                    if window:
                        wave = get_current_wave_number(last_wave=last_wave_number)
                        last_wave_number = wave if wave is not None else last_wave_number

                        upgrades = scan_for_upgrades()
                        if upgrades:
                            if wave is not None:
                                if wave > WAVE_THRESHOLD and wave % 10 not in [1, 6]:
                                    if DEBUG:
                                        log.debug(f"Skipping upgrades at wave {wave} (post-threshold non-priority wave)")
                                    last_scan = time.time()
                                    continue

                            if focus_roblox_window():
                                main_upgrades = scan_for_upgrades()
                                if main_upgrades and select_best_upgrade(main_upgrades):
                                    main_card_upgrades += 1
                                    log.info("[Main] Upgrade successful")

                                    if (
                                        main_card_upgrades >= CAPACITY_UPGRADES_DELAY and
                                        main_cap_upgrades_done < MAX_TOTAL_CAP_UPGRADES
                                    ):
                                        perform_cap_upgrade()
                                        main_cap_upgrades_done += 1
                                        log.info(f"[Main] Cap upgrades: {main_cap_upgrades_done}/{MAX_TOTAL_CAP_UPGRADES}")

                                if MULTI_INSTANCE:
                                    alt_tab()
                                    time.sleep(0.5)

                                    alt_upgrades = scan_for_upgrades()
                                    if alt_upgrades and select_best_upgrade(alt_upgrades):
                                        alt_card_upgrades += 1
                                        log.info("[Alt] Upgrade successful")

                                        if (
                                            alt_card_upgrades >= CAPACITY_UPGRADES_DELAY and
                                            alt_cap_upgrades_done < MAX_TOTAL_CAP_UPGRADES
                                        ):
                                            perform_cap_upgrade()
                                            alt_cap_upgrades_done += 1
                                            log.info(f"[Alt] Cap upgrades: {alt_cap_upgrades_done}/{MAX_TOTAL_CAP_UPGRADES}")

                                    alt_tab()
                            else:
                                log.debug("Could not focus window for upgrade.")
                            last_scan = time.time()
                        else:
                            if DEBUG:
                                log.debug("No upgrades found.")
                            last_scan = time.time() + 2
                    else:
                        last_scan = time.time() + 1
                        
            if int(time.time()) % 30 == 0:
                gc.collect()
                
            else:
                allow_sleep()  # Allow system sleep when paused/stopped
                time.sleep(0.1)  # Reduce CPU usage
                
    except KeyboardInterrupt:
        log.info("Shutting down...")
    
    finally:
        key_thread.running = False
        key_thread.join(timeout=2)
        
        if afk_thread:
            afk_thread.stop()
            afk_thread.join(timeout=2)
            
        if ult_clicker:
            ult_clicker.stop()
            ult_clicker.join(timeout=2)
        
        allow_sleep()

if __name__ == "__main__":
    main_loop()
