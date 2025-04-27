import time
import pydirectinput
import keyboard
import cv2
import numpy as np
import pygetwindow as gw
import os
import json
import platform
import ctypes
import logging
import mss
import psutil
import ctypes
from logging.handlers import RotatingFileHandler
from collections import defaultdict

DEFAULT_CONFIG = {
    "ultrawide_mode": False,
    "scan_interval": 5,
    "start_key": "F6",
    "pause_key": "F7",
    "stop_key": "F8",
    "auto_start": False,
    "money_mode": False,
    "afk_prevention": True,
    "ui_detection": True,
    "button_delay": 0.2,
    "high_priority": ["regen", "boss", "discount", "atk", "health", "units"],
    "low_priority": ["luck", "speed", "heal", "jade", "enemy"],
    "debug": False,
}

CONFIG_PATH = "config.json"

def load_config():
    if not os.path.exists(CONFIG_PATH):
        with open(CONFIG_PATH, "w") as f:
            json.dump(DEFAULT_CONFIG, f, indent=4)
        print(f"[INFO] Created default config at {CONFIG_PATH}")

    with open(CONFIG_PATH, "r") as f:
        return json.load(f)

def add_png_suffix(items):
    return [f"{item}.png" for item in items]

config = load_config()
ULTRAWIDE_MODE = config.get("ultrawide_mode", False)
AUTO_START = config.get("auto_start", False)
SCAN_INTERVAL = config.get("scan_interval", 5)
BUTTON_DELAY = config.get("button_delay", 0.2)
START_KEY = config.get("start_key", "f6")
PAUSE_KEY = config.get("pause_key", "f7")
STOP_KEY = config.get("stop_key", "f8")
DEBUG = config.get("debug", False)
HIGH_PRIORITY = add_png_suffix(config.get("high_priority", DEFAULT_CONFIG["high_priority"]))
LOW_PRIORITY = add_png_suffix(config.get("low_priority", DEFAULT_CONFIG["low_priority"]))
UPGRADE_PRIORITY = HIGH_PRIORITY + LOW_PRIORITY
MONEY_MODE = config.get("money_mode", False)
AFK_PREVENTION = config.get("afk_prevention", True)

IMAGE_FOLDER = "ultrawide" if ULTRAWIDE_MODE else "images"
CONFIDENCE_THRESHOLD = 0.7
UI_TOGGLE_KEY = "\\"
DEBOUNCE_TIME = 0.3
KEY_HOLD_TIME = 0.3

# State variables
is_running = False
is_paused = False
upgrades_purchased = defaultdict(lambda: defaultdict(int))
last_key_press_time = defaultdict(float)
key_hold_state = defaultdict(bool)
last_afk_action = 0
afk_interval = 600

RARITY_COLORS = {
    (71, 99, 189): "Common",
    (190, 60, 238): "Epic",
    (238, 208, 60): "Legendary",
    (238, 60, 60): "Mythic",
}

RARITY_ORDER = ["Unknown", "Common", "Epic", "Legendary", "Mythic"]

# Windows sleep prevention
if platform.system() == "Windows":
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
    def prevent_sleep(): pass
    def allow_sleep(): pass

def setup_logging():
    with open('afm_macro.log', 'w'):
        pass
    
    logger = logging.getLogger("AFM")
    logger.setLevel(logging.DEBUG)
    
    file_handler = RotatingFileHandler("afm_macro.log", encoding="utf-8")
    file_handler.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s: %(message)s", datefmt="%H:%M:%S"))
    
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter("[%(asctime)s] %(message)s", datefmt="%H:%M:%S"))
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    return logger

log = setup_logging()

def get_pid_from_hwnd(hwnd):
    pid = ctypes.c_ulong()
    ctypes.windll.user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
    return pid.value

def get_roblox_window():
    try:
        candidates = gw.getWindowsWithTitle("Roblox")
        best_window = None
        largest_area = 0

        for window in candidates:
            if window.isMinimized:
                continue

            try:
                pid = get_pid_from_hwnd(window._hWnd)
                exe_name = psutil.Process(pid).name().lower()
            except Exception as e:
                if DEBUG:
                    log.debug(f"Failed to get process for window: {e}")
                continue

            if exe_name != "robloxplayerbeta.exe":
                continue

            area = window.width * window.height
            if area > largest_area:
                largest_area = area
                best_window = window

        if not best_window and DEBUG:
            log.warning("No valid Roblox GAME window found.")

        return best_window

    except Exception as e:
        if DEBUG:
            log.error(f"Window detection error: {str(e)}")
        return None

def focus_roblox_window():
    try:
        window = get_roblox_window()
        if window:
            if not window.isActive:
                window.activate()
                time.sleep(0.2)
            return window
        if DEBUG:
            log.warning("Roblox window not found")
    except Exception as e:
        if DEBUG:
            log.error(f"Window focus error: {e}")
    return None

def get_window_screenshot(window):
    try:
        with mss.mss() as sct:
            monitor = {"left": window.left, "top": window.top, 
                      "width": window.width, "height": window.height}
            return cv2.cvtColor(np.array(sct.grab(monitor)), cv2.COLOR_BGRA2BGR)
    except Exception as e:
        log.error(f"Screenshot failed: {str(e)}")
        return None
    
def template_match(screenshot, template_path):
    template = cv2.imread(template_path)
    if template is None:
        log.error(f"Template not found: {template_path}")
        return None
    
    res = cv2.matchTemplate(screenshot, template, cv2.TM_CCOEFF_NORMED)
    _, confidence, _, _ = cv2.minMaxLoc(res)
    return confidence >= CONFIDENCE_THRESHOLD

def get_template_path(filename):
    return os.path.join(IMAGE_FOLDER, filename)

def is_percent_upgrade(card_img, position):
    try:
        template1_path = get_template_path("percent.png")
        if os.path.exists(template1_path):
            template1 = cv2.imread(template1_path)
            if template1 is not None:
                res1 = cv2.matchTemplate(card_img, template1, cv2.TM_CCOEFF_NORMED)
                _, confidence1, _, _ = cv2.minMaxLoc(res1)
                if confidence1 > 0.7:
                    if DEBUG:
                        log.debug(f"Percent detected (template1) at position {position} with confidence {confidence1:.2f}")
                    return True

        template2_path = get_template_path("percent2.png")
        if os.path.exists(template2_path):
            template2 = cv2.imread(template2_path)
            if template2 is not None:
                res2 = cv2.matchTemplate(card_img, template2, cv2.TM_CCOEFF_NORMED)
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

def scan_for_upgrades():
    window = get_roblox_window()
    if not window:
        return []
    
    screenshot = get_window_screenshot(window)
    if screenshot is None:
        return []

    card_width = int(window.width * 0.18)
    gap = int(window.width * 0.03)
    first_card_left = (window.width // 2) - int((card_width + gap) * 1.45)

    found_upgrades = []

    for position in range(3):
        x = first_card_left + position * (card_width + gap)
        card_img = screenshot[0:window.height, x:x+card_width]

        for upgrade_filename in UPGRADE_PRIORITY:
            template_path = os.path.join(IMAGE_FOLDER, upgrade_filename)
            template = cv2.imread(template_path)
            if template is None:
                continue

            res = cv2.matchTemplate(card_img, template, cv2.TM_CCOEFF_NORMED)
            _, confidence, _, _ = cv2.minMaxLoc(res)

            if confidence < CONFIDENCE_THRESHOLD:
                continue

            rarity = get_rarity_from_color(card_img)
            upgrade_name = upgrade_filename.replace(".png", "")

            if "unit" in upgrade_name and rarity == "Common":
                if DEBUG:
                    log.debug(f"Skipping common unit upgrade at pos {position}")
                continue

            if upgrade_filename in ['atk.png', 'health.png'] and not is_percent_upgrade(card_img, position):
                if DEBUG:
                    log.debug(f"Skipping flat {upgrade_name} upgrade at pos {position}")
                continue

            found_upgrades.append({
                "upgrade": upgrade_name,
                "original_upgrade": upgrade_filename,
                "position": position,
                "rarity": rarity,
                "confidence": confidence
            })
            break

    if MONEY_MODE:
        boss_upgrades = [u for u in found_upgrades if u["original_upgrade"] == "boss.png"]
        other_upgrades = [u for u in found_upgrades if u["original_upgrade"] != "boss.png"]

        boss_upgrades.sort(key=lambda u: -RARITY_ORDER.index(u["rarity"]))
        other_upgrades.sort(key=lambda u: (
            0 if u["original_upgrade"] in HIGH_PRIORITY else 1,
            -RARITY_ORDER.index(u["rarity"]),
            UPGRADE_PRIORITY.index(u["original_upgrade"])
        ))

        return boss_upgrades + other_upgrades

    return sorted(found_upgrades, key=lambda u: (
        0 if u["original_upgrade"] in HIGH_PRIORITY else 1,
        -RARITY_ORDER.index(u["rarity"]),
        UPGRADE_PRIORITY.index(u["original_upgrade"])
    ))

def get_rarity_from_color(card_img):
    try:
        height, width = card_img.shape[:2]
        sample_region = card_img[int(height*0.56):int(height*0.57), int(width*0.9):int(width*0.95)]
        avg_color = np.mean(sample_region, axis=(0, 1)).astype(int)
        b, g, r = avg_color[0], avg_color[1], avg_color[2]
        
        closest = min(RARITY_COLORS.items(),
                     key=lambda c: sum((c[0][i] - rgb) ** 2 for i, rgb in enumerate((r, g, b))))
        return closest[1] if sum((closest[0][i] - rgb) ** 2 for i, rgb in enumerate((r, g, b))) < 2500 else "Unknown"
    except Exception as e:
        return "Unknown"

def navigate_to(position_index):
    try:
        toggle_ui_and_confirm(get_roblox_window())
        time.sleep(0.5)
        
        pydirectinput.press('left')
        pydirectinput.press('down')
        time.sleep(BUTTON_DELAY)
        
        for _ in range(position_index):
            pydirectinput.press('right')
            time.sleep(BUTTON_DELAY)
            
        pydirectinput.press('enter')
        time.sleep(0.5)
        pydirectinput.press(UI_TOGGLE_KEY)
        return True
    except Exception as e:
        log.error(f"Navigation error: {str(e)}")
        return False

def prevent_afk():
    global last_afk_action
    current_time = time.time()
    
    if current_time - last_afk_action > afk_interval:
        if is_running and not is_paused:
            log.debug("[AFK] Simulating jump")
            pydirectinput.press('space')
            last_afk_action = current_time

def toggle_ui_and_confirm(window=None):
    if not window:
        window = get_roblox_window()
        if not window:
            log.warning("No Roblox window found for UI confirmation.")
            return False

    template_path = get_template_path("navbox.png")
    template = cv2.imread(template_path)
    if template is None:
        log.error("Failed to load navbox.png")
        return False

    while True:
        pydirectinput.press(UI_TOGGLE_KEY)
        time.sleep(1)  # Give UI time to react

        screenshot = get_window_screenshot(window)
        if screenshot is None:
            continue

        res = cv2.matchTemplate(screenshot, template, cv2.TM_CCOEFF_NORMED)
        _, confidence, _, _ = cv2.minMaxLoc(res)

        if DEBUG:
            log.debug(f"UI box detection: confidence={confidence:.2f}")

        if confidence >= 0.6:
            return True
        else:
            time.sleep(0.5)

def detect_ui_elements_and_respond():
    try:
        window = get_roblox_window()
        focus_roblox_window()
        if not window:
            return

        screenshot = get_window_screenshot(window)
        if screenshot is None:
            return

        settings_detected = template_match(screenshot, get_template_path("settings.png"))
        lobby_detected = template_match(screenshot, get_template_path("lobby.png"))

        def press_keys(keys):
            for key in keys:
                pydirectinput.press(key)
                time.sleep(BUTTON_DELAY)

        if settings_detected:
            log.info("Detected settings UI.")
            toggle_ui_and_confirm(get_roblox_window())
            press_keys(['left'] * 2 + ['up', 'enter'])

            time.sleep(1)
            if not toggle_ui_and_confirm(window):
                log.warning("UI not confirmed after closing settings. Retrying toggle...")
                time.sleep(BUTTON_DELAY)

        if lobby_detected:
            log.info("Detected lobby UI.")
            toggle_ui_and_confirm(get_roblox_window())
            press_keys(['up'] * 5 + ['enter'])

            time.sleep(1)
            screenshot = get_window_screenshot(window)
            if template_match(screenshot, get_template_path("lobby.png")):
                press_keys(['down', 'enter'])

    except Exception as e:
        log.error(f"UI detection error: {e}")

def main_loop():
    global is_running, is_paused, afk_interval
    log.info("=== AFK Macro ===")
    log.info(f"Money Mode: {'ACTIVE' if MONEY_MODE else 'Inactive'}")
    log.info(f"AFK Prevention: {'ACTIVE' if AFK_PREVENTION else 'Inactive'}")
    log.info(f"Press {START_KEY} to start | {PAUSE_KEY} to pause | {STOP_KEY} to stop")
    
    last_scan = 0
    while True:
        if keyboard.is_pressed(STOP_KEY):
            is_running = False
            allow_sleep()
            log.info("=== Stopped ===")
            time.sleep(1)
            break
            
        if keyboard.is_pressed(START_KEY):
            if not is_running:
                is_running = True
                prevent_sleep()
                log.info("=== Started ===")
            time.sleep(0.5)
            
        if keyboard.is_pressed(PAUSE_KEY):
            is_paused = not is_paused
            log.info(f"=== {'Paused' if is_paused else 'Resumed'} ===")
            time.sleep(0.5)
            
        if is_running and not is_paused:
            current_time = time.time()
            
            if AFK_PREVENTION:
                prevent_afk()
            
            if current_time - last_scan > SCAN_INTERVAL:
                window = get_roblox_window()
                if window:
                    detect_ui_elements_and_respond()
                    upgrades = scan_for_upgrades()
                    if upgrades:
                        focus_roblox_window()
                        if navigate_to(upgrades[0]['position']):
                            log.info(f"Selected {upgrades[0]['upgrade']} ({upgrades[0]['rarity']})")
                            upgrades_purchased[upgrades[0]['upgrade']][upgrades[0]['rarity']] += 1
                    last_scan = current_time
                else:
                    last_scan = current_time + 1

if __name__ == "__main__":
    main_loop()