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
from io import BytesIO
from datetime import datetime
from collections import defaultdict

# Configuration
UPGRADE_PRIORITY = [
    "regen.png",
    "boss.png",
    "discount.png",
    "atk.png",
    "health.png",
    "units.png",
    "luck.png",
    "speed.png",
    "heal.png",
    "jade.png",
    "enemy.png"
]

RARITY_COLORS = {
    (71, 99, 189): "Common",     # Blue
    (190, 60, 238): "Epic",      # Purple
    (238, 208, 60): "Legendary", # Yellow
    (238, 60, 60): "Mythic"      # Red
}

RARITY_ORDER = ["Unknown", "Common", "Epic", "Legendary", "Mythic"]

VICTORY_TEMPLATE = "victory.png"

DEFAULT_CONFIG = {
    "scan_interval": 5,
    "stop_key": "f8",
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

config = load_config()
SCAN_INTERVAL = config.get("scan_interval", 5)
STOP_KEY = config.get("stop_key", "right shift")
DISCORD_WEBHOOK_URL = config.get("webhook_url", "")
DEBUG = config.get("debug", False)

CONFIDENCE_THRESHOLD = 0.8
UI_TOGGLE_KEY = '\\'

# State variables
last_victory_time = 0
run_start_time = 0
victory_detected = False
upgrades_purchased = defaultdict(lambda: defaultdict(int))

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

def log(message):
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"[{timestamp}] {message}")

def focus_roblox_window():
    try:
        roblox_windows = gw.getWindowsWithTitle("Roblox")
        if roblox_windows:
            window = roblox_windows[0]
            if not window.isActive:
                window.activate()
                time.sleep(0.2)
            return window
        log("Roblox window not found")
    except Exception as e:
        log(f"Window focus error: {str(e)}")
    return None

def get_rarity_from_color(image, position):
    try:
        height, width = image.shape[:2]
        
        if position == 0:
            center_x = width * 3 // 4
        elif position == 2:
            center_x = width // 4
        else:
            center_x = width // 2
        
        y_base = 610
        sample_width = 10
        
        sample_x = max(0, center_x - sample_width//2)
        sample_region = image[y_base-2:y_base+3, sample_x:sample_x+sample_width]
        
        avg_color = np.mean(sample_region, axis=(0, 1)).astype(int)
        rgb_color = (avg_color[2], avg_color[1], avg_color[0])
        
        if DEBUG:
            log(f"Pos {position} adjusted center {center_x} color: {rgb_color}")

        closest_match = "Unknown"
        min_distance = float('inf')
        
        for ref_color, rarity in RARITY_COLORS.items():
            distance = sum((c1 - c2) ** 2 for c1, c2 in zip(ref_color, rgb_color)) ** 0.5
            if distance < min_distance:
                min_distance = distance
                closest_match = rarity
                if distance < 30:
                    break
        
        return closest_match if min_distance < 50 else "Unknown"
        
    except Exception as e:
        log(f"Rarity detection error: {str(e)}")
        return "Unknown"

def record_upgrade_purchase(upgrade_name, rarity):
    global upgrades_purchased
    upgrades_purchased[upgrade_name.replace('.png', '')][rarity] += 1
    if DEBUG:
        log(f"Recorded purchase: {upgrade_name} ({rarity})")

def generate_upgrade_summary():
    """Generate a formatted summary of upgrades purchased this run with perfect alignment"""
    if not upgrades_purchased:
        return "No upgrades purchased this run"
    
    # Emoji color blocks for rarities
    RARITY_EMOJIS = {
        "Mythic": "🔴",      # Red
        "Legendary": "🟡",   # Yellow
        "Epic": "🟣",        # Purple
        "Common": "🔵"       # Blue
    }
    
    # Create rarity columns in desired order (highest to lowest)
    rarities = ["Mythic", "Legendary", "Epic", "Common"]
    
    # Prepare table rows
    rows = []
    for upgrade in sorted(upgrades_purchased.keys()):
        row = [upgrade.capitalize()]
        for rarity in rarities:
            row.append(str(upgrades_purchased[upgrade].get(rarity, 0)))
        rows.append(row)
    
    # Calculate totals
    totals = ["TOTAL"]
    for i, rarity in enumerate(rarities):
        total = sum(int(row[i+1]) for row in rows)
        totals.append(str(total))
    rows.append(totals)
    
    # Format as a table with emoji headers
    headers = ["Upgrade"] + [RARITY_EMOJIS[r] for r in rarities]
    
    # Calculate column widths - now properly formatted and readable
    # First column (upgrade names)
    first_col_width = max(len(str(row[0])) for row in rows + [headers])
    
    # Other columns (counts)
    other_col_widths = []
    for i in range(len(rarities)):
        max_num_width = max(len(str(row[i+1])) for row in rows)
        header_width = len(headers[i+1])
        other_col_widths.append(max(max_num_width, header_width))
    
    col_widths = [first_col_width] + other_col_widths
    
    # Build the perfectly aligned table
    header_line = " | ".join(f"{h:^{w}}" for h, w in zip(headers, col_widths))
    separator = "-+-".join("-" * w for w in col_widths)
    body_lines = []
    for row in rows:
        # Left-align upgrade names, center-align counts
        line_parts = [f"{row[0]:<{col_widths[0]}}"]  # Left-align name
        line_parts += [f"{item:^{w}}" for item, w in zip(row[1:], col_widths[1:])]  # Center counts
        body_lines.append(" | ".join(line_parts))
    
    return f"```\n{header_line}\n{separator}\n" + "\n".join(body_lines) + "\n```"

def scan_for_upgrades():
    try:
        window = focus_roblox_window()
        if not window:
            return []

        left = window.left
        top = window.top
        width = window.width
        height = window.height

        # Card dimensions and positioning (adjust these based on your debug images)
        card_width = 350  # Width of each upgrade card
        gap = 50         # Space between cards
        first_card_left = (width // 2) - 575  # Left edge of first card

        # Calculate exact regions for each card
        regions = [
            # Left card
            (left + first_card_left, top, card_width, height),
            # Middle card
            (left + first_card_left + card_width + gap, top, card_width, height),
            # Right card
            (left + first_card_left + 2*(card_width + gap), top, card_width, height)
        ]

        found_upgrades = []

        for position, region in enumerate(regions):
            try:
                # Capture with 5px buffer on each side
                buffered_region = (
                    max(left, region[0] - 5),
                    region[1],
                    min(width, region[2] + 10),
                    region[3]
                )
                
                screenshot = pyautogui.screenshot(region=buffered_region)
                screenshot = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)
                
                # Save debug image
                # cv2.imwrite(f'debug_card_pos{position}.png', screenshot)
                
                for upgrade in UPGRADE_PRIORITY:
                    template = cv2.imread(upgrade)
                    if template is None:
                        continue
                        
                    res = cv2.matchTemplate(screenshot, template, cv2.TM_CCOEFF_NORMED)
                    _, confidence, _, _ = cv2.minMaxLoc(res)

                    if confidence >= CONFIDENCE_THRESHOLD:
                        upgrade_name = upgrade
                        is_percent = False
                        
                        if upgrade in ["health.png", "atk.png"]:
                            is_percent = is_percent_upgrade(screenshot, position)
                            if is_percent:
                                upgrade_name = upgrade.replace('.png', '_percent.png')
                                log(f"Identified percent upgrade: {upgrade_name}")
                        
                        rarity = get_rarity_from_color(screenshot, position)
                        found_upgrades.append({
                            'upgrade': upgrade_name,
                            'position': position,
                            'rarity': rarity,
                            'confidence': confidence,
                            'is_percent': is_percent,
                            'original_upgrade': upgrade
                        })
                        break
                        
            except Exception as e:
                if DEBUG:
                    log(f"Error scanning position {position}: {str(e)}")
                    
        return found_upgrades

    except Exception as e:
        log(f"Scan error: {str(e)}")
        return []

def is_percent_upgrade(screenshot, position):
    """Check if the upgrade has a percent symbol using template matching"""
    try:
        # Load the percent symbol template
        percent_template = cv2.imread('percent.png')
        if percent_template is None:
            log("Warning: percent.png template not found")
            return False

        height, width = screenshot.shape[:2]
        
        # Vertical region (same for all positions)
        y_start = height // 2 + 20  # Start slightly below middle
        y_end = height * 2 // 3 + 40  # End lower down
        
        # Position-specific horizontal regions - now showing full card width
        if position == 0:  # Left upgrade
            x_start = 0              # Start at very left
            x_end = width            # End at very right
        elif position == 2:  # Right upgrade
            x_start = 0              # Start at very left
            x_end = width            # End at very right
        else:  # Middle upgrade
            x_start = 0              # Start at very left
            x_end = width            # End at very right

        search_region = screenshot[y_start:y_end, x_start:x_end]
        
        # Debugging - save images with position info
        # timestamp = int(time.time())
        # cv2.imwrite(f'debug_full_card_pos{position}_{timestamp}.png', search_region)
        
        # Perform template matching on the full card region
        res = cv2.matchTemplate(search_region, percent_template, cv2.TM_CCOEFF_NORMED)
        _, max_confidence, _, max_loc = cv2.minMaxLoc(res)
        
        if DEBUG:
            log(f"Checking FULL CARD at pos {position} (X{x_start}-{x_end} Y{y_start}-{y_end})")
            log(f"Max confidence: {max_confidence:.2f} at position {max_loc}")

        return max_confidence > CONFIDENCE_THRESHOLD
        
    except Exception as e:
        log(f"Percent check error: {str(e)}")
        return False

def navigate_to(position_index):
    try:
        log(f"Navigating to position {position_index}")
        
        pydirectinput.keyDown(UI_TOGGLE_KEY)
        time.sleep(0.2)
        pydirectinput.keyUp(UI_TOGGLE_KEY)
        time.sleep(0.2)
        
        pydirectinput.keyDown('left')
        time.sleep(0.2)
        pydirectinput.keyUp('left')
        time.sleep(0.2)
        
        pydirectinput.keyDown('down')
        time.sleep(0.2)
        pydirectinput.keyUp('down')
        time.sleep(0.2)
        
        for _ in range(position_index):
            pydirectinput.keyDown('right')
            time.sleep(0.2)
            pydirectinput.keyUp('right')
            time.sleep(0.2)
        
        pydirectinput.keyDown('enter')
        time.sleep(0.2)
        pydirectinput.keyUp('enter')
        time.sleep(0.2)
        
        pydirectinput.keyDown(UI_TOGGLE_KEY)
        time.sleep(0.2)
        pydirectinput.keyUp(UI_TOGGLE_KEY)
        
        return True
        
    except Exception as e:
        log(f"Navigation error: {str(e)}")
        return False

def select_best_upgrade(upgrades):
    if not upgrades:
        log("No upgrades detected")
        return False

    # First filter out unwanted upgrades
    valid_upgrades = []
    for upgrade in upgrades:
        # Skip Common unit upgrades
        if "unit" in upgrade['upgrade'].lower() and upgrade['rarity'] == "Common":
            if DEBUG:
                log(f"Skipping Common unit upgrade: {upgrade['upgrade']} at pos {upgrade['position']}")
            continue
            
        # Skip flat ATK/HP upgrades (non-percent)
        if (upgrade['original_upgrade'] in ['atk.png', 'health.png'] and 
            not upgrade.get('is_percent', False)):
            if DEBUG:
                log(f"Skipping flat {upgrade['original_upgrade']} upgrade at pos {upgrade['position']}")
            continue
            
        valid_upgrades.append(upgrade)
    
    # Fail-safe if no valid upgrades
    if not valid_upgrades:
        log("No valid upgrades after filtering - using fail-safe")
        for upgrade in upgrades:
            if upgrade['position'] == 0:
                log(f"Fail-safe: Selecting position 0 ({upgrade['upgrade']})")
                if navigate_to(0):
                    record_upgrade_purchase(upgrade['upgrade'], upgrade['rarity'])
                    return True
        return False

    # New improved sorting logic
    def get_sort_key(upgrade):
        # Priority groups (lower number = higher priority)
        if upgrade['original_upgrade'] in ['discount.png', 'boss.png', 'regen.png']:
            group = 0  # Highest priority
        elif UPGRADE_PRIORITY.index(upgrade['original_upgrade']) < 5:
            group = 1  # High priority
        else:
            group = 2  # Low priority

        # For percent upgrades, boost priority within their group
        percent_boost = 0 if upgrade.get('is_percent', False) else 1

        return (
            group,          # Primary group
            percent_boost,  # Percent gets priority within group
            -RARITY_ORDER.index(upgrade['rarity']),  # Higher rarity first
            UPGRADE_PRIORITY.index(upgrade['original_upgrade'])  # Original priority
        )

    valid_upgrades.sort(key=get_sort_key)

    if DEBUG:
        log("Valid upgrades sorted:")
        for idx, upgrade in enumerate(valid_upgrades, 1):
            perc = " (PERCENT)" if upgrade.get('is_percent', False) else ""
            group = ["TOP", "HIGH", "LOW"][
                0 if upgrade['original_upgrade'] in ['discount.png', 'boss.png', 'regen.png'] else
                1 if UPGRADE_PRIORITY.index(upgrade['original_upgrade']) < 5 else 2
            ]
            log(f"{idx}. [{group}] {upgrade['upgrade']}{perc} ({upgrade['rarity']}) at pos {upgrade['position']}")

    # Try to select best upgrade
    for upgrade in valid_upgrades:
        perc = " (PERCENT)" if upgrade.get('is_percent', False) else ""
        log(f"Attempting: {upgrade['upgrade']}{perc} at position {upgrade['position']}")
        if navigate_to(upgrade['position']):
            record_upgrade_purchase(upgrade['upgrade'], upgrade['rarity'])
            return True

    return False

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
            "content": f"🏆 Victory! Run completed in {time_str}\n\n{summary}",
            "username": "AFM Bot"
        }
        
        response = requests.post(
            DISCORD_WEBHOOK_URL,
            data=payload,
            files=files
        )
        
        if response.status_code == 204:
            log("Screenshot and stats uploaded to Discord successfully")
            return True
        else:
            log(f"Discord upload failed: {response.text}")
            return False
            
    except Exception as e:
        log(f"Discord upload error: {str(e)}")
        return False

def detect_victory():
    global last_victory_time, victory_detected, run_start_time
    
    try:
        current_time = time.time()
        if current_time - last_victory_time < 30:
            return False
            
        window = focus_roblox_window()
        if not window:
            return False
        
        time.sleep(2) # Wait for results to load
            
        screenshot = pyautogui.screenshot(region=(
            window.left, window.top, window.width, window.height
        ))
        screenshot = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)
        
        template = cv2.imread(VICTORY_TEMPLATE)
        if template is None:
            return False
            
        res = cv2.matchTemplate(screenshot, template, cv2.TM_CCOEFF_NORMED)
        _, confidence, _, _ = cv2.minMaxLoc(res)
        
        if confidence > CONFIDENCE_THRESHOLD:
            run_time = 0
            if run_start_time > 0:
                run_time = current_time - run_start_time
                run_start_time = 0
            
            log(f"Victory detected! (confidence: {confidence:.2f})")
            if DISCORD_WEBHOOK_URL:
                log("Uploading screenshot and stats to Discord...")
                if not upload_to_discord(screenshot, run_time):
                    log("Failed to upload to Discord")
            else:
                if DEBUG:
                    log("Discord webhook URL not set, skipping upload")
            
            last_victory_time = current_time
            victory_detected = True
            return True
            
        return False
        
    except Exception as e:
        log(f"Victory detection error: {str(e)}")
        return False

def restart_run():
    global run_start_time, victory_detected, upgrades_purchased
    
    log("Attempting to restart run")
    try:
        time.sleep(0.2)
        pydirectinput.keyDown(UI_TOGGLE_KEY)
        time.sleep(0.1)
        pydirectinput.keyUp(UI_TOGGLE_KEY)
        time.sleep(0.1)
        
        for _ in range(3):
            pydirectinput.keyDown('down')
            time.sleep(0.05)
            pydirectinput.keyUp('down')
            time.sleep(0.05)
            
        pydirectinput.keyDown('enter')
        time.sleep(0.1)
        pydirectinput.keyUp('enter')
        time.sleep(0.2)
        
        pydirectinput.keyDown(UI_TOGGLE_KEY)
        time.sleep(0.2)
        pydirectinput.keyUp(UI_TOGGLE_KEY)
        
        run_start_time = time.time()
        victory_detected = False
        upgrades_purchased = defaultdict(lambda: defaultdict(int))
        return True
        
    except Exception as e:
        log(f"Run restart error: {str(e)}")
        return False

def main_loop():
    global run_start_time, victory_detected
    
    log("=== Roblox Upgrade Bot ===")
    last_scan = time.time()
    last_victory_check = time.time()
    
    run_start_time = time.time()
    victory_detected = False
    
    while True:
        if keyboard.is_pressed(STOP_KEY):
            log("=== Bot stopped by user ===")
            allow_sleep()
            break
            
        prevent_sleep()   
            
        current_time = time.time()
        
        # Victory check
        if current_time - last_victory_check > SCAN_INTERVAL:
            if detect_victory():
                if victory_detected:
                    restart_run()
            last_victory_check = current_time
            
        # Upgrade scanning with improved timing
        if current_time - last_scan > SCAN_INTERVAL:
            if focus_roblox_window():
                time.sleep(1)  # Add delay before scanning
                upgrades = scan_for_upgrades()
                if upgrades:  # Only proceed if upgrades were found
                    select_best_upgrade(upgrades)
                    last_scan = time.time() + 5  # Add extra cooldown after selection
                else:
                    log("No upgrades found, waiting...")
                    last_scan = time.time() + 2  # Shorter wait if nothing found
            else:
                last_scan = time.time() + 2  # Shorter wait if window not focused

if __name__ == "__main__":
    main_loop()