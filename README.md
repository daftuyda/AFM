# [AFM - Latest Release](https://github.com/daftuyda/AFM/releases/latest/download/AFM.zip)
 [![](https://dcbadge.limes.pink/api/server/https://discord.gg/BzVmGXQ22e)](https://discord.gg/BzVmGXQ22e)
 Join the support server to stay up to date and to report any issues you have.
 
 [![Build Workflow](https://github.com/daftuyda/AFM/actions/workflows/build.yml/badge.svg)](https://github.com/daftuyda/AFM/actions/workflows/build.yml)

## What is this
 A automatic endless macro for the Roblox game Anime Fantasy Kingdom.

## Is this safe?
 There is currently no word from the staff to say macros are banned, so until further notice this will stay up. The exe itself that is bundled with the release is also safe. Everything is built by Github actions so there can be no tampering with the package. All source code is avaliable to check here and the only web requests it makes is when sending a Discord webhook.

## Usage
 Edit the config.json file the script creates on first load with your desired scan interval, stop key and Discord webhook (optional).
 Make sure UI Navigation Toggle is On in settings.
 Go into an Endless game and press `start_key` to start the macro.

 ### If manual mode is enabled
 Press the `start_key` to scan for upgrades.
 This mode will not automatically scan for upgrades but will scan for the victory screen.

## Default Keybinds
| Key | Function     |
|-----|--------------|
| F6  | Start        |
| F7  | Pause/Resume |
| F8  | Stop         |

## Config

| Config          | Function                                                            |
|-----------------|---------------------------------------------------------------------|
| ultrawide_mode  | Enables support for ultra-wide (21:9) displays. (True/False)        |
| maximize_window | Automatically maximizes window for better detection. (True/False)   |
| auto_reconnect  | Will start a solo run if disconnected. (True/False)                 |
| scan_interval   | How often the window is scanned for upgrades. (Number in seconds)   |
| start_key       | Key that controls the script starting or controls manual mode.      |
| pause_key       | Key that controls the script pausing/resuming.                      |
| stop_key        | Key that stops and quits the script.                                |
| auto_start      | Enables the script automatically starting when run. (True/False)    |
| webhook_url     | Discord webhook URL (Optional)                                      |
| mode            | Controls what mode the script runs in (auto/manual)                 |
| money_mode      | Prioritizes reward increase upgrades. (True/False)                  |
| wave_threshold  | Will hold upgrades for each boss wave past the set wave (Number)    |
| button_delay    | Delay of the navigation speed. (Number in seconds)                  |
| high_priority   | Upgrades in this list will be chosen first.                         |
| low_priority    | Upgrades in list will be chosen if no high priority upgrades found. |
| auto_updates    | Enables dialogue on start up if theres a new update (True/False)    |
| debug           | Enables debug messages to the console (True/False)                  |

## How it works
 This macro will scann the screen every X seconds that is set as the scan interval.
 It will match upgrades and identify them based on their logo and rarity.
 It then uses logic based off of the current endless meta strategy and choose the best upgrade.
 If the victory screen is detected it will take a temporary screenshot and upload it to the Discord webhook.
 Then it will reset and start a new endless run automatically.

### TL;DR
 It uses CV to scan for upgrades.
 Calculates logic on the best upgrade.
 Then navigates to them using roblox in-game navigation

# Upgrade Logic
 (REGEN > BOSS > DISCOUNT > ATK% > HEALTH% > SUMMON) > LUCK% > SPEED% > HEALING% > JADE

 It will always pick the higher rarity choice of the upgrades in the brackets.
 
 Skips:
 - 3 Epic Units
 - Flat Health Increases
 - Flat ATK Increases

 If everything is a skipped upgrade or it can't find upgrades it will default to the 1st option.
