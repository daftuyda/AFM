# AFM
 Anime Fantasy Macro

## What is this
 A automatic endless macro for the Roblox game Anime Fantasy Kingdom.

## Usage
 Edit the config.json file the script creates on first load with your desired scan interval, stop key and Discord webhook (optional).
 Make sure UI Navigation Toggle is On in settings.
 Go into an Endless game and press F6 to start the macro.

## Keybinds
| Key | Function     |
|-----|--------------|
| F6  | Start        |
| F7  | Pause/Resume |
| F8  | Stop         |

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

## Limitations
 This is currently only tested on 1080p and will probably mess up on anything else.
 Which also includes the window not being in fullscreen.
 The Roblox window also needs to be on the primary display.

# Upgrade Logic
 (REGEN > BOSS > DISCOUNT > ATK% > HEALTH% > SUMMON) > LUCK% > SPEED% > HEALING% > JADE
 It will always pick the higher rarity choice of the upgrades in the brackets.
 
 Skips:
 - 3 Epic Units
 - Flat Health Increases
 - Flat ATK Increases

 If everything is a skipped upgrade or it can't find upgrades it will default to the 1st option.
