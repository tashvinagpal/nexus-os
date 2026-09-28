# Subsystem 4: App & game discovery and launching

Code: `nexus/system/apps/` (discovery, catalog, launcher), `nexus/system/games/`
(providers, GameService, watcher), `nexus/tools/apps.py` (tools),
`nexus/ai/router.py` (fast path).

## What NEXUS distinguishes

| You say | It is | Tool | Model needed? |
|---|---|---|---|
| "Open Chrome", "chrome khol do", "fire up Chrome" | installed app | `launch_app` -> the discovered chrome.exe | no (fast path) |
| "Launch GTA V", "GTA chala de", "fire up Grand Theft Auto" | game | `launch_app` -> Steam/Epic/Rockstar provider | no |
| "Go to YouTube", "open github.com" | website | `open_url` | no |
| "Open my Downloads folder", "downloads kholo" | folder | `open_folder` (the "explore" verb: can never run a file) | no |
| "cmd", "PowerShell", "Git Bash", Python | command shell / interpreter | refused | no |
| "Registry Editor", services, .msc consoles | admin tool | refused | no |
| "run C:\...\setup.exe", "random.exe", "x.ps1" | a file, not an app | refused | no |
| "Nexus, I'm bored. Open GTA." / unknown names | - | planner (model), then the same tools | yes |

## Discovery (Windows)

Sources: Start menu (all users + you), Desktop shortcuts (not loose .exe files),
App Paths registrations (aliases only), Store apps (read-only `Get-StartApps`
query), Steam (`libraryfolders.vdf` + `appmanifest_*.acf`), Epic
(`ProgramData/Epic/.../Manifests/*.item`), Rockstar (registry, launcher-owned
titles only). Paths come from Windows' known-folder API and the launchers' own
files - nothing is hard-coded, so other drives work.

`.lnk` files are parsed as data ([MS-SHLLINK]); nothing is resolved through the
shell. On the dev PC 146/150 shortcuts read identically to Windows' own reader;
the other 4 (Riot) store their target only as a shell item and are started via
the shortcut file itself.

Each program becomes one `AppRecord` / `GameRecord` (id, display name, aliases,
kind, source, launch method + target, arguments, publisher, install dir,
running-process identifiers, `mediated`, `requires_elevation`, confidence,
launcher, launcher id). Duplicates merge by what they launch (Steam found three
times = one app); a Chrome *profile* shortcut stays separate from Chrome.

Discovery runs in the background at startup: 153 entries in ~1.7 s, ~3 MB RAM.

## Names

Aliases are generated, not hard-coded: "Grand Theft Auto V Legacy" ->
`gta, gta v, gta 5, gtav, gta5, grand theft auto v, ...`; publisher prefixes
dropped ("Google Chrome" -> chrome); exe names ("Code.exe" -> code); roman/digit
swaps; launch words stripped ("fire up", "chala de", "start karo", "khol do re").
Fuzzy matching tolerates typos ("chorme"). Several close matches are never
guessed: NEXUS asks ("I found Outlook (classic) and Outlook. Which one?") and
the next answer ("the new one", "dusra", "GTA V") is matched by code.

## Launching and verification

1. Resolve against the catalog (never a raw name/path/command).
2. Refuse shells, admin tools, programs in Downloads/temp, programs that demand
   administrator rights (read from the exe's manifest; NEXUS never elevates).
3. Already running? Bring its window to the front instead of a second copy.
4. Launch: the discovered exe with the shortcut's own arguments (no shell),
   a launcher URI (`steam://`, `com.epicgames.launcher://` only), or a Store app id.
5. Verify: wait for a *new* process that belongs to it (exact exe path, the
   game's own exe inside its install folder, or the Store package), then its window.
6. Only then: "Google Chrome is open." / "Grand Theft Auto V Legacy is running."
   Otherwise: "I couldn't verify that X started: gta5.exe didn't appear within 180 seconds."
   Launcher-mediated apps (BlueStacks, a Chrome profile) that can't be verified
   say so: "I asked Windows to open X, but I can't confirm it actually started."

Apps verify inline (usually 1-2 s). **Games launch asynchronously**:
`launch_game()` enters game mode, hands the launch to Steam/Epic/the exe and
returns at once - "Starting Grand Theft Auto V Legacy - I'll tell you once it's
running." - so the console (and a phone) stay responsive. A background monitor
then verifies the game's own process and emits an event; the console prints
"Grand Theft Auto V Legacy is running." (or "I couldn't verify that ... started")
*only after* verification, the audit log records it, and a paired phone that
asked receives it as a signed message. Asking again while it starts doesn't
launch it twice. Measured live: launch_game returns in ~0.1 s.

## Closing apps and games

`close_app(name)` - the only input is a name, resolved through the catalog.
1. The processes come from the catalog record only (exact exe, or the game's own
   exe in its folder). No PID, path or command is ever accepted ("close 1234",
   "close chrome.exe", "taskkill ..." are refused).
2. Protected processes are refused: Windows' own (explorer, csrss, svchost,
   winlogon, dwm...), anything owned by the system or another user, and NEXUS
   itself (its process, console and terminal).
3. Graceful first: its windows are asked to close (WM_CLOSE - the same as
   clicking X; the app may ask to save). No shell, no taskkill.
4. Verified: "Google Chrome is closed." only when its processes are gone;
   "Google Chrome isn't running." if it wasn't; "I asked Google Chrome to close,
   but it's still running - it may be waiting for you..." if it didn't close.
   NEXUS never escalates on its own.
5. `force_close_app(name)` is separate, always asks for confirmation ("unsaved
   work will be lost"), and ends only the processes the record matched, after the
   same protection checks.
6. Closing a game ends game mode at once; AI models are not reloaded.
7. Apps whose process can't be identified (launcher-mediated) are reported as
   not supported rather than guessed at.

## Your own app names (aliases)

"Chrome is my browser", "Call Photoshop my editor", "Remember that Steam is my
game launcher", "OBS mera recorder hai" -> then "open my browser", "launch my
editor", "recorder khol do".

- Stored in the local memory database (`app_aliases` table): device-local,
  visible in `/aliases` and in "what do you remember about me?", removable with
  "remove my browser alias" or `/unalias browser`, kept across restarts. Not
  offered for sync.
- Resolution is exact and deterministic, checked before normal name matching.
  If an installed app later has exactly the same name, both are shown and you choose.
- Refused: shells/admin tools or downloaded programs as targets; reserved names
  (cmd, powershell, terminal, system, nexus, downloads, ...); a name that already
  belongs to another installed app ("Chrome is my steam"); symbols or paths
  ("a/b", "x && del"); more than 4 words.
- "X is my Y" only becomes an alias when X is an installed app; otherwise it is
  saved as a normal memory ("Priya is my sister").

## Response semantics

| Kind | Example | Status / Say kind |
|---|---|---|
| success | "Google Chrome is open." | DONE / RESULT |
| failure (attempted, didn't work) | "I couldn't open Google Chrome: access denied." | FAILED / ERROR |
| refusal (NEXUS won't, on purpose) | "I can't launch Windows PowerShell through NEXUS - ..." | BLOCKED / REFUSAL |
| clarification | "I found Outlook (classic) and Outlook. Which one do you want?" | NEEDS_CLARIFICATION / CLARIFY |
| unsupported | "I don't support launching games through Ubisoft Connect yet." | UNSUPPORTED / UNSUPPORTED |

A security refusal is never described as a failed operation. Game shortcuts for
launchers NEXUS can't drive (Ubisoft Connect, Battle.net, the EA app, GOG
Galaxy, Roblox) are recognised and reported as unsupported.

The game's real process is known for common titles (GTA5.exe, RDR2.exe,
VALORANT-Win64-Shipping.exe...) and otherwise found by name similarity inside
the install folder - not by size (setup and ffmpeg binaries are often bigger).

## Games and resources

`GameService.launch_game()` enters GAME_MODE *before* the game starts
(AI models leave the GPU), and gives it back if the launch can't be verified.
A lightweight `GameWatcher` (one process scan every 5 s) turns game mode on
when a known game runs - also games started outside NEXUS - and off when it
exits. AI models then reload only when a request needs them. A manual
`/gamemode on` is never undone by the watcher.

## Guards against the model (seen in evaluation)

- The model chose `open_url google.com/chrome` for "Chrome khol de re": if you
  name an installed app and don't ask for a website, the app is started instead.
- The model chose `launch_app Steam` for "wo browser wali baat bhool ja": a
  launch of an app you never mentioned becomes a question ("Should I open Steam?").

- The model chose `close_app`/`launch_app` for an app you never mentioned: the
  same grounding check asks first ("Should I close Steam?").

## Measured on the dev PC (final code, 2026-09-28)

| | |
|---|---|
| discovery | 153 entries, 1.5-1.6 s, ~3 MB RAM; idle NEXUS 35 MB RAM, 0 VRAM, no model loaded |
| shortcut parsing | 146/150 identical to Windows |
| name resolution | 42/43 exact; the 43rd ("outlook") correctly asks (two Outlooks installed) |
| shells/admin tools marked for refusal | 8/8 |
| games | 3/3 launcher-installed games (GTA V Legacy, Once Human via Steam; Rocket League via Epic), 3/3 with the game exe located; VALORANT found via its Riot shortcut |
| fast path | 14/14 launch/folder/site requests without a model; 6/6 uncertain ones left to the planner |
| real launch | Character Map: "is open" (process + window verified), then "already running - I switched to it" |
| real close | Character Map: "is closed" (WM_CLOSE, verified gone), then "isn't running"; "close 1234" refused |
| async game lifecycle (live) | launch_game returned in 96 ms; VRAM 6609 -> 1459 MB in game mode; verified in background; closed; game mode off; 8B not reloaded |
| planner (78 phrases) | system 97.4% (fast path + model), model alone 96.2%; fast path 37/78 with 0 wrong; close 5/5, aliases 2/2, refusals 7/7 |

**Subsystem 4 is FROZEN** (2026-09-28): no further feature additions; only fixes.

## Not in Subsystem 4

Riot/Xbox/GOG/Battle.net/EA/Ubisoft providers (recognised, reported as
unsupported), parsing shell-item-only shortcuts (they are launched via the
shortcut file), closing Store apps whose windows belong to ApplicationFrameHost
(reported honestly as not closable gracefully), alias sync between devices
(scope column exists, sync not built), Linux sources (`.desktop` files; Steam on
Linux uses the same manifests).
