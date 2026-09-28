"""Game launchers behind one interface.

    GameProvider
    +-- SteamProvider      steamapps/libraryfolders.vdf + appmanifest_*.acf
    +-- EpicProvider       ProgramData/Epic/EpicGamesLauncher/Data/Manifests/*.item
    +-- RockstarProvider   Rockstar Games Launcher registry entries

The rest of NEXUS only asks GameService: find_game(), launch_game(), is_game_running().
"""
