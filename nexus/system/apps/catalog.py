"""The app catalog: classified, de-duplicated records and name resolution.

The same program is often found several times (Start menu + Desktop + App
Paths; Steam's manifest + a Steam .url shortcut). Entries are merged by what
they actually launch, so "Steam" found three times is one app - while a Chrome
*profile* shortcut stays separate from Chrome itself.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Iterable, Sequence

from nexus.system.apps.names import aliases_for, query_variants, score
from nexus.system.apps.records import AppKind, AppRecord, GameRecord, RawEntry
from nexus.system.apps.sources import MEDIATED_ARGS, classify
from nexus.system.games.providers import KNOWN_GAME_PROCESSES, game_process_names
from nexus.system.apps.names import normalize
from nexus.system.platform.base import LaunchMethod

_PREFERRED_SOURCE = {"steam": 0, "epic": 0, "rockstar": 1, "start_menu": 2, "store": 3, "desktop": 4, "app_paths": 5}
MIN_SCORE = 0.75
CONFIDENT = 0.85
AMBIGUITY_MARGIN = 0.05


@dataclass(frozen=True)
class Resolution:
    query: str
    match: AppRecord | None = None
    candidates: tuple[AppRecord, ...] = ()   # >1 = ambiguous: ask, never guess
    score: float = 0.0

    @property
    def ambiguous(self) -> bool:
        return self.match is None and len(self.candidates) > 1

    @property
    def confident(self) -> bool:
        return self.match is not None and self.score >= CONFIDENT


def record_from_entry(entry: RawEntry) -> AppRecord | None:
    decided = classify(entry)
    if decided is None:
        return None
    kind, launcher = decided
    mediated = bool(MEDIATED_ARGS.search(entry.arguments))
    if entry.url:
        method, target = LaunchMethod.URI, entry.url
    elif entry.aumid:
        method, target = LaunchMethod.PACKAGED, entry.aumid
    elif entry.target:
        method, target = LaunchMethod.EXECUTABLE, entry.target
    elif entry.shortcut:
        method, target = LaunchMethod.SHORTCUT, entry.shortcut
    else:
        return None

    process_paths: frozenset[str] = frozenset()
    process_names: frozenset[str] = frozenset()
    if method is LaunchMethod.EXECUTABLE and not mediated:
        process_paths = frozenset({os.path.normcase(os.path.abspath(target))})
        # Squirrel-style launchers: "Update.exe --processStart Discord.exe"
        parts = entry.arguments.split()
        if "--processStart" in parts and parts.index("--processStart") + 1 < len(parts):
            process_names = frozenset({parts[parts.index("--processStart") + 1].strip('"').lower()})
    if kind is AppKind.GAME:
        core = normalize(entry.name)
        known = next((names for title, names in KNOWN_GAME_PROCESSES.items() if core.startswith(title)), ())
        process_names = frozenset(known) or process_names

    stem = Path(target).stem if method is LaunchMethod.EXECUTABLE and not entry.arguments else None
    unsupported = None
    if method is LaunchMethod.URI and launcher not in ("steam", "epic"):
        unsupported = f"I don't support launching games through {launcher or 'that launcher'} yet."
    cls = GameRecord if kind is AppKind.GAME else AppRecord
    return cls(
        id=f"{entry.source}:{_merge_key(method, target, entry.arguments, entry.name)}",
        display_name=entry.name.strip(), aliases=aliases_for(entry.name, exe_stem=stem, extra=entry.extra_aliases),
        kind=kind, source=entry.source, launch_method=method, target=target, arguments=entry.arguments,
        working_dir=entry.working_dir, publisher=entry.publisher,
        install_dir=str(Path(target).parent) if method is LaunchMethod.EXECUTABLE else None,
        process_names=process_names, process_paths=process_paths,
        package_family=entry.aumid.split("!")[0] if entry.aumid else None,
        mediated=mediated, confidence=entry.confidence, launcher=launcher, unsupported_reason=unsupported)


def _merge_key(method: LaunchMethod, target: str, arguments: str, name: str) -> str:
    if method is LaunchMethod.EXECUTABLE:
        return f"exe|{os.path.normcase(os.path.abspath(target))}|{' '.join(arguments.lower().split())}"
    if method is LaunchMethod.SHORTCUT:
        return f"lnk|{normalize(name)}"  # shell items: same name in several Start menu folders = one app
    return f"{method.value}|{target.lower().rstrip('/')}"


def _merge(records: list[AppRecord]) -> AppRecord:
    records = sorted(records, key=lambda r: (_PREFERRED_SOURCE.get(r.source.split("+")[0], 9), -r.confidence))
    best = records[0]
    kind = best.kind
    for r in records:  # the most restrictive/specific classification wins
        if r.kind in (AppKind.SHELL, AppKind.ADMIN):
            kind = r.kind
            break
        if r.kind is AppKind.GAME:
            kind = AppKind.GAME
    cls = GameRecord if kind is AppKind.GAME else AppRecord
    aliases = frozenset().union(*(r.aliases for r in records))
    fields = {f: getattr(best, f) for f in best.__dataclass_fields__}
    fields.update(
        kind=kind, aliases=aliases,
        source="+".join(dict.fromkeys(s for r in records for s in r.source.split("+"))),
        process_names=frozenset().union(*(r.process_names for r in records)),
        process_paths=frozenset().union(*(r.process_paths for r in records)),
        install_dir=best.install_dir or next((r.install_dir for r in records if r.install_dir), None),
        launcher=best.launcher or next((r.launcher for r in records if r.launcher), None),
        # supported if any source can start it (e.g. a Steam provider record beats an unknown shortcut)
        unsupported_reason=None if any(r.unsupported_reason is None for r in records) else best.unsupported_reason,
        publisher=best.publisher or next((r.publisher for r in records if r.publisher), None),
        confidence=max(r.confidence for r in records))
    return cls(**fields)


class AppCatalog:
    def __init__(self, records: Iterable[AppRecord]) -> None:
        self.records: tuple[AppRecord, ...] = tuple(sorted(records, key=lambda r: r.display_name.lower()))
        self._by_id = {r.id: r for r in self.records}

    @classmethod
    def build(cls, entries: Iterable[RawEntry], games: Iterable[GameRecord] = ()) -> AppCatalog:
        groups: dict[str, list[AppRecord]] = {}
        for game in games:
            groups.setdefault(_merge_key(game.launch_method, game.target, game.arguments, game.display_name),
                              []).append(game)
        game_dirs = {os.path.normcase(g.install_dir): key for key, gs in groups.items()
                     for g in gs if g.install_dir}
        for entry in entries:
            record = record_from_entry(entry)
            if record is None:
                continue
            key = _merge_key(record.launch_method, record.target, record.arguments, record.display_name)
            # An exe inside a launcher-managed game folder is that game (e.g. Rockstar + Steam GTA V).
            if record.launch_method is LaunchMethod.EXECUTABLE and record.install_dir:
                for folder, game_key in game_dirs.items():
                    if os.path.normcase(record.install_dir).startswith(folder):
                        key = game_key
                        break
            groups.setdefault(key, []).append(record)
        # App Paths registrations only add aliases to apps found elsewhere; on their own they are mostly
        # helper binaries ("msoadfsb", "sdxhelper") that would only make names ambiguous.
        merged = [m for m in (_merge(g) for g in groups.values()) if m.source != "app_paths"]
        # Same game from two launchers (same install folder): keep one, preferring the launcher that owns it.
        by_dir: dict[str, AppRecord] = {}
        out = []
        for r in merged:
            folder = os.path.normcase(r.install_dir) if (r.kind is AppKind.GAME and r.install_dir) else None
            if folder and folder in by_dir:
                by_dir[folder] = _merge([by_dir[folder], r])
                continue
            if folder:
                by_dir[folder] = r
            else:
                out.append(r)
        out += list(by_dir.values())
        # Unknown-process games from shortcuts get a process from their folder when it's a real game folder.
        final = []
        for r in out:
            if r.kind is AppKind.GAME and not r.process_names and r.install_dir and not r.mediated:
                r = replace(r, process_names=game_process_names(r.display_name, Path(r.install_dir)))
            final.append(r)
        return cls(final)

    def get(self, record_id: str) -> AppRecord | None:
        return self._by_id.get(record_id)

    def games(self) -> list[AppRecord]:
        return [r for r in self.records if r.kind is AppKind.GAME]

    def resolve(self, query: str, *, kinds: Sequence[AppKind] | None = None) -> Resolution:
        forms = query_variants(query)
        if not forms:
            return Resolution(query)
        scored = []
        for r in self.records:
            if kinds and r.kind not in kinds:
                continue
            s = score(forms, r.aliases) * (0.9 + 0.1 * r.confidence)
            if s >= MIN_SCORE:
                scored.append((s, r))
        if not scored:
            return Resolution(query)
        scored.sort(key=lambda sr: (-sr[0], sr[1].display_name.lower()))
        best_score, best = scored[0]
        close = [r for s, r in scored if s >= best_score - AMBIGUITY_MARGIN]
        if len(close) > 1:
            return Resolution(query, candidates=tuple(close[:5]), score=best_score)
        return Resolution(query, match=best, candidates=(best,), score=best_score)
