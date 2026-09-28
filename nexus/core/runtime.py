"""Wires the platform, policy, tools, memory, apps, resources and audit log into a running kernel.

Every front end (console now; AI, voice, desktop shell, mesh later) starts here.
Memory, apps and resource management belong to the core, not the AI layer:
they work with --no-ai too.
"""

from __future__ import annotations

from dataclasses import dataclass

from nexus.ai.memory.sqlite_store import SQLiteMemoryStore
from nexus.core.config import Settings
from nexus.core.kernel import ActionKernel
from nexus.core.resources import ResourceManager
from nexus.security.audit import AuditLog
from nexus.security.permissions import Policy
from nexus.system.apps.service import AppService, AppSettings
from nexus.system.platform import Platform
from nexus.tools import ToolRegistry, default_tools


@dataclass(frozen=True)
class Runtime:
    settings: Settings
    platform: Platform
    kernel: ActionKernel
    registry: ToolRegistry
    policy: Policy
    audit: AuditLog
    memory: SQLiteMemoryStore | None = None
    resources: ResourceManager | None = None
    apps: AppService | None = None


def create_runtime(settings: Settings, platform: Platform, *, resources: ResourceManager | None = None,
                   app_settings: AppSettings = AppSettings()) -> Runtime:
    # NEXUS may never modify its own code, config, audit trail, identity keys or memory database
    # through tools: all of them live under project_root / data_dir / config.
    builtin_protected = [*platform.protected_paths(), settings.project_root,
                         settings.data_dir, settings.policy_file.parent]
    policy = Policy.from_file(settings.policy_file, builtin_protected=builtin_protected,
                              default_safe_roots=platform.safe_roots())
    memory = SQLiteMemoryStore(settings.data_dir / "memory" / "memory.db")
    resources = resources or ResourceManager()
    # Discovery is lazy; user app aliases live in the local memory database.
    apps = AppService(platform, game_mode=resources, settings=app_settings, aliases=memory)
    registry = ToolRegistry(default_tools(platform, memory, apps))
    audit = AuditLog(settings.audit_file)
    # Background verification results (asynchronous game launches) are audited like everything else.
    apps.subscribe(lambda e: audit.record("app_verified", kind=e.kind, app=e.app, ok=e.ok, verified=e.verified,
                                          device=e.origin, message=e.message[:300]))
    kernel = ActionKernel(registry, policy, audit)
    return Runtime(settings, platform, kernel, registry, policy, audit, memory, resources, apps)
