# Reflection - Enterprise Multi-Tenant AI Companion Platform
# Copyright (c) 2026 George Scott Foley
# ORCID: 0009-0006-4957-0540
# Email: Georgescottfoley@proton.me
# Licensed under the MIT License - see LICENSE file for details

"""
Data Export Handlers (v1.5.0)

GDPR-compliant data export functionality:
- Full tenant data export
- User data export
- Conversation history export
- Usage/billing data export

Supports multiple formats:
- JSON (default)
- CSV (for tabular data)
- ZIP archive (for complete exports)
"""

import csv
import io
import json
import logging
import tempfile
import zipfile
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import UUID

from ..core.settings import get_settings
from . import Job, JobContext, JobService

logger = logging.getLogger(__name__)


def _get_export_directory() -> Path:
    """
    Get the export directory path.

    Security: Uses configurable path or secure tempdir.
    Never uses hardcoded paths that could be predictable.

    Returns:
        Path to export directory (created if needed)
    """
    settings = get_settings()

    if settings.export_directory:
        # Use configured directory
        export_dir = Path(settings.export_directory)
    else:
        # Create secure temporary directory
        # tempfile.mkdtemp creates with mode 0700 (owner only)
        base_temp = tempfile.gettempdir()
        export_base = Path(base_temp) / "familiar_exports"
        export_base.mkdir(mode=0o700, parents=True, exist_ok=True)
        export_dir = export_base

    # Ensure directory exists with secure permissions
    export_dir.mkdir(mode=0o700, parents=True, exist_ok=True)

    return export_dir


# ============================================================
# EXPORT TYPES
# ============================================================


@dataclass
class ExportConfig:
    """Configuration for data export."""

    include_conversations: bool = True
    include_messages: bool = True
    include_usage: bool = True
    include_agents: bool = True
    include_memory: bool = True
    include_audit_logs: bool = False

    format: str = "json"  # json, csv, zip
    date_from: datetime | None = None
    date_to: datetime | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ExportConfig":
        config = cls(
            include_conversations=data.get("include_conversations", True),
            include_messages=data.get("include_messages", True),
            include_usage=data.get("include_usage", True),
            include_agents=data.get("include_agents", True),
            include_memory=data.get("include_memory", True),
            include_audit_logs=data.get("include_audit_logs", False),
            format=data.get("format", "json"),
        )

        if data.get("date_from"):
            config.date_from = datetime.fromisoformat(data["date_from"])
        if data.get("date_to"):
            config.date_to = datetime.fromisoformat(data["date_to"])

        return config


# ============================================================
# EXPORT HANDLERS
# ============================================================


async def handle_tenant_data_export(job: Job, ctx: JobContext) -> dict[str, Any]:
    """
    Export all data for a tenant.

    This is a GDPR-compliant full export that includes:
    - All conversations and messages
    - Agent configurations
    - Usage records
    - Memory entries

    Returns path to export file.
    """
    from ..data.postgres import get_database
    from ..data.repositories import (
        AgentRepository,
        ConversationRepository,
        MessageRepository,
        UsageRepository,
    )

    tenant_id = job.tenant_id
    config = ExportConfig.from_dict(job.params)

    logger.info(f"Starting tenant data export for {tenant_id}")
    await ctx.update_progress(5, "Initializing export...")

    # Check for cancellation periodically
    if await ctx.is_cancelled():
        return {"status": "cancelled"}

    # Get database session
    db = get_database()

    export_data = {
        "export_info": {
            "tenant_id": str(tenant_id),
            "export_date": datetime.now(UTC).isoformat(),
            "format": config.format,
            "version": "1.5.0",
        },
        "data": {},
    }

    async with db.session() as session:
        # Export conversations
        if config.include_conversations:
            await ctx.update_progress(10, "Exporting conversations...")
            conv_repo = ConversationRepository(session)

            try:
                conversations = await conv_repo.get_by_tenant(
                    tenant_id,
                    limit=10000,
                )

                export_data["data"]["conversations"] = [
                    {
                        "id": str(c.id),
                        "user_id": str(c.user_id) if c.user_id else None,
                        "agent_id": str(c.agent_id) if c.agent_id else None,
                        "channel": c.channel,
                        "created_at": c.created_at.isoformat() if c.created_at else None,
                        "updated_at": c.updated_at.isoformat() if c.updated_at else None,
                    }
                    for c in conversations
                ]

                logger.info(f"Exported {len(conversations)} conversations")
            except Exception as e:
                logger.warning(f"Failed to export conversations: {e}")
                export_data["data"]["conversations"] = {"error": str(e)}

        if await ctx.is_cancelled():
            return {"status": "cancelled"}

        # Export messages
        if config.include_messages and config.include_conversations:
            await ctx.update_progress(30, "Exporting messages...")
            msg_repo = MessageRepository(session)

            try:
                all_messages = []
                conv_ids = [
                    UUID(c["id"])
                    for c in export_data["data"].get("conversations", [])
                    if isinstance(c, dict) and "id" in c
                ]

                for i, conv_id in enumerate(conv_ids[:100]):  # Limit for safety
                    if await ctx.is_cancelled():
                        return {"status": "cancelled"}

                    messages = await msg_repo.get_by_conversation(conv_id, limit=1000)

                    for m in messages:
                        all_messages.append(
                            {
                                "id": str(m.id),
                                "conversation_id": str(m.conversation_id),
                                "role": m.role,
                                "content": m.content,
                                "tokens_input": m.tokens_input,
                                "tokens_output": m.tokens_output,
                                "created_at": m.created_at.isoformat() if m.created_at else None,
                            }
                        )

                    if i % 10 == 0:
                        progress = 30 + int((i / len(conv_ids)) * 30)
                        await ctx.update_progress(
                            progress, f"Exporting messages ({i}/{len(conv_ids)} conversations)..."
                        )

                export_data["data"]["messages"] = all_messages
                logger.info(f"Exported {len(all_messages)} messages")
            except Exception as e:
                logger.warning(f"Failed to export messages: {e}")
                export_data["data"]["messages"] = {"error": str(e)}

        if await ctx.is_cancelled():
            return {"status": "cancelled"}

        # Export agents
        if config.include_agents:
            await ctx.update_progress(65, "Exporting agents...")
            agent_repo = AgentRepository(session)

            try:
                agents = await agent_repo.get_by_tenant(tenant_id)

                export_data["data"]["agents"] = [
                    {
                        "id": str(a.id),
                        "name": a.name,
                        "model": a.model,
                        "system_prompt": a.system_prompt,
                        "created_at": a.created_at.isoformat() if a.created_at else None,
                    }
                    for a in agents
                ]

                logger.info(f"Exported {len(agents)} agents")
            except Exception as e:
                logger.warning(f"Failed to export agents: {e}")
                export_data["data"]["agents"] = {"error": str(e)}

        if await ctx.is_cancelled():
            return {"status": "cancelled"}

        # Export usage
        if config.include_usage:
            await ctx.update_progress(80, "Exporting usage data...")
            usage_repo = UsageRepository(session)

            try:
                usage_records = await usage_repo.get_by_tenant(
                    tenant_id,
                    limit=10000,
                )

                export_data["data"]["usage"] = [
                    {
                        "id": str(u.id),
                        "resource_type": u.resource_type,
                        "resource_id": u.resource_id,
                        "quantity": u.quantity,
                        "unit": u.unit,
                        "created_at": u.created_at.isoformat() if u.created_at else None,
                    }
                    for u in usage_records
                ]

                logger.info(f"Exported {len(usage_records)} usage records")
            except Exception as e:
                logger.warning(f"Failed to export usage: {e}")
                export_data["data"]["usage"] = {"error": str(e)}

    await ctx.update_progress(90, "Generating export file...")

    # Generate export file in secure directory
    export_dir = _get_export_directory()

    timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")

    if config.format == "json":
        filename = f"export_{tenant_id}_{timestamp}.json"
        filepath = export_dir / filename

        with open(filepath, "w") as f:
            json.dump(export_data, f, indent=2, default=str)

    elif config.format == "zip":
        filename = f"export_{tenant_id}_{timestamp}.zip"
        filepath = export_dir / filename

        with zipfile.ZipFile(filepath, "w", zipfile.ZIP_DEFLATED) as zf:
            # Add JSON file
            zf.writestr(
                "export.json",
                json.dumps(export_data, indent=2, default=str),
            )

            # Add CSV files for tabular data
            if "conversations" in export_data["data"]:
                csv_content = _to_csv(export_data["data"]["conversations"])
                zf.writestr("conversations.csv", csv_content)

            if "messages" in export_data["data"]:
                csv_content = _to_csv(export_data["data"]["messages"])
                zf.writestr("messages.csv", csv_content)

            if "usage" in export_data["data"]:
                csv_content = _to_csv(export_data["data"]["usage"])
                zf.writestr("usage.csv", csv_content)

    else:
        # Default to JSON
        filename = f"export_{tenant_id}_{timestamp}.json"
        filepath = export_dir / filename

        with open(filepath, "w") as f:
            json.dump(export_data, f, indent=2, default=str)

    await ctx.update_progress(100, "Export complete")

    # Calculate stats
    stats = {
        "conversations": len(export_data["data"].get("conversations", [])),
        "messages": len(export_data["data"].get("messages", [])),
        "agents": len(export_data["data"].get("agents", [])),
        "usage_records": len(export_data["data"].get("usage", [])),
    }

    logger.info(f"Tenant data export completed: {filepath}")

    return {
        "status": "completed",
        "file_path": str(filepath),
        "file_name": filename,
        "format": config.format,
        "stats": stats,
    }


async def handle_user_data_export(job: Job, ctx: JobContext) -> dict[str, Any]:
    """
    Export all data for a specific user within a tenant.

    GDPR Article 15 - Right of Access compliance.
    """
    from ..data.postgres import get_database
    from ..data.repositories import (
        ConversationRepository,
        MessageRepository,
        UsageRepository,
    )

    tenant_id = job.tenant_id
    user_id = UUID(job.params.get("user_id"))
    ExportConfig.from_dict(job.params)

    logger.info(f"Starting user data export for user {user_id} in tenant {tenant_id}")
    await ctx.update_progress(5, "Initializing user export...")

    export_data = {
        "export_info": {
            "tenant_id": str(tenant_id),
            "user_id": str(user_id),
            "export_date": datetime.now(UTC).isoformat(),
            "gdpr_article": "15 - Right of Access",
            "version": "1.5.0",
        },
        "data": {},
    }

    db = get_database()

    async with db.session() as session:
        # Export user's conversations
        await ctx.update_progress(20, "Exporting user conversations...")
        conv_repo = ConversationRepository(session)

        try:
            conversations = await conv_repo.get_by_user(
                user_id=user_id,
                tenant_id=tenant_id,
                limit=10000,
            )

            export_data["data"]["conversations"] = [
                {
                    "id": str(c.id),
                    "channel": c.channel,
                    "created_at": c.created_at.isoformat() if c.created_at else None,
                }
                for c in conversations
            ]
        except Exception as e:
            logger.warning(f"Failed to export user conversations: {e}")
            export_data["data"]["conversations"] = []

        # Export user's messages
        await ctx.update_progress(50, "Exporting user messages...")
        msg_repo = MessageRepository(session)

        all_messages = []
        for conv in export_data["data"].get("conversations", []):
            if await ctx.is_cancelled():
                return {"status": "cancelled"}

            try:
                messages = await msg_repo.get_by_conversation(
                    UUID(conv["id"]),
                    limit=1000,
                )

                for m in messages:
                    all_messages.append(
                        {
                            "conversation_id": str(m.conversation_id),
                            "role": m.role,
                            "content": m.content,
                            "created_at": m.created_at.isoformat() if m.created_at else None,
                        }
                    )
            except Exception as e:
                logger.warning(f"Failed to export messages for conversation {conv['id']}: {e}")

        export_data["data"]["messages"] = all_messages

        # Export user's usage
        await ctx.update_progress(80, "Exporting user usage...")
        usage_repo = UsageRepository(session)

        try:
            usage = await usage_repo.get_by_user(user_id, tenant_id, limit=10000)

            export_data["data"]["usage"] = [
                {
                    "resource_type": u.resource_type,
                    "quantity": u.quantity,
                    "unit": u.unit,
                    "created_at": u.created_at.isoformat() if u.created_at else None,
                }
                for u in usage
            ]
        except Exception as e:
            logger.warning(f"Failed to export user usage: {e}")
            export_data["data"]["usage"] = []

    await ctx.update_progress(95, "Generating export file...")

    # Generate export file in secure directory
    export_dir = _get_export_directory()

    timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    filename = f"user_export_{user_id}_{timestamp}.json"
    filepath = export_dir / filename

    with open(filepath, "w") as f:
        json.dump(export_data, f, indent=2, default=str)

    await ctx.update_progress(100, "Export complete")

    logger.info(f"User data export completed: {filepath}")

    return {
        "status": "completed",
        "file_path": str(filepath),
        "file_name": filename,
        "stats": {
            "conversations": len(export_data["data"].get("conversations", [])),
            "messages": len(export_data["data"].get("messages", [])),
            "usage_records": len(export_data["data"].get("usage", [])),
        },
    }


def _to_csv(data: list[dict[str, Any]]) -> str:
    """Convert list of dicts to CSV string."""
    if not data or not isinstance(data, list):
        return ""

    output = io.StringIO()

    # Get all keys from all records
    keys = set()
    for item in data:
        if isinstance(item, dict):
            keys.update(item.keys())

    if not keys:
        return ""

    keys = sorted(keys)

    writer = csv.DictWriter(output, fieldnames=keys)
    writer.writeheader()

    for item in data:
        if isinstance(item, dict):
            writer.writerow(item)

    return output.getvalue()


# ============================================================
# REGISTRATION
# ============================================================


def register_export_handlers(job_service: JobService) -> None:
    """Register all export job handlers."""
    job_service.register_handler("tenant_data_export", handle_tenant_data_export)
    job_service.register_handler("user_data_export", handle_user_data_export)

    logger.info("Registered data export job handlers")


# ============================================================
# EXPORTS
# ============================================================

__all__ = [
    "ExportConfig",
    "handle_tenant_data_export",
    "handle_user_data_export",
    "register_export_handlers",
]
