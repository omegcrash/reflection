# Reflection - Enterprise Multi-Tenant AI Companion Platform
# Copyright (c) 2026 George Scott Foley
# ORCID: 0009-0006-4957-0540
# Email: Georgescottfoley@proton.me
# Licensed under the MIT License - see LICENSE file for details

"""
Usage Alert Service (v1.3.0)

Budget monitoring and alerting service that:
- Tracks usage against budgets/quotas
- Sends alerts at configurable thresholds (default: 80%, 100%)
- Supports multiple notification channels (webhook, email)
- Provides real-time budget status

Usage:
    service = UsageAlertService()

    # Check and alert if needed
    alerts = await service.check_and_alert(
        tenant_id=tenant_id,
        current_usage=usage,
        limits=limits,
    )

    # Get budget status
    status = await service.get_budget_status(tenant_id)
"""

import asyncio
import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from enum import StrEnum
from typing import Any
from uuid import UUID

import httpx

logger = logging.getLogger(__name__)


# ============================================================
# ALERT TYPES
# ============================================================


class AlertSeverity(StrEnum):
    """Alert severity levels."""

    INFO = "info"  # Below warning threshold
    WARNING = "warning"  # At or above warning threshold (default: 80%)
    CRITICAL = "critical"  # At or above critical threshold (default: 95%)
    EXCEEDED = "exceeded"  # At or above 100%


class AlertType(StrEnum):
    """Types of usage alerts."""

    TOKENS_DAILY = "tokens_daily"
    TOKENS_MONTHLY = "tokens_monthly"
    REQUESTS_MINUTE = "requests_minute"
    REQUESTS_HOURLY = "requests_hourly"
    REQUESTS_DAILY = "requests_daily"
    COST_DAILY = "cost_daily"
    COST_MONTHLY = "cost_monthly"
    CONCURRENT = "concurrent"
    TOOL_EXECUTIONS = "tool_executions"


@dataclass
class UsageAlert:
    """
    A usage alert that may need to be sent.
    """

    tenant_id: UUID
    alert_type: AlertType
    severity: AlertSeverity
    current_value: int | Decimal
    limit_value: int | Decimal
    percentage: float
    message: str
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))

    # Tracking
    alert_id: str = ""
    sent: bool = False
    acknowledged: bool = False

    def __post_init__(self):
        if not self.alert_id:
            self.alert_id = (
                f"{self.tenant_id}:{self.alert_type.value}:{int(self.timestamp.timestamp())}"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "alert_id": self.alert_id,
            "tenant_id": str(self.tenant_id),
            "alert_type": self.alert_type.value,
            "severity": self.severity.value,
            "current_value": str(self.current_value),
            "limit_value": str(self.limit_value),
            "percentage": self.percentage,
            "message": self.message,
            "timestamp": self.timestamp.isoformat(),
            "sent": self.sent,
            "acknowledged": self.acknowledged,
        }


# ============================================================
# ALERT THRESHOLDS
# ============================================================


@dataclass
class AlertThresholds:
    """
    Configurable alert thresholds.

    Alerts are triggered when usage reaches these percentages.
    """

    warning_percent: float = 80.0  # First warning
    critical_percent: float = 95.0  # Critical warning
    exceeded_percent: float = 100.0  # Hard limit exceeded

    # Cooldown to prevent alert spam
    cooldown_minutes: int = 60

    # Which alert types to monitor
    enabled_types: set[AlertType] = field(
        default_factory=lambda: {
            AlertType.TOKENS_DAILY,
            AlertType.TOKENS_MONTHLY,
            AlertType.COST_DAILY,
            AlertType.COST_MONTHLY,
            AlertType.REQUESTS_DAILY,
        }
    )

    def get_severity(self, percentage: float) -> AlertSeverity:
        """Get severity for a given usage percentage."""
        if percentage >= self.exceeded_percent:
            return AlertSeverity.EXCEEDED
        elif percentage >= self.critical_percent:
            return AlertSeverity.CRITICAL
        elif percentage >= self.warning_percent:
            return AlertSeverity.WARNING
        else:
            return AlertSeverity.INFO


# ============================================================
# NOTIFICATION HANDLERS
# ============================================================


class NotificationHandler:
    """Base class for notification handlers."""

    async def send(self, alert: UsageAlert) -> bool:
        """Send an alert notification. Returns True if successful."""
        raise NotImplementedError


class WebhookNotificationHandler(NotificationHandler):
    """Send alerts via webhook."""

    def __init__(
        self,
        webhook_url: str,
        timeout: float = 10.0,
        headers: dict[str, str] | None = None,
    ):
        self.webhook_url = webhook_url
        self.timeout = timeout
        self.headers = headers or {}

    async def send(self, alert: UsageAlert) -> bool:
        """Send alert to webhook."""
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(
                    self.webhook_url,
                    json=alert.to_dict(),
                    headers={
                        "Content-Type": "application/json",
                        **self.headers,
                    },
                )

                if response.status_code < 300:
                    logger.info(
                        f"Alert sent to webhook: {alert.alert_type.value} "
                        f"severity={alert.severity.value}"
                    )
                    return True
                else:
                    logger.warning(
                        f"Webhook returned {response.status_code}: {response.text[:200]}"
                    )
                    return False

        except Exception as e:
            logger.error(f"Failed to send webhook alert: {e}")
            return False


class LogNotificationHandler(NotificationHandler):
    """Log alerts (for development/testing)."""

    def __init__(self, log_level: int = logging.WARNING):
        self.log_level = log_level

    async def send(self, alert: UsageAlert) -> bool:
        """Log the alert."""
        logger.log(
            self.log_level,
            f"USAGE ALERT [{alert.severity.value.upper()}] "
            f"tenant={alert.tenant_id} type={alert.alert_type.value} "
            f"usage={alert.percentage:.1f}% ({alert.current_value}/{alert.limit_value}) "
            f"- {alert.message}",
        )
        return True


class CallbackNotificationHandler(NotificationHandler):
    """Call a custom async callback for alerts."""

    def __init__(self, callback: Callable[[UsageAlert], Any]):
        self.callback = callback

    async def send(self, alert: UsageAlert) -> bool:
        """Call the callback with the alert."""
        try:
            result = self.callback(alert)
            if asyncio.iscoroutine(result):
                await result
            return True
        except Exception as e:
            logger.error(f"Alert callback failed: {e}")
            return False


# ============================================================
# USAGE ALERT SERVICE
# ============================================================


class UsageAlertService:
    """
    Service for monitoring usage and sending alerts.

    This service:
    1. Checks current usage against configured limits
    2. Determines if alerts should be sent based on thresholds
    3. Sends alerts through configured notification handlers
    4. Tracks sent alerts to prevent spam

    Usage:
        # Create service with handlers
        service = UsageAlertService()
        service.add_handler(WebhookNotificationHandler("https://..."))
        service.add_handler(LogNotificationHandler())

        # Check usage and send alerts
        alerts = await service.check_and_alert(
            tenant_id=tenant_id,
            usage_data={
                "tokens_daily": {"current": 45000, "limit": 50000},
                "cost_daily": {"current": Decimal("4.50"), "limit": Decimal("5.00")},
            },
        )
    """

    def __init__(
        self,
        thresholds: AlertThresholds | None = None,
        handlers: list[NotificationHandler] | None = None,
    ):
        """
        Initialize the alert service.

        Args:
            thresholds: Alert threshold configuration
            handlers: List of notification handlers
        """
        self.thresholds = thresholds or AlertThresholds()
        self.handlers: list[NotificationHandler] = handlers or [LogNotificationHandler()]

        # Track sent alerts to prevent spam
        # Key: "{tenant_id}:{alert_type}:{severity}"
        # Value: timestamp of last alert
        self._sent_alerts: dict[str, datetime] = {}

    def add_handler(self, handler: NotificationHandler) -> None:
        """Add a notification handler."""
        self.handlers.append(handler)

    def remove_handler(self, handler: NotificationHandler) -> None:
        """Remove a notification handler."""
        if handler in self.handlers:
            self.handlers.remove(handler)

    async def check_and_alert(
        self,
        tenant_id: UUID,
        usage_data: dict[str, dict[str, Any]],
        force: bool = False,
    ) -> list[UsageAlert]:
        """
        Check usage against limits and send alerts if thresholds are exceeded.

        Args:
            tenant_id: Tenant ID to check
            usage_data: Dictionary mapping alert types to {current, limit} dicts
            force: Send alerts even if within cooldown period

        Returns:
            List of alerts that were triggered (may or may not have been sent)

        Example usage_data:
            {
                "tokens_daily": {"current": 45000, "limit": 50000},
                "tokens_monthly": {"current": 900000, "limit": 1000000},
                "cost_daily": {"current": Decimal("4.50"), "limit": Decimal("5.00")},
            }
        """
        alerts = []

        for type_key, values in usage_data.items():
            # Map string key to AlertType
            try:
                alert_type = AlertType(type_key)
            except ValueError:
                logger.debug(f"Unknown alert type: {type_key}")
                continue

            # Check if this type is enabled
            if alert_type not in self.thresholds.enabled_types:
                continue

            current = values.get("current", 0)
            limit = values.get("limit", 0)

            if limit <= 0:
                continue

            # Calculate percentage
            if isinstance(current, Decimal) or isinstance(limit, Decimal):
                percentage = float(Decimal(str(current)) / Decimal(str(limit)) * 100)
            else:
                percentage = (current / limit) * 100

            # Determine severity
            severity = self.thresholds.get_severity(percentage)

            # Only alert if at warning level or above
            if severity == AlertSeverity.INFO:
                continue

            # Create alert
            message = self._format_message(alert_type, severity, percentage, current, limit)

            alert = UsageAlert(
                tenant_id=tenant_id,
                alert_type=alert_type,
                severity=severity,
                current_value=current,
                limit_value=limit,
                percentage=percentage,
                message=message,
            )

            alerts.append(alert)

            # Check cooldown
            if not force and self._in_cooldown(tenant_id, alert_type, severity):
                logger.debug(f"Alert in cooldown: {alert_type.value} for tenant {tenant_id}")
                continue

            # Send alert
            await self._send_alert(alert)

        return alerts

    async def check_budget(
        self,
        tenant_id: UUID,
        current_cost: Decimal,
        daily_budget: Decimal,
        monthly_budget: Decimal,
        force: bool = False,
    ) -> list[UsageAlert]:
        """
        Convenience method to check cost budgets.

        Args:
            tenant_id: Tenant ID
            current_cost: Current cost for today
            daily_budget: Daily budget limit
            monthly_budget: Monthly budget limit
            force: Force alert even if in cooldown

        Returns:
            List of triggered alerts
        """
        # Get monthly cost (this would come from actual tracking)
        # For now, estimate based on daily * days so far
        day_of_month = datetime.now(UTC).day
        estimated_monthly = current_cost * day_of_month

        usage_data = {
            "cost_daily": {
                "current": current_cost,
                "limit": daily_budget,
            },
            "cost_monthly": {
                "current": estimated_monthly,
                "limit": monthly_budget,
            },
        }

        return await self.check_and_alert(tenant_id, usage_data, force)

    async def get_budget_status(
        self,
        tenant_id: UUID,
        usage_data: dict[str, dict[str, Any]],
    ) -> dict[str, Any]:
        """
        Get current budget/usage status without sending alerts.

        Returns a summary of usage status for all tracked metrics.
        """
        status = {
            "tenant_id": str(tenant_id),
            "timestamp": datetime.now(UTC).isoformat(),
            "metrics": {},
        }

        for type_key, values in usage_data.items():
            current = values.get("current", 0)
            limit = values.get("limit", 0)

            if limit <= 0:
                continue

            if isinstance(current, Decimal) or isinstance(limit, Decimal):
                percentage = float(Decimal(str(current)) / Decimal(str(limit)) * 100)
            else:
                percentage = (current / limit) * 100

            severity = self.thresholds.get_severity(percentage)

            status["metrics"][type_key] = {
                "current": str(current) if isinstance(current, Decimal) else current,
                "limit": str(limit) if isinstance(limit, Decimal) else limit,
                "percentage": round(percentage, 2),
                "remaining": str(limit - current)
                if isinstance(current, Decimal)
                else limit - current,
                "severity": severity.value,
                "at_warning": severity
                in (AlertSeverity.WARNING, AlertSeverity.CRITICAL, AlertSeverity.EXCEEDED),
                "exceeded": severity == AlertSeverity.EXCEEDED,
            }

        # Overall status
        worst_severity = AlertSeverity.INFO
        for metric in status["metrics"].values():
            metric_severity = AlertSeverity(metric["severity"])
            if self._severity_rank(metric_severity) > self._severity_rank(worst_severity):
                worst_severity = metric_severity

        status["overall_status"] = worst_severity.value
        status["any_exceeded"] = worst_severity == AlertSeverity.EXCEEDED
        status["any_warning"] = self._severity_rank(worst_severity) >= self._severity_rank(
            AlertSeverity.WARNING
        )

        return status

    def _in_cooldown(
        self,
        tenant_id: UUID,
        alert_type: AlertType,
        severity: AlertSeverity,
    ) -> bool:
        """Check if an alert is in cooldown period."""
        key = f"{tenant_id}:{alert_type.value}:{severity.value}"

        last_sent = self._sent_alerts.get(key)
        if last_sent is None:
            return False

        cooldown = timedelta(minutes=self.thresholds.cooldown_minutes)
        return datetime.now(UTC) - last_sent < cooldown

    def _record_sent(
        self,
        tenant_id: UUID,
        alert_type: AlertType,
        severity: AlertSeverity,
    ) -> None:
        """Record that an alert was sent."""
        key = f"{tenant_id}:{alert_type.value}:{severity.value}"
        self._sent_alerts[key] = datetime.now(UTC)

    async def _send_alert(self, alert: UsageAlert) -> None:
        """Send alert through all handlers."""
        for handler in self.handlers:
            try:
                success = await handler.send(alert)
                if success:
                    alert.sent = True
            except Exception as e:
                logger.error(f"Handler {handler.__class__.__name__} failed: {e}")

        if alert.sent:
            self._record_sent(alert.tenant_id, alert.alert_type, alert.severity)

    def _format_message(
        self,
        alert_type: AlertType,
        severity: AlertSeverity,
        percentage: float,
        current: Any,
        limit: Any,
    ) -> str:
        """Format a human-readable alert message."""
        type_labels = {
            AlertType.TOKENS_DAILY: "Daily token usage",
            AlertType.TOKENS_MONTHLY: "Monthly token usage",
            AlertType.REQUESTS_MINUTE: "Requests per minute",
            AlertType.REQUESTS_HOURLY: "Hourly requests",
            AlertType.REQUESTS_DAILY: "Daily requests",
            AlertType.COST_DAILY: "Daily cost",
            AlertType.COST_MONTHLY: "Monthly cost",
            AlertType.CONCURRENT: "Concurrent requests",
            AlertType.TOOL_EXECUTIONS: "Tool executions",
        }

        label = type_labels.get(alert_type, alert_type.value)

        if severity == AlertSeverity.EXCEEDED:
            return f"{label} has exceeded the limit: {current}/{limit} ({percentage:.1f}%)"
        elif severity == AlertSeverity.CRITICAL:
            return f"{label} is critically high: {current}/{limit} ({percentage:.1f}%)"
        else:
            return f"{label} is approaching the limit: {current}/{limit} ({percentage:.1f}%)"

    @staticmethod
    def _severity_rank(severity: AlertSeverity) -> int:
        """Get numeric rank for severity comparison."""
        ranks = {
            AlertSeverity.INFO: 0,
            AlertSeverity.WARNING: 1,
            AlertSeverity.CRITICAL: 2,
            AlertSeverity.EXCEEDED: 3,
        }
        return ranks.get(severity, 0)

    def clear_cooldowns(self, tenant_id: UUID | None = None) -> None:
        """
        Clear alert cooldowns.

        Args:
            tenant_id: If provided, only clear for this tenant.
                      If None, clear all cooldowns.
        """
        if tenant_id is None:
            self._sent_alerts.clear()
        else:
            prefix = f"{tenant_id}:"
            self._sent_alerts = {
                k: v for k, v in self._sent_alerts.items() if not k.startswith(prefix)
            }


# ============================================================
# GLOBAL INSTANCE
# ============================================================

_alert_service: UsageAlertService | None = None


def get_usage_alert_service() -> UsageAlertService:
    """Get the global usage alert service."""
    global _alert_service
    if _alert_service is None:
        _alert_service = UsageAlertService()
    return _alert_service


def init_usage_alert_service(
    thresholds: AlertThresholds | None = None,
    webhook_url: str | None = None,
    handlers: list[NotificationHandler] | None = None,
) -> UsageAlertService:
    """
    Initialize the global usage alert service.

    Args:
        thresholds: Custom alert thresholds
        webhook_url: Webhook URL for notifications
        handlers: Custom notification handlers

    Returns:
        The initialized service
    """
    global _alert_service

    if handlers is None:
        handlers = [LogNotificationHandler()]
        if webhook_url:
            handlers.append(WebhookNotificationHandler(webhook_url))

    _alert_service = UsageAlertService(
        thresholds=thresholds,
        handlers=handlers,
    )

    logger.info(
        f"Usage alert service initialized with {len(handlers)} handlers, "
        f"thresholds: warning={thresholds.warning_percent if thresholds else 80}%, "
        f"critical={thresholds.critical_percent if thresholds else 95}%"
    )

    return _alert_service


def reset_usage_alert_service() -> None:
    """Reset the global service (for testing)."""
    global _alert_service
    _alert_service = None


# ============================================================
# EXPORTS
# ============================================================

__all__ = [
    # Types
    "AlertSeverity",
    "AlertType",
    "UsageAlert",
    "AlertThresholds",
    # Handlers
    "NotificationHandler",
    "WebhookNotificationHandler",
    "LogNotificationHandler",
    "CallbackNotificationHandler",
    # Service
    "UsageAlertService",
    "get_usage_alert_service",
    "init_usage_alert_service",
    "reset_usage_alert_service",
]
