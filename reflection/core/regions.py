# Reflection - Enterprise Multi-Tenant AI Companion Platform
# Copyright (c) 2026 George Scott Foley
# ORCID: 0009-0006-4957-0540
# Email: Georgescottfoley@proton.me
# Licensed under the MIT License - see LICENSE file for details

"""
Multi-Region Infrastructure (v2.0.0)

Provides region-aware routing and configuration:
- Region definitions and metadata
- Tenant region assignment
- Latency-based routing hints
- Cross-region replication hooks

Usage:
    from reflection.core.regions import (
        Region,
        RegionService,
        get_region_service,
    )

    # Get tenant's region
    region = await region_service.get_tenant_region(tenant_id)

    # Get nearest region for a client
    nearest = await region_service.get_nearest_region(client_ip)
"""

import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from uuid import UUID

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# ============================================================
# REGION DEFINITIONS
# ============================================================


class RegionCode(StrEnum):
    """Supported region codes."""

    US_EAST = "us-east-1"
    US_WEST = "us-west-2"
    EU_WEST = "eu-west-1"
    EU_CENTRAL = "eu-central-1"
    AP_SOUTHEAST = "ap-southeast-1"
    AP_NORTHEAST = "ap-northeast-1"


class RegionStatus(StrEnum):
    """Region operational status."""

    ACTIVE = "active"
    DEGRADED = "degraded"
    MAINTENANCE = "maintenance"
    OFFLINE = "offline"


@dataclass
class Region:
    """Region definition."""

    code: RegionCode
    name: str
    location: str
    status: RegionStatus = RegionStatus.ACTIVE

    # Endpoints
    api_endpoint: str | None = None
    db_endpoint: str | None = None
    redis_endpoint: str | None = None

    # Coordinates for latency calculation
    latitude: float = 0.0
    longitude: float = 0.0

    # Features
    features: list[str] = field(default_factory=list)

    # Metadata
    is_primary: bool = False
    failover_region: RegionCode | None = None


# Region registry
REGIONS: dict[RegionCode, Region] = {
    RegionCode.US_EAST: Region(
        code=RegionCode.US_EAST,
        name="US East (N. Virginia)",
        location="Virginia, USA",
        latitude=38.9519,
        longitude=-77.4480,
        is_primary=True,
        failover_region=RegionCode.US_WEST,
        features=["full", "gpu"],
    ),
    RegionCode.US_WEST: Region(
        code=RegionCode.US_WEST,
        name="US West (Oregon)",
        location="Oregon, USA",
        latitude=45.8399,
        longitude=-119.7006,
        failover_region=RegionCode.US_EAST,
        features=["full", "gpu"],
    ),
    RegionCode.EU_WEST: Region(
        code=RegionCode.EU_WEST,
        name="EU West (Ireland)",
        location="Dublin, Ireland",
        latitude=53.3498,
        longitude=-6.2603,
        failover_region=RegionCode.EU_CENTRAL,
        features=["full", "gdpr"],
    ),
    RegionCode.EU_CENTRAL: Region(
        code=RegionCode.EU_CENTRAL,
        name="EU Central (Frankfurt)",
        location="Frankfurt, Germany",
        latitude=50.1109,
        longitude=8.6821,
        failover_region=RegionCode.EU_WEST,
        features=["full", "gdpr"],
    ),
    RegionCode.AP_SOUTHEAST: Region(
        code=RegionCode.AP_SOUTHEAST,
        name="Asia Pacific (Singapore)",
        location="Singapore",
        latitude=1.3521,
        longitude=103.8198,
        failover_region=RegionCode.AP_NORTHEAST,
        features=["full"],
    ),
    RegionCode.AP_NORTHEAST: Region(
        code=RegionCode.AP_NORTHEAST,
        name="Asia Pacific (Tokyo)",
        location="Tokyo, Japan",
        latitude=35.6762,
        longitude=139.6503,
        failover_region=RegionCode.AP_SOUTHEAST,
        features=["full"],
    ),
}


# ============================================================
# REGION CONFIGURATION
# ============================================================


class RegionConfig(BaseModel):
    """Region-specific configuration for a tenant."""

    primary_region: str = Field(..., description="Primary region code")
    allowed_regions: list[str] = Field(
        default_factory=list,
        description="Additional allowed regions",
    )
    data_residency: str | None = Field(
        default=None,
        description="Required data residency region (GDPR, etc.)",
    )
    failover_enabled: bool = Field(default=True)
    routing_policy: str = Field(
        default="latency",
        description="Routing policy: latency, geo, sticky",
    )


@dataclass
class TenantRegionAssignment:
    """Tenant's region assignment."""

    tenant_id: UUID
    primary_region: RegionCode
    allowed_regions: list[RegionCode]
    data_residency: RegionCode | None = None
    routing_policy: str = "latency"
    assigned_at: datetime = field(default_factory=lambda: datetime.now(UTC))

    def is_region_allowed(self, region: RegionCode) -> bool:
        """Check if a region is allowed for this tenant."""
        if region == self.primary_region:
            return True
        return region in self.allowed_regions


# ============================================================
# REGION SERVICE
# ============================================================


class RegionService:
    """
    Multi-region management service.

    Handles region assignment, routing, and status.
    """

    def __init__(self, redis, database):
        """
        Initialize region service.

        Args:
            redis: Redis client
            database: Database connection
        """
        self.redis = redis
        self.db = database
        self._current_region: RegionCode | None = None

    def set_current_region(self, region: RegionCode) -> None:
        """Set the current region this instance is running in."""
        self._current_region = region
        logger.info(f"Current region set to: {region.value}")

    def get_current_region(self) -> RegionCode | None:
        """Get the current region."""
        return self._current_region

    # --------------------------------------------------------
    # REGION INFO
    # --------------------------------------------------------

    def get_region(self, code: RegionCode) -> Region | None:
        """Get region by code."""
        return REGIONS.get(code)

    def get_all_regions(self) -> list[Region]:
        """Get all regions."""
        return list(REGIONS.values())

    def get_active_regions(self) -> list[Region]:
        """Get regions with active status."""
        return [r for r in REGIONS.values() if r.status == RegionStatus.ACTIVE]

    def get_regions_with_feature(self, feature: str) -> list[Region]:
        """Get regions that support a specific feature."""
        return [r for r in REGIONS.values() if feature in r.features]

    # --------------------------------------------------------
    # TENANT ASSIGNMENT
    # --------------------------------------------------------

    async def assign_tenant_region(
        self,
        tenant_id: UUID,
        config: RegionConfig,
    ) -> TenantRegionAssignment:
        """
        Assign a tenant to regions.

        Args:
            tenant_id: Tenant ID
            config: Region configuration

        Returns:
            Region assignment
        """
        # Validate regions
        try:
            primary = RegionCode(config.primary_region)
        except ValueError as e:
            raise ValueError(f"Invalid primary region: {config.primary_region}") from e

        allowed = []
        for region_str in config.allowed_regions:
            try:
                allowed.append(RegionCode(region_str))
            except ValueError:
                logger.warning(f"Ignoring invalid region: {region_str}")

        data_residency = None
        if config.data_residency:
            try:
                data_residency = RegionCode(config.data_residency)
            except ValueError as e:
                raise ValueError(f"Invalid data residency region: {config.data_residency}") from e

        assignment = TenantRegionAssignment(
            tenant_id=tenant_id,
            primary_region=primary,
            allowed_regions=allowed,
            data_residency=data_residency,
            routing_policy=config.routing_policy,
        )

        # Store assignment
        import json

        assignment_key = f"region:tenant:{tenant_id}"
        assignment_data = {
            "tenant_id": str(tenant_id),
            "primary_region": primary.value,
            "allowed_regions": [r.value for r in allowed],
            "data_residency": data_residency.value if data_residency else None,
            "routing_policy": config.routing_policy,
            "assigned_at": assignment.assigned_at.isoformat(),
        }

        await self.redis.set(assignment_key, json.dumps(assignment_data))

        logger.info(f"Tenant {tenant_id} assigned to region {primary.value}")
        return assignment

    async def get_tenant_region(self, tenant_id: UUID) -> TenantRegionAssignment | None:
        """Get tenant's region assignment."""
        assignment_key = f"region:tenant:{tenant_id}"
        data = await self.redis.get(assignment_key)

        if not data:
            return None

        import json

        assignment_data = json.loads(data)

        return TenantRegionAssignment(
            tenant_id=UUID(assignment_data["tenant_id"]),
            primary_region=RegionCode(assignment_data["primary_region"]),
            allowed_regions=[RegionCode(r) for r in assignment_data.get("allowed_regions", [])],
            data_residency=RegionCode(assignment_data["data_residency"])
            if assignment_data.get("data_residency")
            else None,
            routing_policy=assignment_data.get("routing_policy", "latency"),
            assigned_at=datetime.fromisoformat(assignment_data["assigned_at"]),
        )

    # --------------------------------------------------------
    # ROUTING
    # --------------------------------------------------------

    async def get_routing_endpoint(
        self,
        tenant_id: UUID,
        client_ip: str | None = None,
    ) -> tuple[Region, str]:
        """
        Get the best endpoint for a tenant request.

        Args:
            tenant_id: Tenant ID
            client_ip: Client IP for latency-based routing

        Returns:
            (Region, endpoint_url)
        """
        assignment = await self.get_tenant_region(tenant_id)

        if not assignment:
            # Default to primary region
            primary = REGIONS[RegionCode.US_EAST]
            return (primary, primary.api_endpoint or "")

        # Get primary region
        primary = REGIONS.get(assignment.primary_region)
        if primary and primary.status != RegionStatus.ACTIVE and primary.failover_region:
            # Failover
            primary = REGIONS.get(primary.failover_region)
        elif not primary:
            primary = None  # will be caught below

        if not primary:
            primary = REGIONS[RegionCode.US_EAST]

        # Apply routing policy
        if assignment.routing_policy == "latency" and client_ip:
            # Find nearest active region
            nearest = await self._find_nearest_region(
                client_ip,
                [assignment.primary_region] + assignment.allowed_regions,
            )
            if nearest:
                primary = nearest

        return (primary, primary.api_endpoint or "")

    async def _find_nearest_region(
        self,
        client_ip: str,
        allowed_regions: list[RegionCode],
    ) -> Region | None:
        """Find nearest region based on IP geolocation."""
        # Simplified - would use IP geolocation service in production
        # For now, just return the first active allowed region

        for region_code in allowed_regions:
            region = REGIONS.get(region_code)
            if region and region.status == RegionStatus.ACTIVE:
                return region

        return None

    def calculate_distance(
        self,
        lat1: float,
        lon1: float,
        lat2: float,
        lon2: float,
    ) -> float:
        """
        Calculate distance between two points using Haversine formula.

        Returns distance in kilometers.
        """
        import math

        earth_radius_km = 6371

        lat1_rad = math.radians(lat1)
        lat2_rad = math.radians(lat2)
        delta_lat = math.radians(lat2 - lat1)
        delta_lon = math.radians(lon2 - lon1)

        a = (
            math.sin(delta_lat / 2) ** 2
            + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(delta_lon / 2) ** 2
        )
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

        return earth_radius_km * c

    # --------------------------------------------------------
    # REGION STATUS
    # --------------------------------------------------------

    async def update_region_status(
        self,
        region_code: RegionCode,
        status: RegionStatus,
    ) -> None:
        """Update a region's status."""
        if region_code in REGIONS:
            REGIONS[region_code].status = status

            # Store in Redis for distributed access
            status_key = f"region:status:{region_code.value}"
            await self.redis.setex(status_key, 300, status.value)

            logger.info(f"Region {region_code.value} status updated to {status.value}")

    async def get_region_status(self, region_code: RegionCode) -> RegionStatus:
        """Get a region's current status."""
        # Check Redis first
        status_key = f"region:status:{region_code.value}"
        cached_status = await self.redis.get(status_key)

        if cached_status:
            return RegionStatus(
                cached_status.decode() if isinstance(cached_status, bytes) else cached_status
            )

        # Fall back to in-memory
        region = REGIONS.get(region_code)
        return region.status if region else RegionStatus.OFFLINE


# ============================================================
# GLOBAL INSTANCE
# ============================================================

_region_service: RegionService | None = None


async def get_region_service() -> RegionService | None:
    """Get the global region service."""
    global _region_service

    if _region_service is not None:
        return _region_service

    try:
        from ..data.postgres import get_database
        from ..data.redis import get_redis

        redis = get_redis()
        db = get_database()

        _region_service = RegionService(redis.client, db)
        logger.info("Region service initialized")

        return _region_service

    except Exception as e:
        logger.error(f"Failed to initialize region service: {e}")
        return None


def reset_region_service() -> None:
    """Reset the global region service (for testing)."""
    global _region_service
    _region_service = None


# ============================================================
# EXPORTS
# ============================================================

__all__ = [
    # Types
    "RegionCode",
    "RegionStatus",
    "Region",
    "RegionConfig",
    "TenantRegionAssignment",
    # Registry
    "REGIONS",
    # Service
    "RegionService",
    "get_region_service",
    "reset_region_service",
]
