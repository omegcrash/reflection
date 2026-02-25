# Reflection - Enterprise Multi-Tenant AI Companion Platform
# Copyright (c) 2026 George Scott Foley
# ORCID: 0009-0006-4957-0540
# Email: Georgescottfoley@proton.me
# Licensed under the MIT License - see LICENSE file for details

"""
Example: Healthcare Organization with HIPAA Compliance

HIPAA routing is driven by a single tenant setting: hipaa_compliant=True

When enabled, the platform automatically:
- Activates PHI/PII detection on all messages
- Routes PHI to self-hosted provider (Ollama)
- Routes non-PHI to API provider (if configured)
- Enables HIPAA audit logging
- Enforces 6-year audit log retention
- Blocks permissive security mode

No manual routing configuration needed — set the flag and go.
"""

from uuid import uuid4
from reflection.tenant_wrappers import TenantAgent


# ============================================================
# Example 1: Simple HIPAA tenant (all traffic self-hosted)
# ============================================================

simple_config = {
    "agent_name": "Healthcare Assistant",
    "persona": "Professional medical assistant",
    
    # This is all you need — everything else is automatic
    "hipaa_compliant": True,
}

agent = TenantAgent(tenant_id=uuid4(), tenant_config=simple_config)

# PHI is automatically detected and routed to Ollama
response = agent.chat("Patient MRN: 12345 has Type 2 Diabetes")
# → PHI detected (MRN, diagnosis) → routed to ollama/llama3.2

# General questions also go to Ollama (safe by default)
response = agent.chat("What are the symptoms of diabetes?")
# → No PHI detected → still routes to ollama (no general provider configured)


# ============================================================
# Example 2: Hybrid HIPAA tenant (PHI self-hosted, general to API)
# ============================================================

hybrid_config = {
    "agent_name": "Clinic AI",
    "persona": "Healthcare operations assistant",
    
    # Enable HIPAA compliance
    "hipaa_compliant": True,
    
    # Self-hosted for PHI (defaults to "ollama" if not specified)
    "phi_provider_name": "ollama",
    "phi_model": "qwen2.5:7b",
    
    # API for non-PHI general tasks (faster, better quality)
    "general_provider_name": "anthropic",
    "general_model": "claude-sonnet-4-20250514",
}

agent = TenantAgent(tenant_id=uuid4(), tenant_config=hybrid_config)

# PHI → self-hosted
response = agent.chat("Patient MRN: 12345 has Type 2 Diabetes")
# → PHI detected → routed to ollama/qwen2.5:7b

# General → API (faster, better quality)
response = agent.chat("What are the symptoms of diabetes?")
# → No PHI → routed to anthropic/claude-sonnet-4-20250514

# Manual PHI tag (user override for sensitive workflows)
response = agent.chat(
    "Uploading patient chart from EHR...",
    contains_phi=True,
)
# → Forced to self-hosted regardless of detection

# Manual non-PHI tag (user override for known-safe content)
response = agent.chat(
    "Analyze this anonymized dataset",
    contains_phi=False,
)
# → Forced to API regardless of detection


# ============================================================
# Example 3: Non-HIPAA tenant (no routing overhead)
# ============================================================

standard_config = {
    "agent_name": "Business Assistant",
    "default_provider": "anthropic",
    # hipaa_compliant defaults to False — no routing, no PHI detection
}

agent = TenantAgent(tenant_id=uuid4(), tenant_config=standard_config)

# All traffic goes directly to the configured provider
# No PHI scanning, no routing overhead
response = agent.chat("Summarize our Q3 results")


# ============================================================
# Example 4: Using TenantConfig dataclass directly
# ============================================================

from reflection.tenants.models import TenantConfig

config = TenantConfig(
    hipaa_compliant=True,
    phi_provider_name="ollama",
    phi_model="qwen2.5:7b",
    general_provider_name="anthropic",
    general_model="claude-sonnet-4-20250514",
)

# Validation is automatic
errors = config.validate_hipaa()
assert errors == [], f"HIPAA validation failed: {errors}"

# audit_log_retention_days was auto-raised to 2190 (6 years)
assert config.audit_log_retention_days == 2190

# Pass as dict to TenantAgent
agent = TenantAgent(
    tenant_id=uuid4(),
    tenant_config=config.to_dict(),
)
