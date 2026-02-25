#!/usr/bin/env python3
# Reflection - Enterprise Multi-Tenant AI Companion Platform
# Copyright (c) 2026 George Scott Foley
# ORCID: 0009-0006-4957-0540
# Email: Georgescottfoley@proton.me
# Licensed under the MIT License - see LICENSE file for details

"""
Reflection - Quick Start Example

This example demonstrates the core Reflection capabilities:
1. Creating a tenant
2. Setting up an async agent
3. Executing chat with tool use
4. Parallel tool execution

Run with:
    python examples/quickstart.py
"""

import asyncio
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


async def main():
    """Demonstrate Reflection capabilities."""
    
    print("=" * 60)
    print("🐍 Reflection - Quick Start Example")
    print("=" * 60)
    print()
    
    # --------------------------------------------------------
    # 1. TENANT MANAGEMENT
    # --------------------------------------------------------
    print("1️⃣  Creating a Tenant")
    print("-" * 40)
    
    from reflection.tenants import (
        Tenant, TenantTier, TenantStatus,
        tenant_context, TenantQuotas
    )
    
    tenant = Tenant.create(
        name="Demo Organization",
        slug="demo-org",
        tier=TenantTier.PROFESSIONAL,
    )
    tenant.activate()
    
    print(f"   Tenant ID:    {tenant.id}")
    print(f"   Name:         {tenant.name}")
    print(f"   Tier:         {tenant.tier.value}")
    print(f"   Status:       {tenant.status.value}")
    print(f"   Max Users:    {tenant.quotas.max_users}")
    print(f"   Daily Tokens: {tenant.quotas.max_tokens_per_day:,}")
    print()
    
    # --------------------------------------------------------
    # 2. TENANT CONTEXT
    # --------------------------------------------------------
    print("2️⃣  Using Tenant Context")
    print("-" * 40)
    
    async with tenant_context(tenant, user_id="user_demo") as ctx:
        print(f"   Tenant ID:  {ctx.tenant_id}")
        print(f"   User ID:    {ctx.user_id}")
        print(f"   Request ID: {ctx.request_id}")
        print()
        
        # --------------------------------------------------------
        # 3. ASYNC TOOL REGISTRY
        # --------------------------------------------------------
        print("3️⃣  Async Tool Registry")
        print("-" * 40)
        
        from reflection.core.tools import AsyncToolRegistry
        from reflection_core import Capability, TrustLevel
        
        registry = AsyncToolRegistry()
        
        # Register a custom async tool
        @registry.register(
            name="calculate",
            description="Perform a calculation",
            parameters={
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "Math expression to evaluate"
                    }
                },
                "required": ["expression"]
            },
        )
        async def calculate(expression: str) -> str:
            # Safe eval for simple math
            allowed = set("0123456789+-*/.() ")
            if all(c in allowed for c in expression):
                try:
                    result = eval(expression)
                    return f"{expression} = {result}"
                except Exception as e:
                    return f"Error: {e}"
            return "Invalid expression"
        
        # List available tools
        tools = registry.list_tools()
        print(f"   Registered tools: {len(tools)}")
        for tool in tools:
            print(f"     - {tool.name}: {tool.description[:40]}...")
        print()
        
        # Execute a tool
        print("   Executing calculate tool...")
        result = await registry.execute(
            "calculate",
            {"expression": "2 + 2 * 10"},
            user_trust=TrustLevel.TRUSTED,
        )
        print(f"   Result: {result.output}")
        print(f"   Execution time: {result.execution_time_ms:.2f}ms")
        print()
        
        # --------------------------------------------------------
        # 4. PARALLEL TOOL EXECUTION
        # --------------------------------------------------------
        print("4️⃣  Parallel Tool Execution")
        print("-" * 40)
        
        import time
        
        # Sequential execution
        start = time.time()
        for expr in ["1+1", "2*2", "3**3"]:
            await registry.execute("calculate", {"expression": expr})
        sequential_time = time.time() - start
        
        # Parallel execution
        start = time.time()
        results = await registry.execute_parallel([
            ("calculate", {"expression": "1+1"}),
            ("calculate", {"expression": "2*2"}),
            ("calculate", {"expression": "3**3"}),
        ])
        parallel_time = time.time() - start
        
        print(f"   Sequential: {sequential_time*1000:.2f}ms")
        print(f"   Parallel:   {parallel_time*1000:.2f}ms")
        print(f"   Speedup:    {sequential_time/parallel_time:.1f}x")
        print()
        
        for r in results:
            print(f"   → {r.output}")
        print()
        
        # --------------------------------------------------------
        # 5. LLM PROVIDER (if API key available)
        # --------------------------------------------------------
        print("5️⃣  LLM Provider")
        print("-" * 40)
        
        anthropic_key = os.getenv("LLM_ANTHROPIC_API_KEY") or os.getenv("ANTHROPIC_API_KEY")
        
        if anthropic_key:
            from reflection.core.providers import AsyncAnthropicProvider
            
            provider = AsyncAnthropicProvider(
                api_key=anthropic_key,
                model="claude-sonnet-4-20250514",
            )
            
            print("   Sending message to Claude...")
            
            response = await provider.chat(
                messages=[{
                    "role": "user",
                    "content": "In one sentence, what is Reflection?"
                }],
                max_tokens=100,
            )
            
            print(f"   Response: {response.text}")
            print(f"   Tokens: {response.usage.total_tokens}")
            
            # Track usage in context
            ctx.track_tokens(
                response.usage.input_tokens,
                response.usage.output_tokens
            )
            
            await provider.close()
        else:
            print("   ⚠️  No ANTHROPIC_API_KEY set, skipping LLM test")
            print("      Set LLM_ANTHROPIC_API_KEY in .env to test")
        
        print()
        
        # --------------------------------------------------------
        # 6. USAGE TRACKING
        # --------------------------------------------------------
        print("6️⃣  Usage Tracking")
        print("-" * 40)
        print(f"   Tokens used:     {ctx.usage.total_tokens}")
        print(f"   LLM calls:       {ctx.usage.llm_calls}")
        print(f"   Tool executions: {ctx.usage.tool_executions}")
    
    print()
    print("=" * 60)
    print("✅ Reflection Quick Start Complete!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
