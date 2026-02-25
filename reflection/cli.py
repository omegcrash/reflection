# Reflection - Enterprise Multi-Tenant AI Companion Platform
# Copyright (c) 2026 George Scott Foley
# ORCID: 0009-0006-4957-0540
# Email: Georgescottfoley@proton.me
# Licensed under the MIT License - see LICENSE file for details

"""
Command Line Interface

Usage:
    reflection serve              # Start API server
    reflection tenant create      # Create a tenant
    reflection migrate            # Run database migrations
"""

import asyncio

import typer
from rich.console import Console
from rich.table import Table

app = typer.Typer(
    name="reflection", help="Reflection - Enterprise Multi-Tenant AI Companion Platform"
)
console = Console()


# ============================================================
# INSTALLATION COMMANDS
# ============================================================


@app.command()
def install(
    gui: bool = typer.Option(True, help="Use GUI installer (auto-detects display)"),
    use_case: str = typer.Option(None, help="Use case: healthcare, privacy, general"),
    provider: str = typer.Option(None, help="LLM provider: anthropic, openai, ollama"),
    auto: bool = typer.Option(False, help="Non-interactive (requires --use-case and --provider)"),
):
    """
    Interactive installation wizard.

    Launches a GUI or CLI installer that will:
    - Detect your system
    - Ask about your use case (Healthcare/Business/Privacy)
    - Install dependencies (Ollama, Docker, etc.)
    - Configure Reflection
    - Launch the application

    Examples:
        reflection install              # GUI (auto-detect)
        reflection install --no-gui     # CLI / terminal
        reflection install --use-case healthcare --provider ollama --auto
    """
    import subprocess
    import sys
    from pathlib import Path

    project_root = Path(__file__).parent.parent

    if gui:
        # Try GUI installer
        wizard_path = project_root / "installer" / "setup_wizard.py"

        if not wizard_path.exists():
            console.print("[red]GUI installer not found![/]")
            console.print(f"Expected at: {wizard_path}")
            console.print("\nFalling back to CLI installer...")
            gui = False
        else:
            # Check tkinter availability
            try:
                import tkinter  # noqa: F401

                console.print("[bold green]Launching GUI installer...[/]")
                subprocess.run([sys.executable, str(wizard_path)])
                return
            except ImportError:
                console.print("[yellow]tkinter not available, using CLI installer[/]")
                console.print("Install tkinter: [cyan]sudo apt install python3-tk[/]")
                gui = False

    if not gui:
        # CLI installer
        cli_path = project_root / "installer" / "cli_installer.py"

        if not cli_path.exists():
            console.print("[red]CLI installer not found![/]")
            console.print(f"Expected at: {cli_path}")
            raise typer.Exit(1)

        cmd = [sys.executable, str(cli_path)]
        if use_case:
            cmd.extend(["--use-case", use_case])
        if provider:
            cmd.extend(["--provider", provider])
        if auto:
            cmd.append("--auto")

        subprocess.run(cmd)


# ============================================================
# SERVER COMMANDS
# ============================================================


@app.command()
def serve(
    host: str = typer.Option("0.0.0.0", help="Host to bind to"),
    port: int = typer.Option(8000, help="Port to bind to"),
    workers: int = typer.Option(4, help="Number of worker processes"),
    reload: bool = typer.Option(False, help="Enable auto-reload (development)"),
):
    """Start the API server."""
    import uvicorn

    console.print(f"[bold green]Starting Reflection on {host}:{port}[/]")

    uvicorn.run(
        "reflection.gateway.app:app",
        host=host,
        port=port,
        workers=workers if not reload else 1,
        reload=reload,
    )


# ============================================================
# TENANT COMMANDS
# ============================================================

tenant_app = typer.Typer(help="Tenant management commands")
app.add_typer(tenant_app, name="tenant")


@tenant_app.command("create")
def tenant_create(
    name: str = typer.Option(..., prompt=True, help="Tenant name"),
    slug: str = typer.Option(..., prompt=True, help="Tenant slug (URL-safe)"),
    tier: str = typer.Option("free", help="Subscription tier"),
    admin_email: str = typer.Option(None, help="Admin user email"),
):
    """Create a new tenant."""

    async def _create():
        from .data.postgres import get_db_session, init_database
        from .data.repositories import TenantRepository, TenantUserRepository

        await init_database()

        async for session in get_db_session():
            tenant_repo = TenantRepository(session)

            # Check slug uniqueness
            existing = await tenant_repo.get_by_slug(slug)
            if existing:
                console.print(f"[red]Tenant with slug '{slug}' already exists.[/]")
                raise typer.Exit(code=1)

            tenant = await tenant_repo.create(
                name=name,
                slug=slug,
                tier=tier,
            )

            table = Table(title="Tenant Created")
            table.add_column("Field", style="cyan")
            table.add_column("Value", style="white")
            table.add_row("ID", str(tenant.id))
            table.add_row("Name", tenant.name)
            table.add_row("Slug", tenant.slug)
            table.add_row("Tier", tenant.tier)
            table.add_row("Status", tenant.status)
            console.print(table)

            # Create admin user if email provided
            if admin_email:
                user_repo = TenantUserRepository(session)
                user = await user_repo.create(
                    tenant_id=tenant.id,
                    email=admin_email,
                    role="admin",
                )
                console.print(f"\n[green]Admin user created:[/] {admin_email} (ID: {user.id})")
                console.print("[yellow]Set a password with: reflection tenant set-password[/]")

    asyncio.run(_create())


@tenant_app.command("list")
def tenant_list():
    """List all tenants."""

    async def _list():
        from .data.postgres import get_db_session, init_database
        from .data.repositories import TenantRepository

        await init_database()

        async for session in get_db_session():
            repo = TenantRepository(session)
            tenants = await repo.list_active()

            if not tenants:
                console.print("[yellow]No tenants found.[/]")
                console.print("Create one with: [bold]reflection tenant create[/]")
                return

            table = Table(title=f"Tenants ({len(tenants)})")
            table.add_column("Slug", style="cyan")
            table.add_column("Name", style="white")
            table.add_column("Tier", style="green")
            table.add_column("Status", style="white")
            table.add_column("ID", style="dim")

            for t in tenants:
                table.add_row(t.slug, t.name, t.tier, t.status, str(t.id))

            console.print(table)

    asyncio.run(_list())


# ============================================================
# DATABASE COMMANDS
# ============================================================

db_app = typer.Typer(help="Database management commands")
app.add_typer(db_app, name="db")


@db_app.command("migrate")
def db_migrate(
    revision: str = typer.Option("head", help="Target revision (default: head)"),
):
    """Run database migrations using Alembic."""
    import subprocess
    import sys

    console.print("[bold]Running migrations...[/]")

    result = subprocess.run(
        [sys.executable, "-m", "alembic", "upgrade", revision],
        capture_output=True,
        text=True,
    )

    if result.returncode == 0:
        console.print("[green]Migrations completed successfully![/]")
        if result.stdout:
            console.print(result.stdout)
    else:
        console.print("[red]Migration failed![/]")
        if result.stderr:
            console.print(result.stderr)
        raise typer.Exit(1)


@db_app.command("downgrade")
def db_downgrade(
    revision: str = typer.Option(
        ..., prompt=True, help="Target revision (e.g., -1 for one step back)"
    ),
):
    """Downgrade database to a specific revision."""
    import subprocess
    import sys

    console.print(f"[bold yellow]Downgrading to revision: {revision}[/]")

    result = subprocess.run(
        [sys.executable, "-m", "alembic", "downgrade", revision],
        capture_output=True,
        text=True,
    )

    if result.returncode == 0:
        console.print("[green]Downgrade completed successfully![/]")
        if result.stdout:
            console.print(result.stdout)
    else:
        console.print("[red]Downgrade failed![/]")
        if result.stderr:
            console.print(result.stderr)
        raise typer.Exit(1)


@db_app.command("current")
def db_current():
    """Show current database revision."""
    import subprocess
    import sys

    result = subprocess.run(
        [sys.executable, "-m", "alembic", "current"],
        capture_output=True,
        text=True,
    )

    if result.stdout:
        console.print(result.stdout)
    if result.stderr:
        console.print(result.stderr)


@db_app.command("history")
def db_history():
    """Show migration history."""
    import subprocess
    import sys

    result = subprocess.run(
        [sys.executable, "-m", "alembic", "history", "--verbose"],
        capture_output=True,
        text=True,
    )

    if result.stdout:
        console.print(result.stdout)
    if result.stderr:
        console.print(result.stderr)


@db_app.command("revision")
def db_revision(
    message: str = typer.Option(..., "-m", "--message", prompt=True, help="Migration message"),
    autogenerate: bool = typer.Option(True, help="Auto-generate from model changes"),
):
    """Create a new migration revision."""
    import subprocess
    import sys

    console.print(f"[bold]Creating migration: {message}[/]")

    cmd = [sys.executable, "-m", "alembic", "revision", "-m", message]
    if autogenerate:
        cmd.append("--autogenerate")

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode == 0:
        console.print("[green]Migration created successfully![/]")
        if result.stdout:
            console.print(result.stdout)
    else:
        console.print("[red]Failed to create migration![/]")
        if result.stderr:
            console.print(result.stderr)
        raise typer.Exit(1)


@db_app.command("reset")
def db_reset(
    force: bool = typer.Option(False, "--force", "-f", help="Skip confirmation"),
):
    """Reset the database (DESTRUCTIVE!)."""
    import subprocess
    import sys

    if not force:
        confirm = typer.confirm("This will delete all data. Are you sure?")
        if not confirm:
            raise typer.Abort()

    console.print("[bold red]Resetting database...[/]")

    # Downgrade to base
    result = subprocess.run(
        [sys.executable, "-m", "alembic", "downgrade", "base"],
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        console.print("[red]Downgrade failed![/]")
        if result.stderr:
            console.print(result.stderr)
        raise typer.Exit(1)

    # Upgrade to head
    result = subprocess.run(
        [sys.executable, "-m", "alembic", "upgrade", "head"],
        capture_output=True,
        text=True,
    )

    if result.returncode == 0:
        console.print("[green]Database reset completed![/]")
    else:
        console.print("[red]Upgrade failed![/]")
        if result.stderr:
            console.print(result.stderr)
        raise typer.Exit(1)


# ============================================================
# UTILITY COMMANDS
# ============================================================


@app.command()
def version():
    """Show version information."""
    from . import __version__

    console.print(f"Reflection v{__version__}")


@app.command()
def generate_key():
    """Generate an encryption key."""
    from reflection_core import generate_key

    key = generate_key().decode()
    console.print("\n[bold green]Generated encryption key:[/]\n")
    console.print(f"  {key}")
    console.print("\n[dim]Add this to your .env file as SECURITY_MASTER_ENCRYPTION_KEY[/]")


@app.command()
def check():
    """Check configuration and dependencies."""
    from .core.settings import get_settings

    console.print("[bold]Checking configuration...[/]\n")

    settings = get_settings()

    checks = []

    # Database
    if "postgresql" in settings.database.url:
        checks.append(("Database URL", "✓", "PostgreSQL configured"))
    else:
        checks.append(("Database URL", "✗", "Not configured"))

    # Redis
    if settings.redis.url:
        checks.append(("Redis URL", "✓", "Configured"))
    else:
        checks.append(("Redis URL", "✗", "Not configured"))

    # LLM Providers
    if settings.llm.anthropic_api_key:
        checks.append(("Anthropic API", "✓", "Key configured"))
    else:
        checks.append(("Anthropic API", "○", "Not configured"))

    if settings.llm.openai_api_key:
        checks.append(("OpenAI API", "✓", "Key configured"))
    else:
        checks.append(("OpenAI API", "○", "Not configured"))

    # Security
    if settings.security.master_encryption_key:
        checks.append(("Encryption Key", "✓", "Configured"))
    else:
        checks.append(("Encryption Key", "✗", "Not configured (run: reflection generate-key)"))

    if settings.security.jwt_secret_key != "CHANGE_ME_IN_PRODUCTION":
        checks.append(("JWT Secret", "✓", "Configured"))
    else:
        checks.append(("JWT Secret", "⚠", "Using default (change in production!)"))

    # Print results
    table = Table(title="Configuration Check")
    table.add_column("Component", style="cyan")
    table.add_column("Status")
    table.add_column("Details", style="dim")

    for name, status, details in checks:
        if status == "✓":
            status_style = "[green]✓[/]"
        elif status == "✗":
            status_style = "[red]✗[/]"
        elif status == "⚠":
            status_style = "[yellow]⚠[/]"
        else:
            status_style = "[dim]○[/]"

        table.add_row(name, status_style, details)

    console.print(table)


# ============================================================
# MAIN
# ============================================================


def main():
    """Main entry point for the CLI."""
    app()


if __name__ == "__main__":
    main()
