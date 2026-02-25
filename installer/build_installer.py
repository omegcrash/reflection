#!/usr/bin/env python3
"""
Reflection Installer Build Script

Builds the graphical installer for all platforms.
Usage: python build_installer.py [--platform windows|macos|linux|all]
"""

import subprocess
import sys
import platform
import argparse
from pathlib import Path


def build_windows():
    """Build Windows .exe installer."""
    print("🪟 Building Windows installer...")
    
    subprocess.run([
        "pyinstaller",
        "installer.spec",
        "--clean",
        "--noconfirm"
    ], check=True)
    
    print("\n✓ Windows installer built:")
    print("   dist/ReflectionSetup.exe")
    print("\nNext steps:")
    print("  1. Test: dist\\ReflectionSetup.exe")
    print("  2. Sign (optional): signtool sign /f cert.pfx dist\\ReflectionSetup.exe")
    print("  3. Create installer with NSIS (optional)")


def build_macos():
    """Build macOS .app installer."""
    print("🍎 Building macOS installer...")
    
    subprocess.run([
        "pyinstaller",
        "installer.spec",
        "--clean",
        "--noconfirm",
        "--windowed"
    ], check=True)
    
    print("\n✓ macOS installer built:")
    print("   dist/ReflectionSetup.app")
    print("\nNext steps:")
    print("  1. Test: open dist/ReflectionSetup.app")
    print("  2. Sign (optional): codesign --deep --force --sign 'Developer ID' dist/ReflectionSetup.app")
    print("  3. Create DMG:")
    print("     hdiutil create -volname 'Reflection' -srcfolder dist/ReflectionSetup.app -ov -format UDZO Reflection.dmg")


def build_linux():
    """Build Linux executable."""
    print("🐧 Building Linux installer...")
    
    subprocess.run([
        "pyinstaller",
        "installer.spec",
        "--clean",
        "--noconfirm"
    ], check=True)
    
    print("\n✓ Linux installer built:")
    print("   dist/ReflectionSetup")
    print("\nNext steps:")
    print("  1. Test: ./dist/ReflectionSetup")
    print("  2. Create AppImage (optional):")
    print("     appimagetool dist/ Reflection-x86_64.AppImage")
    print("  3. Create .tar.gz:")
    print("     tar -czf Reflection-Linux.tar.gz -C dist .")


def main():
    parser = argparse.ArgumentParser(description="Build Reflection installer")
    parser.add_argument(
        "--platform",
        choices=["windows", "macos", "linux", "all"],
        default="auto",
        help="Target platform (default: auto-detect)"
    )
    
    args = parser.parse_args()
    
    # Change to installer directory
    script_dir = Path(__file__).parent
    import os
    os.chdir(script_dir)
    
    print("╔══════════════════════════════════════════╗")
    print("║  Reflection Installer Builder       ║")
    print("╚══════════════════════════════════════════╝\n")
    
    # Auto-detect platform if needed
    if args.platform == "auto":
        system = platform.system()
        if system == "Windows":
            args.platform = "windows"
        elif system == "Darwin":
            args.platform = "macos"
        else:
            args.platform = "linux"
        print(f"Auto-detected platform: {args.platform}\n")
    
    # Build
    if args.platform == "all":
        print("Building for all platforms...\n")
        build_windows()
        build_macos()
        build_linux()
    elif args.platform == "windows":
        build_windows()
    elif args.platform == "macos":
        build_macos()
    elif args.platform == "linux":
        build_linux()
    
    print("\n✅ Build complete!")


if __name__ == "__main__":
    main()
