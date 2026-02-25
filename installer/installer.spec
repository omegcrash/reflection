# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller spec file for Reflection Installer

Build instructions:
  Windows: pyinstaller installer.spec
  macOS:   pyinstaller installer.spec --windowed
  Linux:   pyinstaller installer.spec

Output: dist/ReflectionSetup.exe (or .app, or executable)
"""

block_cipher = None

a = Analysis(
    ['gui_installer.py'],
    pathex=[],
    binaries=[],
    datas=[
        # Include Reflection files
        ('../reflection', 'reflection'),
        ('../familiar', 'familiar'),
        ('../docker-compose.yml', '.'),
        ('../pyproject.toml', '.'),
        ('../README.md', '.'),
    ],
    hiddenimports=[
        'tkinter',
        'tkinter.ttk',
        'tkinter.messagebox',
        'tkinter.scrolledtext',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='ReflectionSetup',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,  # No console window
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon='../docs/icon.ico' if os.path.exists('../docs/icon.ico') else None,
)

# macOS app bundle
app = BUNDLE(
    exe,
    name='ReflectionSetup.app',
    icon='../docs/icon.icns' if os.path.exists('../docs/icon.icns') else None,
    bundle_identifier='ai.familiar.mother.installer',
    info_plist={
        'NSHighResolutionCapable': 'True',
        'NSRequiresAquaSystemAppearance': 'False',
    },
)
