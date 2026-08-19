# -*- mode: python ; coding: utf-8 -*-


a = Analysis(
    ['run_realtime.py'],
    pathex=['D:/MVS/Development/Samples/Python/MvImport'],
    binaries=[],
    datas=[('config/realtime.json', 'config')],
    hiddenimports=['MvCameraControl_class', 'PixelType_header', 'CameraParams_const', 'CameraParams_header', 'MvErrorDefine_const', 'MvISPErrorDefine_const'],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='HikCounter',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
