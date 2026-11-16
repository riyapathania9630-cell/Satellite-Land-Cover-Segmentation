"""
verify_setup.py
===============
Pre-flight check. Run this once after installing requirements:

    python verify_setup.py

It confirms that the Python version, all required libraries, the project
modules, and the model architecture are working, and reports whether
dataset folders contain data.
"""

import importlib
import os
import sys

REQUIRED = ["tensorflow", "numpy", "cv2", "pandas", "matplotlib", "sklearn", "PIL", "tqdm"]
OPTIONAL = ["rasterio"]


def _check(name: str, optional: bool = False) -> bool:
    try:
        module = importlib.import_module(name)
        print(f"  [OK]   {name} {getattr(module, '__version__', '')}")
        return True
    except ImportError:
        tag = "WARN" if optional else "FAIL"
        print(f"  [{tag}] {name} is not installed")
        return optional


def main() -> int:
    ok = True
    print(f"Python {sys.version.split()[0]}")
    if sys.version_info < (3, 9):
        print("  [FAIL] Python 3.9+ is required")
        ok = False

    print("\nLibraries:")
    for pkg in REQUIRED:
        ok &= _check(pkg)
    for pkg in OPTIONAL:
        _check(pkg, optional=True)
    if not ok:
        print("\nSetup incomplete. Run: pip install -r requirements.txt")
        return 1

    print("\nProject modules:")
    for mod in ["config", "dataset", "model", "losses", "metrics", "visualization",
                "utils.helpers", "train", "evaluate", "predict", "change_detection"]:
        try:
            importlib.import_module(mod)
            print(f"  [OK]   {mod}")
        except Exception as exc:  # noqa: BLE001
            print(f"  [FAIL] {mod}: {exc}")
            ok = False

    if ok:
        from model import build_unet
        net = build_unet()
        print(f"\nModel builds correctly ({net.count_params():,} parameters).")

        import config
        print("\nDataset folders:")
        for label, path in [("train", config.TRAIN_IMAGE_DIR), ("val", config.VAL_IMAGE_DIR),
                            ("test", config.TEST_IMAGE_DIR),
                            ("raw", os.path.join(config.RAW_DATA_DIR, "images"))]:
            n = len([f for f in os.listdir(path)
                     if not f.startswith(".")]) if os.path.isdir(path) else 0
            print(f"  {label:6s}: {n} image(s)")
        print("\nIf all folders show 0, add your data or run: python generate_sample_data.py")

    print("\nAll checks passed." if ok else "\nSome checks failed.")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
