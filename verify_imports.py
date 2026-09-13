import importlib
import sys

root = r"c:\Users\HP\OneDrive\Documents\Stock Projects\Dark Wing"
sys.path.insert(0, root)
mods = ["Trade_Implement.Signals.pair_signal", "Trade_Implement.Signals.multi_pair_signal"]
for mod_name in mods:
    try:
        mod = importlib.import_module(mod_name)
        print(f"OK {mod_name}: {getattr(mod, '__file__', None)}")
    except Exception as exc:
        print(f"FAIL {mod_name}: {exc}")
