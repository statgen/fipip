# fipip/cli.py
import os, sys, importlib

def main():
    # discover available subcommand modules under fipip/scripts
    scripts_dir = os.path.join(os.path.dirname(__file__), "scripts")
    module_list = []
    for fname in os.listdir(scripts_dir):
        if fname.endswith(".py") and fname != "__init__.py":
            module_list.append(os.path.splitext(fname)[0])  # e.g. "predict_from_json"

    module_map = {m: m for m in module_list}  # command -> module

    # usage & command parsing
    if len(sys.argv) < 2:
        print("Usage: fipip <command> <args>   (use: fipip <command> -h for help)")
        print("Available commands:\n\t" + "\n\t".join(sorted(module_map.keys())))
        return

    cmd_raw = sys.argv[1]
    cmd = cmd_raw.replace("-", "_")  # allow hyphens as aliases

    if cmd not in module_map:
        print(f"Unknown command: {cmd_raw}")
        print("Available commands:\n\t" + "\n\t".join(sorted(module_map.keys())))
        return

    module_name = "fipip.scripts." + module_map[cmd]
    module = importlib.import_module(module_name)

    # Prefer a function named after the command; fallback to main()
    if hasattr(module, cmd):
        func = getattr(module, cmd)
    elif hasattr(module, "main"):
        func = getattr(module, "main")
    else:
        print(f"The module '{module_name}' does not expose '{cmd}()' or main().")
        return

    # forward remaining argv to the subcommand
    func(sys.argv[2:])
