import os, sys, logging, importlib
import logging

def main():
    module_list = []
    scripts_dir = os.path.join(os.path.dirname(__file__), 'scripts')
    for f in os.listdir(scripts_dir):
        if f.endswith(".py") and f != "__init__.py":
            module_list.append(os.path.splitext(f)[0])

    module_map = {m:m for m in module_list}

    if len(sys.argv) < 2:
        print("Usage: fipip <command> <args>, fipip <command> -h to see arguments for each command")
        print("Available commands:\n\t"+"\n\t".join(sorted(list(module_map.keys()) )))
        return
    elif sys.argv[1] not in module_map:
        print("Unknown command: "+sys.argv[1])
        print("Available commands:\n\t"+"\n\t".join(sorted(list(module_map.keys()) )))
        return

    function_name = sys.argv[1]
    module_name = "fipip.scripts." + module_map[function_name]
    module = importlib.import_module(module_name)
    function = getattr(module, function_name)

    function(sys.argv[2:])
