import ast
import os
from pathlib import Path
from importlib import metadata


# ---------- CONFIG ----------
# Change this if you want to scan a different folder
PROJECT_ROOT = Path(__file__).resolve().parent.parent  # repo root (.. from tools/)
EXCLUDE_DIRS = {"venv", ".git", ".idea", ".vscode", "__pycache__"}
# -----------------------------


def iter_python_files(root: Path):
    """Yield all .py files under root, excluding EXCLUDE_DIRS."""
    for dirpath, dirnames, filenames in os.walk(root):
        # Skip excluded dirs
        dirnames[:] = [d for d in dirnames if d not in EXCLUDE_DIRS]

        for fname in filenames:
            if fname.endswith(".py"):
                yield Path(dirpath) / fname


def collect_imports(root: Path):
    """
    Parse all .py files and collect top-level imported module names
    (e.g. 'requests', 'torch', 'numpy').
    """
    imported_modules = set()

    for py_file in iter_python_files(root):
        try:
            with open(py_file, "r", encoding="utf-8") as f:
                source = f.read()
        except (UnicodeDecodeError, OSError):
            # Skip unreadable files
            continue

        try:
            tree = ast.parse(source, filename=str(py_file))
        except SyntaxError:
            # Skip files with syntax errors
            continue

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    # import x.y.z -> "x"
                    top_level = alias.name.split(".")[0]
                    imported_modules.add(top_level)
            elif isinstance(node, ast.ImportFrom):
                # from x.y import z -> "x"
                if node.module is not None:
                    top_level = node.module.split(".")[0]
                    imported_modules.add(top_level)

    return imported_modules


def map_modules_to_distributions(imported_modules):
    """
    Map imported module names to installed distributions using
    importlib.metadata.packages_distributions().
    """
    try:
        pkg_map = metadata.packages_distributions()
    except Exception:
        # Fallback: build map manually (less accurate, but better than nothing)
        pkg_map = {}
        for dist in metadata.distributions():
            name = dist.metadata["Name"]
            try:
                top_levels = dist.read_text("top_level.txt")
            except FileNotFoundError:
                continue
            if not top_levels:
                continue
            for line in top_levels.splitlines():
                mod = line.strip()
                if not mod:
                    continue
                pkg_map.setdefault(mod, set()).add(name)

    module_to_dists = {}
    for mod in imported_modules:
        dists = pkg_map.get(mod, [])
        # ensure it's a list
        if isinstance(dists, (set, tuple)):
            dists = list(dists)
        module_to_dists[mod] = dists

    return module_to_dists


def get_installed_distributions():
    """Return a dict: {dist_name_lower: dist} for all installed packages."""
    installed = {}
    for dist in metadata.distributions():
        name = dist.metadata.get("Name")
        if not name:
            continue
        installed[name.lower()] = dist
    return installed


def main():
    print(f"Scanning project root: {PROJECT_ROOT}")
    imported_modules = collect_imports(PROJECT_ROOT)
    print(f"Found {len(imported_modules)} imported top-level modules in project.")

    module_to_dists = map_modules_to_distributions(imported_modules)
    installed = get_installed_distributions()

    used_distributions = set()
    for mod, dists in module_to_dists.items():
        for dist_name in dists:
            used_distributions.add(dist_name.lower())

    # Only consider packages that are actually installed via pip in this venv
    installed_names = set(installed.keys())

    # Some imported modules might be stdlib (no dist) -> they won't appear here
    used_pip_packages = installed_names & used_distributions
    unused_pip_packages = sorted(installed_names - used_pip_packages)

    print("\n=== SUMMARY ===")
    print(f"Installed packages in this venv: {len(installed_names)}")
    print(f"Packages used in your code (mapped from imports): {len(used_pip_packages)}")
    print(f"Potentially unused packages: {len(unused_pip_packages)}")

    if unused_pip_packages:
        print("\nPotentially UNUSED packages (double-check before uninstalling):")
        for name in unused_pip_packages:
            print("  -", name)

        # Write to text file
        out_txt = PROJECT_ROOT / "unused_packages.txt"
        with open(out_txt, "w", encoding="utf-8") as f:
            for name in unused_pip_packages:
                f.write(name + "\n")

        # Generate PowerShell uninstall script
        uninstall_script = PROJECT_ROOT / "uninstall_unused.ps1"
        with open(uninstall_script, "w", encoding="utf-8") as f:
            f.write("# Auto-generated script to uninstall potentially unused packages\n")
            f.write("# REVIEW this list before running.\n\n")
            for name in unused_pip_packages:
                f.write(f"pip uninstall -y {name}\n")

        print(f"\nSaved list to: {out_txt}")
        print(f"Generated PowerShell uninstall script: {uninstall_script}")
        print(">>> Review these before running the uninstall script.")
    else:
        print("\nNo unused packages detected (or everything imported maps to an installed dist).")


if __name__ == "__main__":
    main()
