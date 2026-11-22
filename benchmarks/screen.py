"""Static screen for selector modules.

A selector may compute only from the batch it receives. This screen rejects the
obvious ways to read files, reach the network, start processes or execute
generated code. It is a tripwire for honest mistakes and shortcuts, not a
security boundary: run untrusted selectors in an isolated environment as well.
"""

import ast

BLOCKED_MODULES = {
    "builtins",
    "ctypes",
    "glob",
    "http",
    "importlib",
    "io",
    "multiprocessing",
    "os",
    "pathlib",
    "pickle",
    "requests",
    "shutil",
    "socket",
    "subprocess",
    "sys",
    "tempfile",
    "urllib",
    "zipfile",
}
BLOCKED_CALLS = {"open", "eval", "exec", "compile", "__import__", "globals", "vars"}
BLOCKED_ATTRIBUTES = {
    "load",
    "loadtxt",
    "genfromtxt",
    "fromfile",
    "memmap",
    "read_csv",
    "save",
    "savez",
    "tofile",
}


def screen(source: str) -> list[str]:
    """Return the reasons a selector source is rejected. Empty means it passes."""
    try:
        tree = ast.parse(source)
    except SyntaxError as error:
        return [f"SyntaxError: {error}"]
    problems = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            names = [node.module or ""]
        else:
            names = []
        for name in names:
            if name.split(".")[0] in BLOCKED_MODULES:
                problems.append(f"line {node.lineno}: imports {name}")
        if isinstance(node, ast.Call):
            function = node.func
            if isinstance(function, ast.Name) and function.id in BLOCKED_CALLS:
                problems.append(f"line {node.lineno}: calls {function.id}()")
            if (
                isinstance(function, ast.Attribute)
                and function.attr in BLOCKED_ATTRIBUTES
            ):
                problems.append(f"line {node.lineno}: calls .{function.attr}()")
    if not any(
        isinstance(node, ast.FunctionDef) and node.name == "select"
        for node in tree.body
    ):
        problems.append("defines no top-level select(batch)")
    return problems
