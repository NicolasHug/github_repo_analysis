#!/usr/bin/env -S grimaldi --kernel bento_kernel_default
# FILE_UID: 05b7a4cb-6a11-4e26-98b1-2706b6cd9195

""":py '770914644450499'"""
from pathlib import Path
import json
import linecache
import pandas as pd
from joblib import Parallel, delayed
import ast
from collections import defaultdict

""":py"""
# Visitor class taken from https://github.com/fmassa/python-lib-stats/pull/1
class _Visitor(ast.NodeVisitor):
    def __init__(self, hook=None):
        super().__init__()
        self.remapped = {}
        self.called = defaultdict(list)
        self.attrs = {}

        self.import_count = defaultdict(int)
        self.call_count = defaultdict(int)
        self.access_count = defaultdict(int)

        def _noop(*args, **kwargs):
            pass
        self.hook = hook or _noop

    def visit_Import(self, node: ast.AST):
        for n in node.names:
            if n.asname:
                self.remapped[n.asname] = n.name
            else:
                self.remapped[n.name] = n.name
            self.import_count[n.name] += 1
            self.hook(node=node, api_name=n.name, kind="import")

    def visit_ImportFrom(self, node: ast.AST):
        module = node.module
        if module is None:
            module = "{local_import}"
        for n in node.names:
            name = module + '.' + n.name
            if n.asname:
                self.remapped[n.asname] = name
            else:
                self.remapped[n.name] = name

            self.import_count[name] += 1
            self.hook(node=node, api_name=name, kind="import")

    def visit_Call(self, node: ast.AST):
        self.generic_visit(node)
        args = node.args
        if _is_getattr_call(node):
            func = node.func.args[0]
            if func in self.attrs:
                name = self.attrs[func]
                n1 = node.func.args[1]
                v = None
                if n1 in self.attrs:
                    v = "{?}"
                elif isinstance(n1, ast.Constant):
                    v = node.func.args[1].value
                if v is None:
                    # print("Unsupported", ast.dump(n1))
                    pass
                else:
                    name = name + "." + v
                    self.called[name] += args
                    self.call_count[name] += 1
                    self.hook(node=node, api_name=name, kind="call")
                    return

        func = node.func
        if func in self.attrs:
            name = self.attrs[func]
            self.called[name] += args
            self.call_count[name] += 1
            self.hook(node=node, api_name=name, kind="call")
        # all other cases are not supported for now

    def visit_Assign(self, node: ast.AST):
        self.generic_visit(node)
        # easy cases
        if not isinstance(node.targets[0], ast.Name):
            return
        name = node.targets[0].id
        if node.value in self.attrs:
            new_name = self.attrs[node.value]
            self.remapped[name] = new_name

    def visit(self, node: ast.AST):
        if _is_nested_attribute_and_name(node):
            nid, sts = _nested_attribute_and_name(node)
            if nid in self.remapped:
                nid = self.remapped[nid]
            name = ".".join([nid] + sts)
            self.attrs[node] = name
            self.access_count[name] += 1
            self.hook(node=node, api_name=name, kind="access")
            return
        return super().visit(node)


def _is_nested_attribute_and_name(node: ast.AST) -> bool:
    while isinstance(node, ast.Attribute):
        node = node.value
    return isinstance(node, ast.Name)


def _nested_attribute_and_name(node: ast.AST):
    sts = []
    while isinstance(node, ast.Attribute):
        sts.append(node.attr)
        node = node.value
    assert isinstance(node, ast.Name)
    sts = list(reversed(sts))
    return node.id, sts


def _is_getattr_call(base_node: ast.AST) -> bool:
    """
    finds the pattern getattr(mylib, 'const')()
    """
    if not isinstance(base_node, ast.Call):
        return False
    node = base_node.func
    if not isinstance(node, ast.Call):
        return False
    if not (isinstance(node.func, ast.Name) and node.func.id == "getattr"):
        return False
    return True


""":py"""
def _get_file_contents(path) -> str:
    # Taken from https://www.internalfb.com/code/fbsource/[20e1e9b06f25]/fbcode/dataswarm-pipelines/tasks/aml/platform/adoption/external/github_analysis/analysis.py?lines=135
    codec_aliases = [
        "utf-8",
        "latin1",
        "mbcs",
        "cp65001",
        "utf-16",
        "utf-32",
        "us-ascii",
    ]
    for i, codec_alias in enumerate(codec_aliases):
        try:
            with open(path, "r", encoding=codec_alias) as in_file:
                content = in_file.read()
            return content
        except Exception as ex:
            if i >= len(codec_aliases) - 1:
                raise ValueError(f"Couldn't decode {path}") from ex


""":py '1485936142330478'"""
def parse_folder(
    folder: str | dict[str, str],
    code_context: str = "line",
    save_csv : bool | str = False
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Parse all .py files in a folder and return all calls, imports and attribute accesses from ANY library/namespace.

    Example:
    >>> calls, imports, _, = parse_folder("path/to/repo")
    >>> imports
    ...               api_name    kind     permalink                                              code      ...
    ...
    ... 0                torch  import     https://github.com/...                         import torch      ...
    ... 1  torch.nn.functional  import     https://github.com/...      import torch.nn.functional as F      ...
    ...
    >>> calls
    ...               api_name    kind     permalink                                              code      ...
    ... 0        torch.compile    call     https://github.com/...                 torch.compile(model)      ...
    ...

    Args:
        folder(str or dict): A path to the folder, or a dict which must have
            at least the key "folder".
        code_context(str, optional). Defines how accurate the 'code' column will be for calls.
            "line" will give you only the line where the call happend e.g. it could just be "DataLoader(".
            "full" will properly match multiline calls typically up to the corresponding ')', e.g.:
            "DataLoader(
                dataset,
                ...
                pin_memory=True,
            )"
            Default is "line" which is less accurate, but a lot faster.
            Consider running this as a standalone script and not as a notebook if you're using "full".
            Note: this only affects calls, not imports or attribute accesses which are always line-based.
        save_csv (bool, optional): Whether to save the output as csv files in folder/calls.csv, folder/imports.csv,
            and folder/access.csv.

    Returns:
        3 Pandas DataFrames containing all calls, imports and accesses respectively.

        Columns are:
        - api_name e.g. "torch.compile"
        - kind, one of "call", "import", "access"
        - permalink -- only non-empty if `folder` dict contains the keys "url" and "hash".
        - code
        - file_path
        - All keys/values in folder are also saved here
    """

    if isinstance(folder, dict):
        assert "folder" in folder
        repo_dict = folder
        folder = folder["folder"]
    else:
        repo_dict = {"folder": folder}

    folder = Path(folder)


    print(f"Parsing {str(folder)}")

    v = _Visitor()
    apis = defaultdict(list)
    for file_path in folder.glob("**/*.py"):
        try:
            file_contents = _get_file_contents(file_path)
        except Exception as e:
            print(f"Couldn't decode {file_path}")
            print(e)
            continue

        def hook(node, api_name, kind):
            if "url" in repo_dict and "hash" in repo_dict:
                relative_path = file_path.relative_to(folder)
                permalink = f"{repo_dict['url']}/blob/{repo_dict['hash']}/{relative_path}#L{node.lineno}"
            else:
                permalink = ""

            if code_context == "full" and kind == "call":
                code = ast.get_source_segment(file_contents, node)
            else:
                code = linecache.getline(str(file_path), node.lineno).strip("\n")

            apis["api_name"].append(api_name)
            apis["kind"].append(kind)
            apis["permalink"].append(permalink)
            apis["code"].append(code)
            apis["file_path"].append(file_path)
            for k, v in repo_dict.items():
                apis[k].append(v)

        v.hook = hook
        try:
            v.visit(ast.parse(file_contents))
        except Exception as e:
            print(f"Couldn't parse {file_path} because of following exception:")
            print(e)


    if not apis:
        return

    apis = pd.DataFrame.from_dict(apis, dtype=None)

    calls = apis[apis["kind"] == "call"]
    imports = apis[apis["kind"] == "import"]
    access = apis[apis["kind"] == "access"]

    if save_csv:
        print(f"Saving csvs in {str(folder)}")
        for df, name in ((calls, "calls"), (imports, "imports"), (access, "access")):
            dest = folder / f"{name}.csv"
            try:
                df.to_csv(dest)
            except Exception as e:
                print(f"Couldn't save {dest} because of following exception:")
                print(e)
    return calls, imports, access


def is_in_notebook():
    import __main__ as main
    return not hasattr(main, '__file__')


def find_repos(root : str) -> list[dict]:
    repos = []
    for meta_file in Path(root).glob("**/analysis_meta.json"):
        with open(meta_file) as f:
            s = json.load(f)
        repos.append(json.loads(s))

    return repos

""":py '913390823562317'"""
REPOS_DIR = Path("~/repos").expanduser()  # USER_EDIT

repos = find_repos(REPOS_DIR)
print(f"Found {len(repos)} repos")

""":py"""
# Note: this is much faster to run as a standalone script instead of in a notebook/bento
# as it cannot be parallelized on a notebook/bento

code_context = "full"  # USER_EDIT. Can be "line" (fast, line-only) or "full" (slow, accurate).

calls = (delayed(parse_folder)(repo, code_context=code_context, save_csv=True) for repo in repos)
run_in_parallel = Parallel(n_jobs=(1 if is_in_notebook() else -1))
_ = run_in_parallel(calls)
