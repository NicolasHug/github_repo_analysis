#!/usr/bin/env -S grimaldi --kernel bento_kernel_default
# FILE_UID: 0aae3a55-7785-4c41-a5f7-4fa96faa9dee

""":py '737838371396778'"""
from pathlib import Path
import json
import subprocess
from typing import Optional
from joblib import Parallel, delayed # pip install joblib

""":py '740614251468780'"""
def download_repo(url : str, dest : str, extra_meta : Optional[dict[str, str]] = None): 
    """Download a github repo from its `url`, saves it into `dest`.

    Also save a small analysis_meta.json file in `dest`, with some basic metadata like github URL and current commit hash.
    If passed, `extra_meta` is saved in that .json file as well.
    """
    url = url.split("/")
    repo_name = url[-1]
    repo_user = url[-2]

    github_url = f"git@github.com:{repo_user}/{repo_name}"

    META_FILE_NAME =  "analysis_meta.json"
    dest = Path(dest) / repo_name
    if (dest / META_FILE_NAME).exists():  # Already cloned
        return

    out = (
        subprocess.Popen(["git", "clone", github_url, dest], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        .communicate()[1]
        .decode("utf-8")
    )
    if "Could not read from remote repository" in out:
        return
    print(out)

    hash = (
        subprocess.Popen(["git", "rev-parse", "HEAD"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd=str(dest))
        .communicate()[0]
        .decode("utf-8")
    ).strip("\n")
    
    extra_meta = extra_meta or {}
    with open(dest / META_FILE_NAME, "w") as meta_file:
        meta = json.dumps({"name":str(repo_name), "url":"/".join(url), "folder":str(dest), "hash": hash, **extra_meta})
        json.dump(meta, meta_file)

""":py"""
REPOS_DIR = Path("~/repos").expanduser()  # USER_EDIT

# cvpr and iccv URLs can be found in P1184259737 and P1184259833 respectively
URLS_FILE = "~/dev/repo_analysis/{conf}_urls"  # USER_EDIT


def is_in_notebook():
    import __main__ as main
    return not hasattr(main, '__file__')

# Note: this is much faster to run as a standalone script instead of in a notebook/bento
# as it cannot be parallelized on a notebook/bento

for conf in ("cvpr", "iccv"):
    with open(Path(URLS_FILE.format(conf=conf)).expanduser(), "r") as f:
        urls = [line.strip("\n") for line in f]

    dest = (REPOS_DIR / conf)
    extra_meta = {"conf": conf}

    calls = (delayed(download_repo)(url, dest=dest, extra_meta=extra_meta) for url in urls)
    run_in_parallel = Parallel(n_jobs=(1 if is_in_notebook() else -1))
    run_in_parallel(calls)

""":py"""

