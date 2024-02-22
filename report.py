#!/usr/bin/env -S grimaldi --kernel bento_kernel_default
# FILE_UID: 1faa81f3-7708-4937-b735-86b2bdf750c4
# NOTEBOOK_NUMBER: N4981684 (656719973175728)

""":py"""
import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict
import re

""":py '1822053851595064'"""
# START HERE:
# - Look for instances of "USER_TODO" and "USER_EDIT" and follow instructions.
# - Then take a look at example queries below
# Note: This cell can take a few minutes since the csv files are big, but you only need to run it once (per notebook)

# USER_TODO: Download csv files from https://drive.google.com/drive/folders/1MYiMvFBFZwFl9CjNonoqMNP5A4qkGDkf?usp=sharing
# and put them in CSVS_DIR
I_HAVE_DOWNLOADED_THE_ICCV_AND_CVPR_CSV_FILES_ALREADY = True # USER_EDIT

if I_HAVE_DOWNLOADED_THE_ICCV_AND_CVPR_CSV_FILES_ALREADY:
    # Assumes csvs containing all calls/imports/access have been downloadead in CSVS_DIR
    CSVS_DIR = Path("~/csvs").expanduser()  # USER_EDIT

    try:
        calls = pd.read_csv(CSVS_DIR / "all_calls.csv", dtype=str)
        imports = pd.read_csv(CSVS_DIR / "all_imports.csv", dtype=str)
        access = pd.read_csv(CSVS_DIR / "all_access.csv", dtype=str)
    except FileNotFoundError as e:
        raise FileNotFoundError("You need to download the csv files from TODO and put them in CSVS_DIR") from e
else:
    # Assumes you downloaded the repos with download_repos.py and ran `parse_repos.py`.
    REPOS_DIR = Path("~/repos").expanduser()  # USER_EDIT

    calls = (pd.read_csv(path, dtype="str") for path in REPOS_DIR.glob("**/calls.csv"))
    calls = pd.concat(calls)

    imports = (pd.read_csv(path, dtype="str") for path in REPOS_DIR.glob("**/imports.csv"))
    imports = pd.concat(imports)

    access = (pd.read_csv(path, dtype="str") for path in REPOS_DIR.glob("**/access.csv"))
    access = pd.concat(access)

# Sometime the code is Na which makes the rest of the script fail... figure out why - probs file encoding
calls["code"] = calls["code"].apply(str)
imports["code"] = imports["code"].apply(str)
access["code"] = access["code"].apply(str)

""":py"""
def get_calls(api_name):
    return calls[calls["api_name"] == api_name]


def get_imports(api_name, include_subs=False):
    if include_subs:
        # Will catch all of `torchvision.transforms` *and* the submodules like `torchvision.transforms.functional`
        # Note the call to drop_duplicate to remove entries like
        # from torchvision.transforms import Compose, Resize, ToTensor
        # which would be present 3 times.
        return imports[imports["api_name"].str.startswith(api_name)].drop_duplicates(subset=["file_path", "code"])
    else:
        return imports[imports["api_name"] == api_name]


def get_access(api_name):
    return access[access["api_name"] == api_name]


def pprint(df, verbose=True, out=None):
    """Pretty-print a pandas dataframe, optionally outputs to `out` file."""
    if out:
        out_file = Path(out).expanduser()
        f = open(out_file, "w+")

    repos, counts = np.unique(df["url"], return_counts=True)
    repo_to_count = {repo: count for (repo, count) in zip(repos, counts)}

    def _print(s=""):
        print(s)
        if out:
            f.write(s + "\n")

    _print(f"Found {len(df)} times in {repos.size} repos.")

    if not verbose:
        for repo, count in repo_to_count.items():
            _print(f"{repo} - {count}")
        return

    prev_repo = None
    permalinks = []
    for _, row in df[["url", "permalink", "code"]].iterrows():
        repo = row["url"]
        if repo != prev_repo:
            for permalink in permalinks:
                _print(permalink)

            _print()
            _print("=" * len(repo))
            _print(row["url"])
            _print("=" * len(repo))
            _print()
            _print(f"Found {repo_to_count[repo]} times")

            prev_repo = repo
            permalinks = []
        _print(str(row["code"]))
        permalinks.append(row["permalink"])
    for permalink in permalinks:
        _print(permalink)
    if out:
        f.close()

""":py '958803902268974'"""
# Q: Are most researchers CPU-poor?
#    What are the most common DataLoader `num_workers` values?
# A: It's overwhelmingly <= 4 (a bit of 8 too).
# Note: most of them aren't hard-coded and cannot be inferred statically.

dl_calls = calls[calls["api_name"] == "torch.utils.data.DataLoader"]
codes = dl_calls[dl_calls["code"].str.contains("num_workers=")]["code"]
counts = defaultdict(int)
for code in codes:
    for line in code.split("\n"):
        if (match := re.match(".*num_workers=([^,]*),", line)):
            try:
                num_workers = int(match.group(1))
            except ValueError:
                num_workers = -1000  # dynamically set e.g. "args.num_workers"
            counts[num_workers] += 1

counts = {("dynamic" if k == -1000 else k): v for (k, v) in reversed(sorted(counts.items(), key=lambda x:x[1]))}
print("num_workers      counts")
for (k, v) in counts.items(): 
    print(f"{k:>11}        {v:>4}")

""":py '1098236214662999'"""
pprint(dl_calls, out="~/reports/torch.utils.data.DataLoader__calls")

""":py '355747120773009'"""
# Q: What are the most popular video decoders, and who's using them? How?

api_names = ["cv2.VideoCapture", "torchvision.io.read_video", "torchvision.io.VideoReader", "decord.VideoReader", "av.open", "ffmpegcv.VideoCapture", "ffmpeg.input"]
dfs = [get_calls(api_name) for api_name in api_names]
for df, api_name in reversed(sorted(zip(dfs, api_names), key=lambda x: len(x[0]))):
    print(f"{api_name:<27} {len(df['url'].unique())} repos")

""":py '3634515930140424'"""
for df, api_name in zip(dfs, api_names):
    pprint(df, out=f"~/reports/{api_name}__calls")

""":py '2370923193298330'"""
# Q: I know this operator is slow and widely used. I need to optimize it but it has a lot of backends.
#    Which one of its configuration should I prioritize?
resize_calls = calls[calls["api_name"].isin(("torchvision.transforms.Resize", "torchvision.transforms.functional.resize"))]

bicubic_calls = resize_calls[resize_calls["code"].str.contains("cubic", case=False)]
bilinear_calls = resize_calls[
    (resize_calls["code"].str.contains("linear", case=False))
    # if interpolation= isn't passed, the default mode is bilinear
    | ~(resize_calls["code"].str.contains("interpolation", case=False))
]
nearest_calls = resize_calls[resize_calls["code"].str.contains("nearest", case=False)]
print("mode       calls")
for df, mode in zip((bilinear_calls, bicubic_calls, nearest_calls), ("bilinear", "bicubic", "nearest")):
    print(f"{mode:<10} {len(df)}")

""":py '2092613244415589'"""
# Q: Between Iterable and MapStyle datasets, which is the most widely used?
#    And where?
iterable_access = get_access("torch.utils.data.IterableDataset")
custom_iterable_dataset = iterable_access[iterable_access["code"].str.startswith("class")]

mapstyle_access = get_access("torch.utils.data.Dataset")
custom_mapstyle_dataset = mapstyle_access[mapstyle_access["code"].str.startswith("class")]
for df, name in zip((custom_iterable_dataset, custom_mapstyle_dataset), ("Iterable", "MapStyle")):
    print(f"{name}: {len(df)} custom definitions in {df['url'].unique().size} repos")

""":py '921030326004880'"""
pprint(custom_iterable_dataset, out="~/reports/custom_iterable_datasets")

""":py '1387059812179600'"""
pprint(custom_mapstyle_dataset, out="~/reports/custom_mapstyle_datasets")

""":py '1564504817669536'"""
# Q: I want to know how users are combining datasets. Maybe I can look at those repos that define a lot of custom ones?

TOP = 20
repos, counts = np.unique(custom_mapstyle_dataset["url"], return_counts=True)
for (repo, count), _ in zip(reversed(sorted(zip(repos, counts), key=lambda r_c: r_c[1])), range(TOP)):
    print(f"{repo:<70} {count}")

""":py '812813894193744'"""
# Q: I'm "competing" against other libraries in the OSS space. What are their most popular APIs?

libs = ["kornia", "albumentations", "cv2"]
dfs = [calls[calls["api_name"].str.startswith(lib)] for lib in libs]

TOP = 10
for df, lib in zip(dfs, libs):
    api_names, counts = np.unique(df["api_name"], return_counts=True)
    for (api_name, count), _ in zip(reversed(sorted(zip(api_names, counts), key=lambda a_c: a_c[1])), range(TOP)):
        print(f"{api_name:<60} {count}")
    print()

""":py '339004585143301'"""
for df, lib in zip(dfs, libs):
    pprint(df, out=f"~/reports/{lib}__calls")

""":py '417500534176722'"""
# Q: The `pretrained` parameter of torchvision model builders has been deprecated for a while.
#    But I know it's still widely used. What are these repos, so I can send them an automated TorchFix PR?
torchvision_models_calls = calls[calls["api_name"].str.startswith("torchvision.models")]
torchvision_models_calls = torchvision_models_calls[torchvision_models_calls["code"].str.contains("pretrained=")]
TOP = 20
repos, counts = np.unique(torchvision_models_calls["url"], return_counts=True)
for (repo, count), _ in zip(reversed(sorted(zip(repos, counts), key=lambda r_c: r_c[1])), range(TOP)):
    print(f"{repo:<70} {count}")

""":py '224962784028056'"""
# Q: The API of `make_grid` and `save_image` in torchvision is clunky and as a maintainer I'd like to deprecate them.
#    How disruptive would that be?
# A: Too disruptive :'(
pprint(calls[calls["api_name"].isin(("torchvision.utils.save_image", "torchvision.utils.make_grid"))], out="~/reports/make_grid__save_image__calls")

""":py"""


""":py"""


""":py '307995155601820'"""


""":py '1171898860642538'"""

