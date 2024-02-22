Your favourite conference is out, you have links to all the GitHub repos for
that conf and you want to get a sense of what researchers have been using? This
collection of scripts will help you figure out *where* and *how* Python APIs are
used.

IN: A bunch of GitHub URLs, e.g. all the URLS from ICCV/CVPR.
OUT: The ability to find all calls, imports or access of any API e.g. all calls
     to `torch.compile` or `some_random_lib.CoolClass` with:
     - the code snippet in which the call/import/access happened.
     - the GitHub permalink so you can browse more on GitHub and gain context.
     - Answers to random questions like:
         - what are the most common values for DataLoader's `num_workers`?
         - what is the most popular video decoder?
         - who's using iterable datasets? who's using mapstyle datasets?
         - what are the most popular APIs from other libraries competing with mine?
         - who's still using this depreacted parameter?
         - when people use torchvision' Resize() or resize(), do they use bilinear or bicubic mode?

See Examples section below.

How it works
------------

(Roughly)

- 1. `download_repos.py`: Clone GitHub repositories in batches.
- 2. `parse_repos.py`: Parse Python files in the repos and record all
     calls/imports/attribute accesses of any api. Results are available as pandas
     DataFrames and saved in csv files.
- 3. `report.py`: Query the saved csv files and get reports for the APIs
     you're interested in.  Open this as a bento notebook.

These files are meant to be copy/pasted and modified on your own devvm or laptop.
This isn't meant to be a buck project (although it could).

Pre-requisite / dependencies
----------------------------

TL;DR: 32GB of RAM and joblib, pandas, numpy.

On a devvm with 32 cores, this scales reasonably well for the ~2,500 repos of
ICCV/CVPR.  It should take <20 minutes to download all repos, run the analysis
and load the resulting csvs to start querying. Each api report should then just
take a few seconds. The aggregated resuling csv files/dataframes are about ~20GB
in total.

`download_repos.py` and `parse_repos.py` can be executed as bento notebooks for
exploratory work, but running them as standalone i.e. `python
download_repos.py` will be a lot faster as it will leverage multiprocessing.

To do that on a devvm you'll need a conda/virtualenv env with `joblib`, `pandas` and
`numpy` installed:

```bash
$ https_proxy=http://fwdproxy:8080 http_proxy=http://fwdproxy:8080 pip install pandas numpy joblib
```

Where to start
--------------

See 1. 2. 3. from above. The global vars you might want to change are noted as
`USER_EDIT`. For now it's just:
```bash
grep -nr USER_EDIT --exclude README.md
download_repos.py:50:REPOS_DIR = Path("~/repos").expanduser()  # USER_EDIT
download_repos.py:53:URLS_FILE = "~/dev/repo_analysis/{conf}_urls"  # USER_EDIT
parse_repos.py:298:REPOS_DIR = Path("~/repos").expanduser()  # USER_EDIT
parse_repos.py:307:code_context = "line"  # USER_EDIT. Can be "line" (fast, line-only) or "full" (slow, accurate).
report.py:10:REPOS_DIR = Path("~/repos").expanduser()  # USER_EDIT

```

Where to find GitHub URLs for a conference
------------------------------------------

I got the CVPR and ICCV URLs by scraping
https://github.com/DmitryRyumin/CVPR-2023-Papers and
https://github.com/DmitryRyumin/ICCV-2023-Papers/.

The results are in P1184259737 and P1184259833 respectively.
For other conferences... IDK, but someone else probably did it already. Maybe PapersWithCode?

References
----------

Some key parts of these scripts are adapted from:

- https://github.com/fmassa/python-lib-stats
- https://fburl.com/code/ybwdm7ty

Report/Queries Examples
-----------------------

Once you have downloaded and ran the repos analysis (steps 1 and 2), you can
play with the resulting pandas DataFrames, or just call `report()` from
`report.py`.

You can also run more funky stuff like:

```py
resize_calls = calls[calls["api_name"].isin(("torchvision.transforms.Resize", "torchvision.transforms.functional.resize"))]

bicubic_calls = resize_calls[resize_calls["code"].str.contains("cubic", case=False)]
bilinear_calls = resize_calls[
    (resize_calls["code"].str.contains("linear", case=False))
    # if interpolation= isn't passed, the default mode is bilinear
    | ~(resize_calls["code"].str.contains("interpolation="))
]
nearest_calls = resize_calls[resize_calls["code"].str.contains("nearest", case=False)]
len(bilinear_calls), len(bicubic_calls), len(nearest_calls)
# (3981, 553, 209)
```
