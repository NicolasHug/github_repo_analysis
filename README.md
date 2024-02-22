Your favourite conference is out, you have links to all the GitHub repos for
that conf and you want to get a sense of what researchers have been using? This
collection of scripts will help you figure out *where* and *how* Python APIs are
used.

**IN**: A bunch of GitHub URLs, e.g. all the URLS from ICCV/CVPR.
**OUT**: The ability to find all calls, imports or access of any API e.g. all calls
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

See in `report.py` or in N4981684.

Getting Started
---------------

(Roughly)

1. `download_repos.py`: Clone specified GitHub repositories in batches.
2. `parse_repos.py`: Parse Python files in the repos and record all
   calls/imports/attribute accesses of any api. Results are available as pandas
   DataFrames and saved in csv files.
3. `report.py`: Query the saved csv files and get reports for the APIs
   you're interested in. Open this as a bento notebook.

These files are meant to be copy/pasted and modified on your own devvm or laptop.
This isn't meant to be a buck project (although it could).


Steps 1 and 2 have been done for all 2.5k repos from ICCV and CVPR. You can
download the resulting csv files from
https://drive.google.com/drive/folders/1MYiMvFBFZwFl9CjNonoqMNP5A4qkGDkf?usp=sharing
and go straight to step 3.

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
grep -nr -e USER_EDIT -e USER_TODO --exclude README.md

download_repos.py:50:REPOS_DIR = Path("~/repos").expanduser()  # USER_EDIT
download_repos.py:53:URLS_FILE = "~/dev/repo_analysis/{conf}_urls"  # USER_EDIT
parse_repos.py:295:REPOS_DIR = Path("~/repos").expanduser()  # USER_EDIT
parse_repos.py:304:code_context = "full"  # USER_EDIT. Can be "line" (fast, line-only) or "full" (slow, accurate).
report.py:14:# - Look for instances of "USER_TODO" and "USER_EDIT" and follow instructions.
report.py:18:# USER_TODO: Download csv files from https://drive.google.com/drive/folders/1MYiMvFBFZwFl9CjNonoqMNP5A4qkGDkf?usp=sharing
report.py:20:I_HAVE_DOWNLOADED_THE_ICCV_AND_CVPR_CSV_FILES_ALREADY = True # USER_EDIT
report.py:24:    CSVS_DIR = Path("~/csvs").expanduser()  # USER_EDIT
report.py:34:    REPOS_DIR = Path("~/repos").expanduser()  # USER_EDIT

```

Where to find GitHub URLs for a conference
------------------------------------------

I got the CVPR and ICCV URLs by scraping
https://github.com/DmitryRyumin/CVPR-2023-Papers and
https://github.com/DmitryRyumin/ICCV-2023-Papers/.

The results are in `cvpr_urls` and `iccv_urls`.
For other conferences... IDK, but someone else probably did it already. Maybe PapersWithCode?

References
----------

Some key parts of these scripts are adapted from:

- https://github.com/fmassa/python-lib-stats
- https://fburl.com/code/ybwdm7ty
