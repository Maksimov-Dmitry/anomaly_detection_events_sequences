personalized anomaly detection
==============================

A short description of the project.

Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io


--------

### GitLab Flow

- If not familiar please read this [intro to GitLab Flow](https://docs.gitlab.com/ee/topics/gitlab_flow.html)
Specifically the sections ***Merge/pull requests with GitLab flow***, ***Issue tracking with GitLab flow*** and ***Linking and closing issues from merge requests***

### Coding style docs

- [PEP8](https://www.python.org/dev/peps/pep-0008/) guidelines are followed
- Documentation is provided. New functions and classes should have numpy/scipy style [docstrings](https://numpydoc.readthedocs.io/en/latest/format.html#docstring-standard)
- [PEP 484](https://www.python.org/dev/peps/pep-0484/) we use type hinting in addition to docstrings 
- TBD: use [yapf](https://github.com/google/yapf) or [Flake8](https://pypi.org/project/flake8/) for automatic style checking; yapf seems very neet and more powerful
- TBD: use [sphinx](https://www.sphinx-doc.org/en/master/)
