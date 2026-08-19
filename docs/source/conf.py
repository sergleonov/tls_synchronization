# Configuration file for the Sphinx documentation builder.
# Referenced https://github.com/readthedocs-examples

import os
import sys
from datetime import datetime
 
# -- Path setup --------------------------------------------------------------
# The package is normally imported from the installed environment. Adding the
# src directory makes autodoc work for local builds even without installation.
sys.path.insert(0, os.path.abspath("../../src"))
 
# -- Project information ------------------------------------------------------
project = "TLS Synchronization"
author = "Sergei Leonov"
copyright = f"{datetime.now():%Y}, {author}"
release = "0.1.0"
version = "0.1.0"
 
# -- General configuration ----------------------------------------------------
extensions = [
    "sphinx.ext.autodoc",       # pull documentation from docstrings
    "sphinx.ext.autosummary",   # generate summary tables
    "sphinx.ext.napoleon",      # NumPy- and Google-style docstrings
    "sphinx.ext.viewcode",      # add links to highlighted source
    "sphinx.ext.intersphinx",   # cross-link to other projects' docs
    "sphinx.ext.mathjax",       # render math
    "sphinx.ext.todo",          # todo directives
]
 
templates_path = ["_templates"]
exclude_patterns = []
 
# -- autodoc / autosummary ----------------------------------------------------
autosummary_generate = True
autodoc_typehints = "description"
autodoc_member_order = "bysource"
autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "show-inheritance": True,
}
 
# Heavy / compiled dependencies can be mocked so the docs build even where they
# are not importable (e.g. a minimal CI image). Uncomment as needed.
# autodoc_mock_imports = ["qutip", "oqupy", "PyQt5", "mpmath"]
 
# -- napoleon -----------------------------------------------------------------
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True
napoleon_use_rtype = False
 
# -- intersphinx --------------------------------------------------------------
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
    "matplotlib": ("https://matplotlib.org/stable/", None),
    "qutip": ("https://qutip.readthedocs.io/en/stable/", None),
    "oqupy": ("https://oqupy.readthedocs.io/en/latest/", None),
}
 
# -- todo ---------------------------------------------------------------------
todo_include_todos = True
 
# -- Options for EPUB output
epub_show_urls = "footnote"

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "sphinx_rtd_theme"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]
html_theme_options = {
    "navigation_depth": 3,
    "collapse_navigation": False,
    "sticky_navigation": True,
}