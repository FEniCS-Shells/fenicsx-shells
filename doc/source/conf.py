# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import os
import sys

sys.path.insert(0, os.path.abspath("."))
import jupytext_process

jupytext_process.process()

project = "FEniCSx-Shells"
copyright = "2022-2024, FEniCSx-Shells Authors"
author = "FEniCSx-Shells Authors"
release = "0.10.0.dev0"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.mathjax",
    "sphinx.ext.napoleon",
    "sphinx.ext.todo",
    "sphinx.ext.viewcode",
    "myst_parser",
]

templates_path = ["_templates"]
exclude_patterns = []

source_suffix = [".rst", ".md"]

html_theme = "sphinx_rtd_theme"

myst_enable_extensions = [
    "dollarmath",
]

autodoc_default_options = {
    "members": True,
    "show-inheritance": True,
    "imported-members": True,
    "undoc-members": True,
}
autosummary_generate = True
autoclass_content = "both"

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_static_path = ["_static"]
