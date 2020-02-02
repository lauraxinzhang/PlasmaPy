#!/usr/bin/env python3.6
# -*- coding: utf-8 -*-
#
# PlasmaPy documentation build configuration file, created by
# sphinx-quickstart on Wed May 31 18:16:46 2017.
#
# This file is execfile()d with the current directory set to its
# containing dir.
#
# Note that not all possible configuration values are present in this
# autogenerated file.
#
# All configuration values have a default; values that are commented out
# serve to show the default.

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#

import os
import sys

sys.path.insert(0, os.path.abspath('..'))

try:
    import pyvista
    import numpy as np
    # Manage errors
    pyvista.set_error_output_file('errors.txt')
    # Ensure that offscreen rendering is used for docs generation
    pyvista.OFF_SCREEN = True # Not necessary - simply an insurance policy
    # Preferred plotting style for documentation
    pyvista.set_plot_theme('document')
    pyvista.rcParams['window_size'] = np.array([1024, 768]) * 2
    # Save figures in specified directory
    pyvista.FIGURE_PATH = os.path.join(os.path.abspath('./images/'), 'auto-generated/')
    if not os.path.exists(pyvista.FIGURE_PATH):
        os.makedirs(pyvista.FIGURE_PATH)
except ImportError:
    pass

# -- General configuration ------------------------------------------------

# If your documentation needs a minimal Sphinx version, state it here.
#
# needs_sphinx = '1.0'

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = ['sphinx.ext.autodoc', 'sphinx.ext.mathjax',
              'sphinx.ext.napoleon', 'sphinx.ext.intersphinx',
              'sphinx_automodapi.automodapi',
              'sphinx_automodapi.smart_resolver',
              'sphinx_gallery.gen_gallery',
              'sphinx.ext.graphviz']


intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'numpy': ('https://docs.scipy.org/doc/numpy', None),
    'scipy': ('https://docs.scipy.org/doc/scipy/reference/', None),
    'pandas': ('https://pandas.pydata.org/pandas-docs/stable/', None),
    'astropy': ('http://docs.astropy.org/en/stable/', None),
    'numba': ('https://numba.pydata.org/numba-doc/dev/', None)}
# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# The suffix(es) of source filenames.
# You can specify multiple suffix as a list of string:
#
# source_suffix = ['.rst', '.md']
source_suffix = '.rst'

# The master toctree document.
master_doc = 'index'

# General information about the project.
project = 'PlasmaPy'
copyright = '2015-2019, PlasmaPy Community'
author = 'PlasmaPy Community'

# The version info for the project you're documenting, acts as replacement for
# |version| and |release|, also used in various other places throughout the
# built documents.
#
# The short X.Y version.
version = '0.2'
# The full version, including alpha/beta/rc tags.
release = '0.2.0'

# The language for content autogenerated by Sphinx. Refer to documentation
# for a list of supported languages.
#
# This is also used if you do content translation via gettext catalogs.
# Usually you set "language" from the command line for these cases.
language = None

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This patterns also effect to html_static_path and html_extra_path
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = 'sphinx'

# If true, `todo` and `todoList` produce output, else they produce nothing.
todo_include_todos = False

default_role = 'obj'

# -- Options for HTML output ----------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
# html_theme = 'alabaster'
# html_theme = 'traditional'
# html_theme = 'agogo'
html_theme = "sphinx_rtd_theme"

# Theme options are theme-specific and customize the look and feel of a theme
# further.  For a list of options available for each theme, see the
# documentation.
#
# html_theme_options = {}

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
# html_static_path = ['_static']


# -- Options for HTMLHelp output ------------------------------------------

# Output file base name for HTML help builder.
htmlhelp_basename = 'PlasmaPydoc'


# -- Options for LaTeX output ---------------------------------------------

latex_elements = {
    # The paper size ('letterpaper' or 'a4paper').
    #
    # 'papersize': 'letterpaper',

    # The font size ('10pt', '11pt' or '12pt').
    #
    # 'pointsize': '10pt',

    # Additional stuff for the LaTeX preamble.
    #
    # 'preamble': '',

    # Latex figure (float) alignment
    #
    # 'figure_align': 'htbp',
}

# Grouping the document tree into LaTeX files. List of tuples
# (source start file, target name, title,
#  author, documentclass [howto, manual, or own class]).
latex_documents = [
    (master_doc, 'PlasmaPy.tex', 'PlasmaPy Documentation',
     'PlasmaPy Community', 'manual'),
]


# -- Options for manual page output ---------------------------------------

# One entry per manual page. List of tuples
# (source start file, name, description, authors, manual section).
man_pages = [
    (master_doc, 'plasmapy', 'PlasmaPy Documentation',
     [author], 1)
]


# -- Options for Texinfo output -------------------------------------------

# Grouping the document tree into Texinfo files. List of tuples
# (source start file, target name, title, author,
#  dir menu entry, description, category)
texinfo_documents = [
    (master_doc, 'PlasmaPy', 'PlasmaPy Documentation',
     author, 'PlasmaPy', 'Python package for plasma physics',
     'Miscellaneous'),
]

html_favicon = "./_static/icon.ico"

# -- Options for Sphinx Gallery -----------------

# Patch sphinx_gallery.binder.gen_binder_rst so as to point to .py file in repository
# Original code as per sphinx_gallery, under the BSD 3-Clause license

import sphinx_gallery.binder
def patched_gen_binder_rst(fpath, binder_conf, gallery_conf):
    """Generate the RST + link for the Binder badge.
    Parameters
    ----------
    fpath: str
        The path to the `.py` file for which a Binder badge will be generated.
    binder_conf: dict or None
        If a dictionary it must have the following keys:
        'binderhub_url'
            The URL of the BinderHub instance that's running a Binder service.
        'org'
            The GitHub organization to which the documentation will be pushed.
        'repo'
            The GitHub repository to which the documentation will be pushed.
        'branch'
            The Git branch on which the documentation exists (e.g., gh-pages).
        'dependencies'
            A list of paths to dependency files that match the Binderspec.
    Returns
    -------
    rst : str
        The reStructuredText for the Binder badge that links to this file.
    """
    binder_conf = sphinx_gallery.binder.check_binder_conf(binder_conf)
    binder_url = sphinx_gallery.binder.gen_binder_url(fpath, binder_conf, gallery_conf)
    binder_url = binder_url.replace(gallery_conf['gallery_dirs'] + os.path.sep, "").replace("ipynb", "py")

    rst = (
        "\n"
        "  .. container:: binder-badge\n\n"
        "    .. image:: https://mybinder.org/badge_logo.svg\n"
        "      :target: {}\n"
        "      :width: 150 px\n").format(binder_url)
    return rst
sphinx_gallery.binder.gen_binder_rst = patched_gen_binder_rst

sphinx_gallery_conf = {
    # path to your examples scripts
    'examples_dirs': '../plasmapy/examples',
    # path where to save gallery generated examples
    'backreferences_dir': 'gen_modules/backreferences',
    'gallery_dirs': 'auto_examples',
    'binder': {
        'org': 'PlasmaPy',
        'repo': 'PlasmaPy',
        'branch': 'master',
        'binderhub_url': 'https://mybinder.org',
        'dependencies': ['../binder/requirements.txt'],
        'notebooks_dir': 'plasmapy/examples',
    },
    'image_scrapers': ('pyvista',),
}
