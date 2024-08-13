# -- Project information -----------------------------------------------------
project = 'DriftLens'
copyright = '2024, Your Name'
author = 'Your Name'

# -- General configuration ---------------------------------------------------
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
]

templates_path = ['_templates']
exclude_patterns = []

# -- Options for HTML output -------------------------------------------------
html_theme = 'sphinx_rtd_theme'

# Add the path to the static files
html_static_path = ['_static']

html_logo = "_static/images/Drift_Lens_Logo.png"
html_theme_options = {
    'logo_only': True,
    'display_version': False,
}

# Context for the "Edit on GitHub" button
html_context = {
    "github_user": "grecosalvatore",  # Your GitHub username
    "github_repo": "drift-lens",      # Your repository name
    "github_version": "main",         # The branch you want to link to
    "doc_path": "docs",               # The path to your documentation source files
}


