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


# Context for the "Edit on GitHub" button
html_context = {
    "github_user": "grecosalvatore",  # Your GitHub username
    "github_repo": "drift-lens",      # Your repository name
    "github_version": "main",         # The branch you want to link to
    "doc_path": "docs",               # The path to your documentation source files
}

# Update HTML context for the logo
html_theme_options = {
    'logo_only': True,  # Show only the logo without the project name
}

# Add your custom CSS file
html_static_path = ['_static']
html_css_files = [
    'custom.css',
]
