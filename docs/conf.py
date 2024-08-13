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
html_theme = 'pydata_sphinx_theme'

# Customize the theme
html_theme_options = {
    "github_url": "https://github.com/grecosalvatore/drift-lens",
    "use_edit_page_button": True,
    "show_prev_next": False,
    "primary_color": "#00A1D6",  # Your primary color (from your logo)
    "secondary_color": "#00405D",  # A darker or complementary color
    "header_color": "#00A1D6",  # Same as primary or a lighter shade
}

# Context for the "Edit on GitHub" button
html_context = {
    "github_user": "grecosalvatore",  # Your GitHub username
    "github_repo": "drift-lens",      # Your repository name
    "github_version": "main",         # The branch you want to link to
    "doc_path": "docs",               # The path to your documentation source files
}

# Add your custom CSS file
html_static_path = ['_static']
html_css_files = [
    'custom.css',
]
