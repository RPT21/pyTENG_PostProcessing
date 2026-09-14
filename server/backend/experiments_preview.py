from flask import render_template, request, redirect, url_for, session, flash

def experiments_preview():
    """
    Render the experiments preview page.

    Returns:
        str: Rendered HTML template for the experiments preview page.
    """
    return render_template('experiments_preview.html')