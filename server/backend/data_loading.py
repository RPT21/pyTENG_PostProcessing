from flask import render_template, request, redirect, url_for, session, flash

def data_loading():
    """
    Render the data loading page.

    Returns:
        str: Rendered HTML template for the data loading page.
    """
    return render_template('data_loading.html')