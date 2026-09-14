from flask import render_template, request, redirect, url_for, session, flash

def figure_generation():
    """
    Render the figure generation page.

    Returns:
        str: Rendered HTML template for the figure generation page.
    """
    return render_template('figure_generation.html')