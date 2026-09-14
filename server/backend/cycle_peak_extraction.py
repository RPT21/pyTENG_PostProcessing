from flask import render_template, request, redirect, url_for, session, flash

def cycle_peak_extraction():
    """
    Render the cycle peak extraction page.

    Returns:
        str: Rendered HTML template for the cycle peak extraction page.
    """
    return render_template('cycle_peak_extraction.html')