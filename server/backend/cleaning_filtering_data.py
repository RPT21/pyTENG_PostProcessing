from flask import render_template, request, redirect, url_for, session, flash

def cleaning_filtering_data():
    """
    Render the cleaning and filtering data page.

    Returns:
        str: Rendered HTML template for the cleaning and filtering data page.
    """
    return render_template('cleaning_filtering_data.html')