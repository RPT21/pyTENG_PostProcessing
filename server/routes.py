from server.backend.data_loading import data_loading, browse_folder
from server.backend.cleaning_filtering_data import (
    cleaning_filtering_data,
    preview_recipe,
    save_clean_data,
)
from server.backend.cycle_peak_extraction import (
    cycle_peak_extraction,
    preview_cycles,
    save_cycle_data,
)
from server.backend.experiments_preview import (
    experiments_preview,
    open_clean_data,
    open_cycle_data,
    generate_plots,
)
from server.backend.figure_generation import figure_generation

def register_routes(app):

    @app.route('/', methods=['GET', 'POST'])
    def render_data_loading():
        return data_loading()

    @app.route('/browse_folder', methods=['GET'])
    def render_browse_folder():
        return browse_folder()

    @app.route('/experiments_preview', methods=['GET'])
    def render_experiments_preview():
        return experiments_preview()

    @app.route('/experiments_preview/clean_data/<int:experiment_id>', methods=['POST'])
    def render_open_clean_data(experiment_id):
        return open_clean_data(experiment_id)

    @app.route('/experiments_preview/cycle_data/<int:experiment_id>', methods=['POST'])
    def render_open_cycle_data(experiment_id):
        return open_cycle_data(experiment_id)

    @app.route('/experiments_preview/generate_plots', methods=['POST'])
    def render_generate_plots():
        return generate_plots()

    @app.route('/cleaning_filtering_data', methods=['GET'])
    def render_cleaning_filtering_data():
        return cleaning_filtering_data()

    @app.route('/cleaning_filtering_data/preview', methods=['POST'])
    def render_preview_recipe():
        return preview_recipe()

    @app.route('/cleaning_filtering_data/save', methods=['POST'])
    def render_save_clean_data():
        return save_clean_data()

    @app.route('/cycle_peak_extraction', methods=['GET'])
    def render_cycle_peak_extraction():
        return cycle_peak_extraction()

    @app.route('/cycle_peak_extraction/preview', methods=['POST'])
    def render_preview_cycles():
        return preview_cycles()

    @app.route('/cycle_peak_extraction/save', methods=['POST'])
    def render_save_cycle_data():
        return save_cycle_data()

    @app.route('/figure_generation', methods=['GET', 'POST'])
    def render_figure_generation():
        return figure_generation()
