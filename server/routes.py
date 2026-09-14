from server.backend.data_loading import data_loading, browse_folder
from server.backend.cleaning_filtering_data import cleaning_filtering_data
from server.backend.cycle_peak_extraction import cycle_peak_extraction
from server.backend.experiments_preview import experiments_preview
from server.backend.figure_generation import figure_generation

def register_routes(app):

    @app.route('/', methods=['GET', 'POST'])
    def render_data_loading():
        return data_loading()

    @app.route('/browse_folder', methods=['GET'])
    def render_browse_folder():
        return browse_folder()

    @app.route('/experiments_preview', methods=['GET', 'POST'])
    def render_experiments_preview():
        return experiments_preview()

    @app.route('/cleaning_filtering_data', methods=['GET', 'POST'])
    def render_cleaning_filtering_data():
        return cleaning_filtering_data()

    @app.route('/cycle_peak_extraction', methods=['GET', 'POST'])
    def render_cycle_peak_extraction():
        return cycle_peak_extraction()

    @app.route('/figure_generation', methods=['GET', 'POST'])
    def render_figure_generation():
        return figure_generation()
