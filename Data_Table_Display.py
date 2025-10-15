# @tidy {"drop": true}
# We do this to avoid "extension already loaded" warnings below.
%unload_ext google.colab.data_table

from google.colab import data_table
data_table.enable_dataframe_formatter()

from google.colab import data_table
data_table.disable_dataframe_formatter()

#exploring data tables
from google.colab import data_table
import vega_datasets

data_table.enable_dataframe_formatter()
vega_datasets.data.airports()

from google.colab import data_table

data_table.disable_dataframe_formatter()
vega_datasets.data.airports()

from google.colab import data_table

data_table.DataTable(
    vega_datasets.data.airports(), include_index=False, num_rows_per_page=10
)
