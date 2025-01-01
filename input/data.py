# Internal Libraries
from .generator import *
from .modifier import *
from .reader import *

# Initializing data
BSk24 = get_bsk24()  
BSk24_MASS_TABLE = get_bsk24_mass_table()
BSk24_EXPERIMENTAL_MASS_TABLE = get_bsk24_experimental_mass_table()
BSk24_VARIANS = get_bsk24_varians(full_data=True)
BSk24_VARIANS_MASS_TABLE = get_bsk24_varians_mass_table(full_data=True)
