import pathlib

import matplotlib as mpl
import matplotlib.style

# custom plotting settings
mpl.style.use(pathlib.Path(__file__).parent.joinpath("settings.rc"))
