import os

import matplotlib as mpl
import matplotlib.style

# choose default backend
mpl.use("qtagg")
mpl.get_backend()

# custom plotting settings
this_dir, _ = os.path.split(__file__)
mpl.style.use(os.path.join(this_dir, "settings.rc"))
