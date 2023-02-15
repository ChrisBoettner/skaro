import os

import matplotlib.pyplot as plt
import matplotlib.style
import matplotlib as mpl

# choose default backend
plt.switch_backend("QtAgg")

# custom plotting settings
this_dir, _ = os.path.split(__file__)
mpl.style.use(os.path.join(this_dir, "settings.rc"))
