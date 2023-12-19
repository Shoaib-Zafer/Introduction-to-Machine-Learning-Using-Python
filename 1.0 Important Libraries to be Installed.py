# Below Libraries need to be installed for the code to function properly
# !pip install works in Jupyter Lab for other IDEs use the command pip install in command prompt  

!pip install numpy
from gettext import install
import numpy as np
print("NumPy version: {}".format(np.__version__))

!pip install pandas
import pandas as pd
print("pandas version: {}".format(pd.__version__))

!pip install scipy
import scipy as sp
print("SciPy version: {}".format(sp.__version__))

!pip install matplotlib
import matplotlib
print("matplotlib version: {}".format(matplotlib.__version__))


!pip install sklearn
import sklearn
print("scikit-learn version: {}".format(sklearn.__version__))
