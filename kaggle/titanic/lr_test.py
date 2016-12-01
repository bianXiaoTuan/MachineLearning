#!/bin/env python
#_*_ encoding=utf-8 _*_

import numpy as np
import pandas as pd

a = pd.DataFrame([1, 2, 3, 4, 5, 6, 7])
print a

b = a.loc([a > 4])
print b

