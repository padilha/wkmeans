# wkmeans

wkmeans is a simple implementation of the k-means clustering algorithm with weighted objects.

This package is based on the paper:  
Dhillon, I. S., Guan, Y., & Kulis, B. (2004). Kernel k-means, spectral clustering and normalized cuts.
In Proceedings of the tenth ACM SIGKDD international conference on Knowledge discovery and data mining
(pp. 551-556). ACM.

## Installation
    pip install -r requirements.txt
    python setup.py install

### Dependencies

Python 3.3+ recommended.  
Also, see requirements.txt.

## Example of use

```python
from sklearn import datasets

import numpy as np
import wkmeans

data = datasets.load_iris().data

# generating random weights in the interval [0.0, 1.0)
weights = np.random.uniform(0.0, 1.0, size=data.shape[0])

labels = wkmeans.run(data, k=3, weights=weights, max_iter=500, tol=1e-4)

print(labels)
```

## License (GPLv3)
    wkmeans: a simple implementation of the k-means clustering algorithm with weighted objects.
    Copyright (C) 2017  Victor Alexandre Padilha

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
