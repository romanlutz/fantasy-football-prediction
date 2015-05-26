# Copyright (c) Roman Lutz. All rights reserved.
# The use and distribution terms for this software are covered by the
# Eclipse Public License 1.0 (http://opensource.org/licenses/eclipse-1.0.php)
# which can be found in the file LICENSE.md at the root of this distribution.
# By using this software in any fashion, you are agreeing to be bound by
# the terms of this license.
# You must not remove this notice, or any other, from this software.

import matplotlib.pyplot as plt
import pylab as P
import numpy as np

def histogram(y_vals, pred):
    P.hist(map(lambda (y, p): np.fabs(y-p), zip(y_vals,pred)), bins=range(35), rwidth=1.0, histtype='bar')
    P.xlabel('Absolute Error')
    P.ylabel('number of data cases')
    #P.xlim([0, 1])
    #P.xticks([0.25, 0.75], ['female', 'male'])
    P.title('Absolute Error Distribution')
    P.savefig('absolute_error_distribution.pdf')
    # P.show()
    P.close('all')
