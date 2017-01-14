from __future__ import division
import numpy as np

def decay_fn(total_iterations, output_range=[1,.1]):
    """
    Returns a function that decays its outputs with respect
    to the iterations. Useful for decreasing exploration and
    learning rates over time.
    """
    first_output, last_output = output_range
    step_update = (last_output - first_output) / total_iterations

    def decay(x):
        """ Output decays linearly with iterations. """
        return first_output + x * step_update
    return decay

