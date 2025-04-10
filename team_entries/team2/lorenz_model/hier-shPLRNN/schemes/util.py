from schemes.projection import projection_hierarchisation
from schemes.baseline import baseline_hierarchisation


def get_scheme_by_name(name):
    """Returns the hierarchisation scheme by name."""
    if name == 'projection':
        return projection_hierarchisation
    elif name == 'baseline':
        return baseline_hierarchisation
    else:
        raise NotImplementedError('Unknown hierarchisation scheme.')