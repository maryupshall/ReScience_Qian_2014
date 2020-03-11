"""Handle units and define aliases for common current and conductance units."""
from pint import UnitRegistry

ureg = UnitRegistry()

mS_PER_CM2 = ureg.millimho / ureg.centimeter ** 2
uA_PER_CM2 = ureg.microamp / ureg.centimeter ** 2


def strip_dimension(p):
    """Remove the pint dimension while ensuring proper scale.

    :param p: Parameter to strip
    :return: Striped parameter
    """
    try:
        if p.check(ureg.V):
            return (p.to(ureg.mV)).magnitude
        elif p.check(uA_PER_CM2):
            return (p.to(uA_PER_CM2)).magnitude
        elif p.check(mS_PER_CM2):
            return (p.to(mS_PER_CM2)).magnitude
        elif p.check(ureg.s):
            return (p.to(ureg.ms)).magnitude
        else:
            raise ValueError("Unknown Unit Type")
    except AttributeError:  # non united p
        return p
