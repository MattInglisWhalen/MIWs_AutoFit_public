# required built-ins

# required external libraries
import numpy as np
import pytest

# required internal classes
from autofit.src.data_handler import DataHandler
from autofit.src.package import pkg_path

# Testing tools

def assertRelativelyEqual(exp1, exp2):
    diff = exp1 - exp2
    av = (exp1 + exp2) / 2
    relDiff = np.abs(diff / av)
    assert relDiff < 1e-6

def assertListEqual(l1, l2):
    list1 = list(l1)
    list2 = list(l2)
    assert len(list1) == len(list2)
    for a, b in zip(list1, list2) :
        assertRelativelyEqual(a,b)


def test_init():
    with pytest.raises(TypeError) :
        default = DataHandler()

    csv = DataHandler(filepath=pkg_path()+"/data/linear_data_yerrors.csv")
    xls = DataHandler(filepath=pkg_path()+"/data/file_example_XLS_10.xls")
    xlsx = DataHandler(filepath=pkg_path()+"/data/DampedOscillations.xlsx")
    ods = DataHandler(filepath=pkg_path()+"/data/linear_noerror_multisheet.ods")
