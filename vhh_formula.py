import re
from functools import cached_property

from heptools.physics import FormulaXS


class ggZHH_NNLO(FormulaXS):
    @cached_property
    def diagrams(self):
        return ['CV', 'C2V', 'C3', 'CF'], [
            [0, 0, 0, 2], [1, 0, 0, 1], [0, 0, 1, 1],
            [2, 0, 0, 0], [1, 0, 1, 0], [0, 1, 0, 0]]
    @cached_property
    def search_pattern(self):
        return re.compile(r'((C3_(?P<C3>\d+_\d+))|(C2V_(?P<C2V>\d+_\d+))|(CV_(?P<CV>\d+_\d+))|(CF_(?P<CF>\d+_\d+))|_)+')

class VVHH_Vertex_LO(FormulaXS):
    @cached_property
    def diagrams(self):
        return ['CV', 'C2V', 'C3'], [[1, 0, 1], [2, 0, 0], [0, 1, 0]]
    @cached_property
    def search_pattern(self):
        return re.compile(r'((C3_(?P<C3>\d+_\d+))|(C2V_(?P<C2V>\d+_\d+))|(CV_(?P<CV>\d+_\d+))|_)+')

    # # datacard
    # number_separator = 'p'
    # number_pattern   = '{:.1g}'

class ZHH4b_LO(VVHH_Vertex_LO):
    @cached_property
    def search_pattern(self):
        return re.compile(r'ZHHTo4B_' + super().search_pattern.pattern)
    format_pattern = 'ZHHTo4B_CV_{CV}_C2V_{C2V}_C3_{C3}'

class WHH4b_LO(VVHH_Vertex_LO):
    @cached_property
    def search_pattern(self):
        return re.compile(r'WHHTo4B_' + super().search_pattern.pattern)
    format_pattern = 'WHHTo4B_CV_{CV}_C2V_{C2V}_C3_{C3}'

class VHH4b_LO(VVHH_Vertex_LO):
    @cached_property
    def search_pattern(self):
        return re.compile(r'VHHTo4B_' + super().search_pattern.pattern)