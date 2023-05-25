import re

from heptools.physics import FormulaXS


class VVHH_Vertex_LO(FormulaXS):
    diagrams = ['CV', 'C2V', 'C3'], [[1, 0, 1], [2, 0, 0], [0, 1, 0]]
    search_pattern = r'((C3_(?P<C3>\d+_\d+))|(C2V_(?P<C2V>\d+_\d+))|(CV_(?P<CV>\d+_\d+))|_)+'

    # # datacard
    # number_separator = 'p'
    # number_pattern   = '{:.1g}'

class ZHH4b_LO(VVHH_Vertex_LO):
    search_pattern = re.compile(r'ZHHTo4B_' + VVHH_Vertex_LO.search_pattern)
    format_pattern = 'ZHHTo4B_CV_{CV}_C2V_{C2V}_C3_{C3}'

class WHH4b_LO(VVHH_Vertex_LO):
    search_pattern = re.compile(r'WHHTo4B_' + VVHH_Vertex_LO.search_pattern)
    format_pattern = 'WHHTo4B_CV_{CV}_C2V_{C2V}_C3_{C3}'

class VHH4b_LO(VVHH_Vertex_LO):
    search_pattern = re.compile(r'VHHTo4B_' + VVHH_Vertex_LO.search_pattern)