from heptools.physics import Formula

class VVHH_Vertex_LO(Formula):
    _pattern = r'CV_{CV}_C2V_{C2V}_C3_{C3}'
    _diagram = ['CV', 'C2V', 'C3'], [[1, 0, 1], [2, 0, 0], [0, 1, 0]]

    _decimal_separator = '_'
    _decimal_pattern   = '{:.1f}'

    # # for datacard
    # _decimal_separator = 'p'
    # _decimal_pattern   = '{:.1g}'

class ZHH4b_LO(VVHH_Vertex_LO):
    _pattern = r'ZHHTo4B_' + VVHH_Vertex_LO._pattern

class WHH4b_LO(VVHH_Vertex_LO):
    _pattern = r'WHHTo4B_' + VVHH_Vertex_LO._pattern

class VHH4b_LO(VVHH_Vertex_LO):
    _pattern = r'VHHTo4B_' + VVHH_Vertex_LO._pattern