from typing import Iterable

from xs import ZHH4b_samples, WHH4b_samples

class era:
    _eras = {
        '2016': ['2016preVFP', '2016postVFP'],
        '2017': ['2017'],
        '2018': ['2018'],
    }
    @classmethod
    def _list(cls, _iter: Iterable):
        return sorted(list(set(_iter)))
    def __init__(self, years: Iterable) -> None:
        self.years = self._list(str(year) for year in years)
    @property
    def data(self):
        return self.years
    @property
    def mc(self):
        return self._list((sum((self._eras[year] for year in self.years), [])))
    def __iter__(self):
        for _era in self._list(self.mc + self.data):
            yield _era

class VHH_2j4b:
    # file
    eos = False
    user = 'chuyuanl'
    input = f'root://cmseos.fnal.gov//store/user/{user}/condor/VHH/' if eos else f'/VHH/input/'
    output = f'root://cmseos.fnal.gov//store/user/{user}/condor/VHH/' if eos else f'/VHH/output/'
    SvB  = 'SvB_MA_VHH_8nc.root'
    # dataset
    eras = era({2016, 2017, 2018})
    MC_signal = ZHH4b_samples.basis_process + WHH4b_samples.basis_process
    HLT = {
        '2016': [
            'QuadJet45_TripleBTagCSV_p087',
            'DoubleJet90_Double30_TripleBTagCSV_p087',
            'DoubleJetsC100_DoubleBTagCSV_p014_DoublePFJetsC100MaxDeta1p6'
        ],
        '2017': [
            'PFHT300PT30_QuadPFJet_75_60_45_40_TriplePFBTagCSV_3p0',
            'DoublePFJets100MaxDeta1p6_DoubleCaloBTagCSV_p33'
        ],
        '2018': [
            'DoublePFJets116MaxDeta1p6_DoubleCaloBTagDeepCSV_p71',
            'PFHT330PT30_QuadPFJet_75_60_45_40_TriplePFBTagDeepCSV_4p5'
        ]
    }
    lumi = {# [1/pb] https://twiki.cern.ch/twiki/bin/view/CMS/LumiRecommendationsRun2
        '2016':  36.3e3,  # 35.8791
        '2016preVFP':  19.5e3,
        '2016postVFP': 16.5e3,
        '2017':  36.7e3,  # 36.7338
        '2018':  59.8e3,  # 59.9656
        'RunII': 132.8e3
    }
    # analysis
    blind = True

    corr      = True
    corr_syst = ...
    '''enable=`...` disable=`None`'''

    selections = ['pass6j', 'passmVjj']
    regions    = [0b10, 0b01, 0b00]
    '''SR=`0b10` SB=`0b01` other=`0b00`'''
    ntags      = [2, 3, 4]
    '''(n<3)=`2` (n=3)=`3` (n>3)=`4`'''
    puJetID_wp = 0b110
    '''disable=`0b000` Loose=`0b100` Medium=`0b110` Tight=`0b111`'''
    bTag_wp    = 0.6