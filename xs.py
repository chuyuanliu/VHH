from vhh_formula import WHH4b_LO, ZHH4b_LO, VHH4b_LO
from heptools.physics import XSection

XSection.BRs = {
    'H->bb'  : 0.5824,   # mH = 125 https://twiki.cern.ch/twiki/bin/view/LHCPhysics/CERNYellowReportPageBR
    'Z->bb'  : 0.1512,   # PDG
    'W->qq'  : 0.6741,   # PDG
}

ZHH4b_samples = ZHH4b_LO(
    [1.0, 1.0,  1.0, 2.642e-04],
    [1.5, 1.0,  1.0, 5.738e-04],
    [0.5, 1.0,  1.0, 1.663e-04],
    [1.0, 2.0,  1.0, 6.770e-04],
    [1.0, 0.0,  1.0, 9.037e-05],
    [1.0, 1.0,  0.0, 1.544e-04],
    [1.0, 1.0,  2.0, 4.255e-04],
    [1.0, 1.0, 20.0, 1.229e-02]
)
WHH4b_samples = WHH4b_LO(
    [1.0, 1.0,  1.0, 4.152e-04],
    [1.5, 1.0,  1.0, 8.902e-04],
    [0.5, 1.0,  1.0, 2.870e-04],
    [1.0, 2.0,  1.0, 1.115e-03],
    [1.0, 0.0,  1.0, 1.491e-04],
    [1.0, 1.0,  0.0, 2.371e-04],
    [1.0, 1.0,  2.0, 6.880e-04],
    [1.0, 1.0, 20.0, 2.158e-02]
)

XSection.add(ZHH4b_samples, decay='H->bb**2', kfactors={'NNLO': 1.35394}) # GenXsecAnalyzer
XSection.add(WHH4b_samples, decay='H->bb**2', kfactors={'NLO': 1.22102}) # GenXsecAnalyzer
XSection.add(  r'^HH$', xs=0.03105) # https://twiki.cern.ch/twiki/bin/view/LHCPhysics/LHCHWGHH
XSection.add(  r'^ZH$', xs=0.7612 ) # https://twiki.cern.ch/twiki/bin/view/LHCPhysics/CERNYellowReportPageAt13TeV#ZH_Process
XSection.add(r'^ggZH$', xs=0.1227 ) # https://twiki.cern.ch/twiki/bin/view/LHCPhysics/CERNYellowReportPageAt13TeV#ZH_Process
XSection.add(  r'^ZZ$', xs=15.5   ) # https://arxiv.org/pdf/1607.08834.pdf#page=12
XSection.add(  r'^TT$', xs=831.76 ) # TODO TT xsec reference

XSection.add(r'^HH4b$', xs='[HH]', decay='H->bb**2')
XSection.add(r'^ZH4b$', xs='[ZH]', decay='Z->bb*H->bb')
XSection.add(r'^ZZ4b$', xs='[ZZ]', decay='Z->bb**2')
XSection.add(r'^ggZH4b$', xs='[ggZH]', decay='Z->bb*H->bb')
XSection.add(r'^bothZH4b$', xs='[ZH]+[ggZH]', decay='Z->bb*H->bb')
XSection.add(r'^TTTo2L2Nu$', xs='[TT]', decay='(1-W->qq)**2')
XSection.add(r'^TTToHadronic$', xs='[TT]', decay='W->qq**2')
XSection.add(r'^TTToSemiLeptonic$', xs='[TT]', decay='2*W->qq*(1-W->qq)')
XSection.add(VHH4b_LO._re_pattern, xs='["ZHHTo4B_CV_{CV}_C2V_{C2V}_C3_{C3}","NNLO"]+["WHHTo4B_CV_{CV}_C2V_{C2V}_C3_{C3}","NLO"]')