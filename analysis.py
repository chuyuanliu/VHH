import os

import awkward as ak
import heptools
import numpy as np
import uproot
from coffea import processor
from coffea.nanoevents import NanoAODSchema, NanoEventsFactory
from heptools.aktools import or_fields, sort_field, update_fields, where
from heptools.cms import (BTagSF_Shape, PileupJetIDSF, PileupWeight,
                          jsonPOG_integration)
from heptools.correction import EventWeight
from heptools.hist import Collection, Fill, Systematic
from heptools.physics.object import Jet, LorentzVector

from config import VHH_2j4b as VHH
from schemas import MultiClassifierSchema
from xs import XSection

NanoAODSchema.warn_missing_crossrefs = False

def corr_file(era, file):
    return jsonPOG_integration(era, file) if VHH.corr else None

class analysis(processor.ProcessorABC):
    def process(self, events):
        np.random.seed(0)
        events.behavior |= heptools.behavior

        fname   = events.metadata['filename']
        estart  = events.metadata['entrystart']
        estop   = events.metadata['entrystop']
        process = events.metadata['process']
        era     = events.metadata['era']
        year    = era[0:4]
        isMC    = events.metadata.get('isMC',  False)

        # weight
        weight = EventWeight()
        if isMC:
            with uproot.open(fname) as rfile:
                Runs = rfile['Runs']
                genEventSumw = np.sum(Runs['genEventSumw'])
            weight.add('genWeight', 'genWeight')
            weight.add('genWeightNorm', 1/genEventSumw)
            weight.add('xs', XSection(process, 'NLO', 'NNLO'))
            weight.add('lumi', VHH.lumi[era])
            weight.add('prefiring', ('L1PreFiringWeight', 'Nom'), up = ('L1PreFiringWeight', 'Up'), down = ('L1PreFiringWeight', 'Dn'), off = 1)
            weight.add('triggerWeight', ('trigWeight', 'Data'), sim = 'passHLT', emu_MC = ('trigWeight', 'MC'), off = 1)
            weight.add('btagSF', **BTagSF_Shape(corr_file(f'{era}_UL', 'btagging'), VHH.corr_syst, jet = 'selJet'))
            weight.add('puJetIDSF', **PileupJetIDSF(corr_file(f'{era}_UL', 'jmar'), VHH.corr_syst, jet = 'selJet', working_point=VHH.puJetID_wp))
            weight.add('puWeight', **PileupWeight(corr_file(f'{era}_UL', 'puWeights'), VHH.corr_syst), off = 1)
            if 'trigWeight' not in events.fields:
                weight.add('triggerWeight', 1)

        # hists
        fill = Fill(process = process, era = era, weight = ('weights', 'weight'))
        hist = Collection(process = [],
                             era     = VHH.eras,
                             tag     = VHH.ntags + [-3],
                             region  = VHH.regions,
                             kvvkl   = [-1.0, 0.0, 1.0],
                             **dict((s, ...) for s in VHH.selections))
        cutflow = fill + hist.add('cutflow')

        fill += hist.add('SvB_MA_ps', (100, 0, 1, ('SvB_MA.VHH_ps', 'SvB_MA Regressed P(VHH)')))
        fill += LorentzVector.plot(('selJets', 'Selected Jets'), 'selJet', count = True)
        fill += LorentzVector.plot(('canJets', 'Higgs Candidate Jets'), 'canJet')
        fill += LorentzVector.plot(('othJets', 'Other Jets'), 'othJet', count = True)
        fill += LorentzVector.plot_pair(('p4bHH', R'$HH_{4b}$'), 'p4bHH')
        fill += LorentzVector.plot_pair(('p2jOth', R'Other Dijets'), 'p2jOth')
        fill += LorentzVector.plot_pair(('p2j', R'Vector Boson Candidate Dijets'), 'p2jV')
        fill += Systematic('SvB_MA_ps', weight, weight = 'weights')

        # BDT, SvB
        events['kvvkl'] = events.BDT.kl
        events['SvB_MA'] = NanoEventsFactory.from_root(os.path.join(os.path.dirname(fname), VHH.SvB), entry_start=estart, entry_stop=estop, schemaclass=MultiClassifierSchema).events().SvB_MA

        # HLT
        events['passHLT'] = or_fields(events.HLT, *VHH.HLT[year])

        # selected jets
        events['Jet', 'pileup'] = (events.Jet.puId < VHH.puJetID_wp) & (events.Jet.pt < 50)
        events['selJet'] = events.Jet[(events.Jet.pt >= 40) & (np.abs(events.Jet.eta) <= 2.4) & ~events.Jet.pileup]
        events['nSelected'] = ak.num(events.selJet)
        events['pass6j'] = events.nSelected >= 6
        # b-tagging
        events['selJet', 'bTagged'] = events.selJet.btagDeepFlavB >= VHH.bTag_wp
        events['selJet', 'bTagged_loose'] = events.selJet.btagDeepFlavB >= (VHH.bTag_wp/2)
        events['nBTagged'] = ak.sum(events.selJet.bTagged, axis = 1)
        events['nBTagged_loose'] = ak.sum(events.selJet.bTagged_loose, axis = 1)
        events['tag'] = where(events.nBTagged_loose, (events.nBTagged > 3, 4), (events.nBTagged_loose < 3, 2), ((events.nBTagged_loose > 3) & (events.nBTagged < 4), -3))

        # failed events
        cutflow(events[events.nSelected < 4], weight = lambda x: weight(x, 'btagSF', 'puJetIDSF', 'puWeight', 'prefiring', 'trigger').weight, region = 0b00, **dict((s, False)for s in VHH.selections))
        ### cut ###
        events = events[events.pass6j]

        # 2 higgs bosons
        events['selJet'] = sort_field(events.selJet, 'btagDeepFlavB')
        events['canJet'] = sort_field(events.selJet[:, 0:4], 'pt')
        update_fields(events.canJet, events.canJet * events.canJet.bRegCorr)
        events['othJet'] =  sort_field(events.selJet[:, 4:], 'pt')
        h_d  = Jet.pair(events.canJet, mode = 'combination', combinations = 2)
        hh_q = Jet.pair(h_d[:, :, 0], h_d[:, :, 1])

        _mdr = ((360/hh_q.mass -0.5 < hh_q.lead_st.dr) & (hh_q.lead_st.dr < np.maximum(650/hh_q.mass + 0.5, 1.5)) + 
               (235/hh_q.mass      < hh_q.subl_st.dr) & (hh_q.subl_st.dr < np.maximum(650/hh_q.mass + 0.7, 1.5)))
        _xHH = np.sqrt(((hh_q.lead_st.mass - 125 * 1.02)/(0.1 * hh_q.lead_st.mass))**2 + 
                       ((hh_q.subl_st.mass - 125 * 0.98)/(0.1 * hh_q.subl_st.mass))**2)

        hh_q['passDijetMass'] = (
            (52 < hh_q.lead_st.mass ) & (hh_q.lead_st.mass < 180) &
            (50 < hh_q.subl_st.mass ) & (hh_q.subl_st.mass < 173)
        )
        hh_q['rankMDR'] = hh_q.passDijetMass * 10 + _mdr + np.random.uniform(low=0.1, high=0.9, size=(len(hh_q), 3))
        hh_q['SR'] = _xHH < 1.9
        hh_q['SB'] = hh_q.passDijetMass & ~hh_q.SR
        hh_q = sort_field(hh_q, 'rankMDR')
        events['region'] = hh_q[:, 0].SR * 0b10 + hh_q[:, 0].SB * 0b01
        events['p4bHH'] = hh_q[:, 0]

        # 1 vector boson
        v_d = Jet.pair(events.othJet, mode = 'combination')
        v_d = sort_field(v_d, 'pt')
        events['p2jOth'] = v_d
        v_d = v_d[(65 < v_d.mass) & (v_d.mass < 105)]
        events['p2jV'] = v_d
        events['passmVjj'] = ak.num(events.p2jV) > 0

        # blind data
        ### cut ###
        if not isMC and VHH.blind:
            events = events[~((events.region == 0b10) & (events.tag == 4))]

        # calculate weight
        events['weights'] = weight(events)

        # fill hists
        cutflow(events)
        fill(events)

        return hist.output

    def postprocess(self, accumulator):
        ...