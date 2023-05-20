import os
import pickle
import re
import time

from coffea import processor
from coffea.nanoevents import NanoAODSchema

from analysis import analysis
from config import VHH_2j4b as VHH

if __name__ == '__main__':
    fileset = {}
    for year in VHH.eras.mc:
        for signal in VHH.MC_signal:
            folder = f'{signal}_{{year}}/'.format(year = re.sub(r'(pre|post)VFP', r'_\g<0>', year))
            path = os.path.join(VHH.input, folder, 'picoAOD.root')
            fileset[f'{signal}_{year}'] = {
                'files': [path, ],
                'metadata': {
                    'process': signal,
                    'era'    : year,
                    'isMC'   : True,
                }}

    tstart = time.time()
    output = processor.run_uproot_job(
        fileset,
        treename='Events',
        processor_instance=analysis(),
        executor=processor.futures_executor,
        executor_args={'schema': NanoAODSchema, 'workers': 4},
        chunksize=100_000,
    )
    elapsed = time.time() - tstart
    print(f'total {elapsed}s')

    with open(f'{VHH.output}/VHH_hists.pkl', 'wb') as hfile:
        pickle.dump(output, hfile)