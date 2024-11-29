import os
import sys
import pickle
import numpy as np
import pandas as pd
import logging
log = logging.getLogger(__name__)
import multiprocessing
import time
from tabulate import tabulate
import TCclass
import CWCclass

class initiate():
    def __init__(self, batchFile):
        self.start = time.time()
        print("Reading batch file")
        log.info("Reading batch file")
        self.caseDir= os.path.dirname(batchFile)

        #read batch file
        data = pd.read_csv(batchFile, sep=',', comment='#', skipinitialspace=True)
        self.pulses = data['shotnumber'].values
        self.machines = data['machine'].values
        self.tag = data['tag'].values        
        self.CWC_sheet = data['CWC_Input'].values     
        self.TC_sheet = data['TC_Input'].values
        self.array_length = data['array_length'].values
        self.mode = data['mode'].values
        self.Output = [x.split(":") for x in data['Output'].values]
        
        for ishot, shotnumber in enumerate(self.pulses):
            storagePath = '%s/%s/' %(os.environ['dataPath'], shotnumber)
            os.environ["storagePath"] = storagePath
            if not os.path.isdir(storagePath):
                print('New storage directory %s is created' %(os.environ["storagePath"]))
                log.info('New storage directory %s is created' %(os.environ["storagePath"]))
                os.system('mkdir -p '+str(storagePath)+'')
                
            if 'TC' in self.Output[ishot]:
                TCana = TCclass.TCclass(shotnumber, self.machines[ishot], self.TC_sheet[ishot], self.array_length[ishot], str(self.tag[ishot]), self.mode[ishot])
                TCana.read_in()
                TCana.evaluate_data()
                TCana.output()
            if 'CWC' in self.Output[ishot]:
                CWCana = CWCclass.CWCclass(shotnumber, self.machines[ishot], self.CWC_sheet[ishot], self.array_length[ishot], str(self.tag[ishot]), self.mode[ishot])
                CWCana.read_in()
                CWCana.evaluate_data()
                CWCana.output()
                
            print('Elapsed time: %f s' %(time.time()-self.start))
        return
