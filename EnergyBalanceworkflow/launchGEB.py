#1st attempt GEB workflow, A. Redl 19.07.2024
#python launchGEB.py --f workflowinput_inter.dat

import os
import sys
import argparse
import GEBclass    
                        
def loadEnviron():
    #load energy balance workflow environment
    #default home directory
    try:
        homeDir = '%s' %os.getcwd()
    except:
        print("HOME env var not set. Set before running energy balance workflow!")
        sys.exit()
    #default data storage directory
    dataPath = homeDir + '/GEB'
    #Set relevant environment variables
    os.environ["homeDir"] = homeDir
    os.environ["dataPath"] = dataPath
    os.umask(0)
    #create dataPath
    if not os.path.exists(dataPath):
        os.makedirs(dataPath)
    return
def launchGEB(args):
    batchFile = vars(args)['f']
    if batchFile == 'None':
        print('No batchfile.dat found in arguments. Required for TUI, aborting')
        sys.exit()
    print('batch file exits!')     
    GEB = GEBclass.initiate(batchFile)
    return
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="""Energy balance workflow... Use this command to launch the Energy Balance workflow. You can run the workflow in terminal only.""")
    parser.add_argument('--f', type=str, help='Batch file path', required=False)
    args = parser.parse_args()
    loadEnviron()
    launchGEB(args)
