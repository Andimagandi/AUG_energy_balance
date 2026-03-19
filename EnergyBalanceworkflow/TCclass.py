import numpy as np
import pandas as pd
from scipy.optimize import curve_fit, fsolve
from scipy.integrate import trapezoid as trapz
from scipy.interpolate import interp1d
import os
import pickle
from sklearn.metrics import r2_score
import sys
import importlib

class TCclass():
    def __init__(self, shotnumber, machine, Overview_TC, machine_spec, path, array_length, tag_in, tag_out):
        self.machine = machine
        self.shotnumber = shotnumber
        self.tag_in = tag_in
        self.tag_out = tag_out
        self.path = path
        self.array_length = array_length
        file1 = pd.read_csv(Overview_TC, index_col=0)
        file2 = file1.to_dict("split")
        self.OVERVIEW_TC = dict(zip(file2["index"], file2["data"]))
        self.REL_TC=list(self.OVERVIEW_TC.keys())   
        self.REL_TC=[x for x in self.REL_TC if 'time' not in x]  
        machine_specs = pd.read_csv(machine_spec, sep=',', comment='#', skipinitialspace=True)
        #machine-specific parameters, must be set before workflow use
        self.cut = machine_specs['cut_TC'].values #starting scheme #actual time values #needed
        self.reso = machine_specs['reso'].values #starting scheme #actual time values #needed
        self.cut_scan = machine_specs['cut_scan'].values #end runs #actual time values #must be checked
        self.uncertainty = machine_specs['uncertainty_TC'].values 
        self.material_properties = machine_specs['material_path'].values
        print('class to read in and evaluate thermocouple data is ready!')
        return    
    def read_in(self):
        #read in of data is machine-specific, must be set before workflow use
        self.data_file={}
        suffix = ''
        path_data = os.environ["homeDir"]+'/Input/'
        if self.path != 'nan':
            path_data = self.path
        if self.tag_in != 'nan':
            suffix = '_'+str(self.tag_in)+''
        Output = pickle.load(open(path_data+str(self.shotnumber)+suffix+'_TC.p', "rb" ))   
        filter = (Output['TimeBase']-Output['TimeBase'][0])<=self.array_length
        self.data_file['TimeBase']=(Output['TimeBase']-Output['TimeBase'][0])[filter]                        
        for i, iVal in enumerate(self.REL_TC):
            print('Reading in data of thermocouple '+str(iVal)+'')
            self.data_file[iVal]=Output[iVal][filter]             
        return
    def evaluate_data(self):  
        sys.path.append(str(os.environ["dataPath"]))
        material_properties = importlib.import_module(self.material_properties[0])
        self.output_TC={} 
        self.output_score = {}
        timebase_TC = self.data_file['TimeBase']
        for i, Signal in enumerate(self.REL_TC):
            print('Evaluating thermocouple '+str(Signal)+'')      
            #Reading in correct temperature signal
            data_signal=self.data_file[Signal]
            #Reading in mass and number of TC
            mass = float(self.OVERVIEW_TC[Signal][0])
            number = float(self.OVERVIEW_TC[Signal][1])     
            #Choosing the heat capacity      
            c_p = getattr(material_properties, self.OVERVIEW_TC[Signal][2])
            #Cutting off the first seconds due to some annoying peaks 
            TC_data=data_signal[timebase_TC>=0.0]    
            TC_time=timebase_TC[timebase_TC>=0.0]    
            #Defining cooling down function     
            target_value = 0.0
            def cooling_func(t,a):
                return (a_0-u_0)*np.exp(-a*(t-t_0))+u_0 - target_value
            j = 0
            score = []
            fitop = []  
            fitsuc = [] 
            while self.cut+j*self.reso <= self.cut_scan and self.cut+j*self.reso < TC_time[-1]:
                #Determining the maximum peak and decay profile
                ref_point = np.argmin(np.abs(np.trunc(TC_time) - (self.cut+j*self.reso)))
                temp_short,time_short=TC_data[ref_point:],TC_time[ref_point:]
                #Defining initial parameter, uncertainty and fitting cooling down coefficient   
                t_0=time_short[0]
                a_0=temp_short[0]
                if data_signal[0]>data_signal[-1]: 
                    u_0=data_signal[-1]
                else:
                    u_0=data_signal[0] 
                NAN=np.isnan(temp_short)
                for i,iVal in enumerate(NAN):
                    if NAN[i]==True:
                        temp_short[i]=temp_short[int(i-1)]
                Uncertainty = np.ones(len(temp_short))*self.uncertainty
                p_1, success = curve_fit(cooling_func, time_short, temp_short, maxfev=100000, bounds= (0, np.inf), sigma=Uncertainty)
                Success=np.sqrt(np.diagonal(success)) 
                if Success[0] > p_1[0]:
                    Success[0]=0.6827*p_1[0] #alternative value motivate by sigma definition: within the range of 1 sigma 68,27% of all data are found -> 68.27% as alternative value
                fitsuc.append(Success)    
                fitop.append(p_1)
                r2_score_fit = r2_score(temp_short, cooling_func(time_short,*p_1))
                score.append(r2_score_fit)
                j=j+int(1)

            #Determining full decay profile
            val = int(score.index(max(score)))
            p_1 = fitop[val]   
            ref_point = np.argmin(np.abs(np.trunc(TC_time) - (self.cut+val*self.reso)))
            a_0,t_0 = TC_data[ref_point], TC_time[ref_point]
            target_value = u_0
            t_0_guess = 1000 
            if p_1[0] <= 1e-5:
                p_1[0] = 0.0001
            t_converge = fsolve(cooling_func, t_0_guess, args=(*p_1,))    
            target_value = 0.0
            TC_time_extra = np.linspace(TC_time[0], int(t_converge[0]), int(t_converge[0]+1))
            self.output_TC[Signal] = (np.abs(trapz((c_p(cooling_func(TC_time_extra,*p_1))*mass),(cooling_func(TC_time_extra,*p_1)))))*number   
        return          
    def output(self):
        suffix = ''
        if self.tag_out != 'nan':
            suffix = ''+str(self.tag_out)+'_'
        print('TCs based energy distribution is stored under %s' %(''+str(os.environ["storagePath"])+''+str(suffix)+'TC_results.p'))   
        pickle.dump(self.output_TC , open(''+str(os.environ["storagePath"])+''+str(suffix)+'TC_results.p', "wb"))
        return
