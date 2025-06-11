import numpy as np
import pandas as pd
from scipy.optimize import curve_fit, fsolve
from scipy.integrate import trapezoid as trapz
from scipy.interpolate import interp1d
import os
import pickle
from sklearn.metrics import r2_score

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
            if self.OVERVIEW_TC[Signal][2] == 'carbon':                 
                c_p=c_p_carbon
            if self.OVERVIEW_TC[Signal][2] == 'tungsten': 
                c_p=c_p_tungsten
            if self.OVERVIEW_TC[Signal][2] == 'steel':
                c_p=c_p_steel  
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
            TC_time_extra = np.linspace(TC_time[0], int(t_converge), int(t_converge+1))
            self.output_TC[Signal] = (np.abs(trapz((c_p(cooling_func(TC_time_extra,*p_1)+273.15)*mass),(cooling_func(TC_time_extra,*p_1)+273.15))))*number   
        return          
    def output(self):
        suffix = ''
        if self.tag_out != 'nan':
            suffix = ''+str(self.tag_out)+'_'
        print('TCs based energy distribution is stored under %s' %(''+str(os.environ["storagePath"])+''+str(suffix)+'TC_results.p'))   
        pickle.dump(self.output_TC , open(''+str(os.environ["storagePath"])+''+str(suffix)+'TC_results.p', "wb"))
        return
# compute heat capacity vs T
#source:
#http://www.metalspiping.com/typical-physical-properties-of-p92-t92-steel.html 
#heat capacity originally defined in dimension J/(kg*K), but input for fit formula in Celsius
def c_p_steel(temp):
    c_p=np.array([420,420,430,450,460,470,480,500,510,530,580,600,630,640])
    temp_ori=np.array([20,50,100,150,200,250,300,350,400,450,500,550,600,650])+273.15
    f = interp1d(temp_ori, c_p, kind='cubic', fill_value='extrapolate')
    TC_poly = np.polyfit(temp_ori, f(temp_ori), 4)
    p = np.poly1d(TC_poly)
    return p(temp)
#source: Optimization of experimental data on the heat capacity, volume, and bulk moduli of minerals
#required for determination fit parameters
#values taken from table 2
#heat capacity of tungsten at constant pressure c_p!!! 
#heat capacity originally defined in dimension J/(kg*K), input for fit formula in Kelvin
#output given in J/(kg*K)
def c_p_tungsten(temp):
    new=np.array([100,200,300,400,500,600,700,800,900,1000,1200,1400,1600,1800,2000,2200,2400,2600,2800,3000,3200,3400,3600])
    alt=np.array([15.55,22.24,24.19,25.09,25.67,26.11,26.49,26.83,27.16, 27.48,28.14,28.87,29.74,30.80,32.05,33.52,35.18,37.03,39.06,41.26,43.68,46.35,49.39])
    f = interp1d(new, alt*1000./183.84, kind='quadratic', fill_value='extrapolate')
    TC_poly = np.polyfit(new,  f(new), 16)
    p = np.poly1d(TC_poly)
    return p(temp)
#source: JOURNAL OF NUCLEAR MATERIALS 49 (1973/74) 45-56.0 NORTH-HOLLAND PUBLISHING COMPANY
#THE SPECIFIC HEAT OF GRAPHITE: AN EVALUATION OF MEASUREMENTS
#A.T.D. BUTLAND and R.J. MADDISON 
#heat capacity of carbon at constant pressure c_p!!! 
#values taken from table 7 'unadjusted polynomial'
#heat capacity originally defined in dimension cal/(g*K), input for fit formula in Kelvin!
#output given in J/(kg*K)
def c_p_carbon(temp):
    c_p=np.array([0.17035, 0.36786, 0.42854, 0.47727, 0.49409])*4186.8
    temp_alt=np.array([300,700,1000,1500,1800])
    temp_2 = np.linspace(273.15,2000,1001)
    f = interp1d(temp_alt, c_p, kind='quadratic', fill_value='extrapolate')
    TC_poly = np.polyfit(temp_2, f(temp_2), 10)
    p = np.poly1d(TC_poly)
    return p(temp)
