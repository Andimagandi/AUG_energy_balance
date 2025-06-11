import numpy as np
import pandas as pd
from scipy.optimize import curve_fit, fsolve
from scipy.integrate import trapezoid as trapz
from scipy.interpolate import interp1d
import os
import sys
import pickle
from sklearn.metrics import r2_score

class CWCclass():
    def __init__(self, shotnumber, shotnumber_offset, machine, Overview_CWC, machine_spec, path, array_length, tag_in, tag_out):
        self.shotnumber = int(shotnumber)
        self.shotnumber_offset = int(shotnumber_offset)
        self.machine = machine
        self.array_length = array_length
        self.tag_in = tag_in
        self.tag_out = tag_out    
        self.path = path    
        file1 = pd.read_csv(Overview_CWC, index_col=0)
        file2 = file1.to_dict("split")
        self.OVERVIEW_CWC = dict(zip(file2["index"], file2["data"]))
        self.REL_CWC=list(self.OVERVIEW_CWC.keys())
        self.rel_CU_Out =  [x for x in self.REL_CWC if 'TGE' not in x and 'PT' not in x]    
        machine_specs = pd.read_csv(machine_spec, sep=',', comment='#', skipinitialspace=True)
        #machine-specific parameters, must be set before workflow use
        self.cut = machine_specs['cut_CWC'].values #actual time values
        self.cut_scan = machine_specs['cut_scan'].values #actual time values
        self.reso = machine_specs['reso'].values #actual time values   
        self.uncertainty = machine_specs['uncertainty_CWC'].values
        print('class to read in and evaluate cooling water calorimetry data is ready!')
        return   
    def read_in(self):
        self.data_file={}
        self.data_file_offset={}
        suffix = ''
        path_data = os.environ["homeDir"]+'/Input/'
        if self.path != 'nan':
            path_data = self.path
        if self.tag_in != 'nan':
            suffix = '_'+str(self.tag_in)+''
        Output = pickle.load(open(path_data+str(int(self.shotnumber))+suffix+'_CWC.p', "rb"))   
        Output_offset = pickle.load(open(path_data+str(int(self.shotnumber_offset))+suffix+'_CWC_offset.p', "rb"))   
        filter = (Output['TimeBase']-Output['TimeBase'][0])<=self.array_length
        self.data_file['TimeBase']=(Output['TimeBase']-Output['TimeBase'][0])[filter]
        self.data_file_offset['TimeBase']=Output_offset['TimeBase']
        for i, iVal in enumerate(self.REL_CWC):
            print('Reading in data of cooling unit '+str(iVal)+'')
            self.data_file[iVal]=Output[iVal][filter]
            self.data_file_offset[iVal]=Output_offset[iVal]
        return                
    def evaluate_data(self):
        self.output_CWC={} 
        self.output_score = {}
        timebase_CWC = np.array(self.data_file['TimeBase'])
        timebase_CWC_offset = np.array(self.data_file_offset['TimeBase'])
        for i, Signal in enumerate(self.rel_CU_Out):
            print('Evaluating cooling unit '+str(Signal)+'')
            data_signal=meanfil(np.array(self.data_file[Signal]))
            data_signal_TG=meanfil(np.array(self.data_file[self.OVERVIEW_CWC[Signal][2]]))
            signal_offset=meanfil(np.array(self.data_file_offset[Signal]))
            signal_offset_TG=meanfil(np.array(self.data_file_offset[self.OVERVIEW_CWC[Signal][2]]))                
            mean_step=np.mean(np.diff(timebase_CWC))
            #Reading in the flow rates for each CU
            dm_dt=float(self.OVERVIEW_CWC[Signal][0])/3600.
            #Defining time delay between low and return temperature
            diff=int(self.OVERVIEW_CWC[Signal][1]/mean_step)
            #correct time difference between flow and return temperature
            p_TG_offset=interp1d(timebase_CWC_offset, signal_offset_TG, kind='nearest', fill_value='extrapolate') 
            time_offset = np.linspace(timebase_CWC_offset[0]-diff*mean_step, timebase_CWC_offset[-1]-diff*mean_step, len(timebase_CWC_offset))
            offset=p_TG_offset(time_offset)-signal_offset  
            data_signal=data_signal+np.mean(offset)
            p_TG_interp=interp1d(timebase_CWC, data_signal_TG, kind='nearest', fill_value='extrapolate') 
            time_TG=np.linspace(timebase_CWC[0]-diff*mean_step, timebase_CWC[-1]-diff*mean_step, len(timebase_CWC))  
            p_TG = p_TG_interp(time_TG) 
            #Cutting off the first seconds due to some annoying peaks
            data_shortened,p_TG_shortened,time_shortened=data_signal[timebase_CWC>0.0],p_TG[timebase_CWC>0.0], timebase_CWC[timebase_CWC>0.0]                      
            fit_data_shortened=data_shortened-p_TG_shortened
            if fit_data_shortened[0]<0.0:
                fit_data_shortened=fit_data_shortened-fit_data_shortened[0]+0.01   
            start_point=time_shortened[np.where(fit_data_shortened==fit_data_shortened.max())[0][-1]]
            #Defining cooling down function 
            target_value = 0.0
            def cooling_func(t,a,b,c):
                return a*(np.exp(-b*(t-t_0))+np.exp(-c*(t-t_0))) - target_value    
            j=0
            score = []
            fitop = [] 
            fitsuc = []  
            while self.cut+j*self.reso <=self.cut_scan and start_point+self.cut+j*self.reso < time_shortened[-1]:    
                ref_point = np.argmin(np.abs(np.trunc(time_shortened) - (start_point+self.cut+j*self.reso)))
                temp_short,time_short=fit_data_shortened[int(ref_point):], time_shortened[int(ref_point):]
                temp_short_pre,time_short_pre,p_TG_shortened_pre=fit_data_shortened[:int(ref_point)], time_shortened[:int(ref_point+1)],p_TG_shortened[:int(ref_point+1)]
                #Defining initial parameter
                t_0=time_short[0]
                Uncertainty = np.ones(len(temp_short))*self.uncertainty
                #Fitting cooling down coefficient
                p_1, success = curve_fit(cooling_func, time_short, temp_short, sigma=Uncertainty, bounds=(0, np.inf), maxfev=100000)
                Success=np.sqrt(np.diagonal(success)) 
                for t,tVal in enumerate(p_1):
                    if Success[t] > p_1[t]:
                        Success[t]=0.6827*p_1[t] #alternative value motivate by sigma definition: within the range of 1 sigma 68,27% of all data are found -> 68.27% as alternative value
                fitsuc.append(Success)
                fitop.append(p_1)    
                try:
                    r2_score_fit = r2_score(temp_short, cooling_func(time_short,*p_1))
                except:
                    r2_score_fit = -1
                score.append(r2_score_fit)                    
                j=j+int(1)    
            #Determining full decay profile
            try:
                val = int(score.index(max(score)))
                ref_point = np.argmin(np.abs(np.trunc(time_shortened) - (start_point+self.cut+val*self.reso)))
                temp_short_pre,time_short_pre,p_TG_shortened_pre=fit_data_shortened[:(ref_point+1)], time_shortened[:(ref_point+1)],p_TG_shortened[:(ref_point+1)]   
                p_1 = fitop[val]
                t_0 = time_shortened[ref_point]
                target_value = fit_data_shortened[0]
                t_0_guess = 1000  # initial guess for t
                for i in range(len(p_1[1:])):
                    if p_1[int(i+1)] <= 1e-5:
                        p_1[int(1+i)] = 0.0001
                t_converge = fsolve(cooling_func, t_0_guess, args=(*p_1,))                
                new_time_extra = np.linspace(t_0, int(t_converge), int(t_converge+1))      
                target_value = 0.0
                #Calculating CU energy deposition and storaging in dict
                energy_post=trapz((cooling_func(new_time_extra,*p_1)*c_v(p_TG_interp(new_time_extra))*dm_dt),new_time_extra)      
                energy_pre=trapz(temp_short_pre*c_v(p_TG_shortened_pre)*dm_dt,time_short_pre)  
                self.output_CWC[Signal] = (energy_post+energy_pre)  
            except:
                #Calculating CU energy deposition and storaging in dict
                new_time_extra=time_shortened[fit_data_shortened>=fit_data_shortened[fit_data_shortened>0.0].min()]     
                p_TG_shortened_extra=p_TG_shortened[fit_data_shortened>=fit_data_shortened[fit_data_shortened>0.0].min()]
                new_temp_extra=fit_data_shortened[fit_data_shortened>=fit_data_shortened[fit_data_shortened>0.0].min()] 
                energy=trapz(new_temp_extra*c_v(p_TG_shortened_extra)*dm_dt,new_time_extra)  
                self.output_CWC[Signal] = (energy)   
        return 
    def output(self):
        suffix = ''
        if self.tag_out != 'nan':
            suffix = ''+str(self.tag_out)+'_'
        print('CUs based energy distribution is stored under %s' %(''+str(os.environ["storagePath"])+''+str(suffix)+'CWC_results.p'))      
        pickle.dump(self.output_CWC , open(''+str(os.environ["storagePath"])+''+str(suffix)+'CWC_results.p', "wb"))
        return             
#from scipy.interpolate import interp1d
#source:https://www.engineeringtoolbox.com/specific-heat-capacity-water-d_660.html
#required for determination fit parameters
#temp= np.array([0.01, 10, 20, 25, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 140, 160, 180, 200, 220, 240, 260, 280, 300, 320, 340, 360])      
#c_v=np.array([4.2174, 4.1910, 4.1570, 4.1379, 4.1175, 4.0737, 4.0264, 3.9767, 3.9252, 3.8729, 3.8204, 3.7682, 3.7167, 3.6662, 3.5694, 3.4788, 3.3949, 3.3179, 3.2479, 3.1850, 3.1301, 3.0849, 3.0530, 3.0428, 3.0781, 3.2972])
#f = interp1d(temp, c_v, kind='quadratic', fill_value='extrapolate')
#c_v_fit = f(temp)
#TG_poly = np.polyfit(temp, c_w_fit, 6)
#p = np.poly1d(TG_poly)
#heat capacity of eater at constant volume c_v!!! 
#heat capacity originally defined in dimension kJ/(kg*K), input for fit formula in Celsius
def c_v(temp):
    def func(x,*p_0):
        return p_0[0]*x**6.+p_0[1]*x**5+p_0[2]*x**4.+p_0[3]*x**3.+p_0[4]*x**2.+p_0[5]*x+p_0[6]
    new=np.array([1.155e-14,- 9.946e-12, 2.969e-09, - 3.067e-07, - 2.203e-06, - 0.003337, 4.223])
    return func(temp, *new)*1000.
def meanfil(data,res=5):
    for i in range(res, len(data) - res):  # Avoid out-of-bounds indices
        # Create the window of 11 values centered around data[i]
        window = data[i - res:i + res + 1]  
        mean = np.median(window)
        std = np.std(window)
        # Replace outliers with the mean
        if abs(data[i] - mean) > std:
            data[i] = mean        
    return data            
