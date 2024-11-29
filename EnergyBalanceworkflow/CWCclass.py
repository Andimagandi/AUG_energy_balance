import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.integrate import trapz
from scipy.interpolate import interp1d
import os
import pickle
import logging
from sklearn.metrics import r2_score
log = logging.getLogger(__name__)

class CWCclass():
    def __init__(self, shotnumber, machine, Overview_CWC, array_length, tag, mode):
        self.shotnumber = int(shotnumber)
        self.machine = machine
        self.array_length = array_length
        self.tag = tag
        self.mode = mode        
        file1 = pd.read_csv(Overview_CWC, index_col=0)
        file2 = file1.to_dict("split")
        self.OVERVIEW_CWC = dict(zip(file2["index"], file2["data"]))
        self.REL_CWC=list(self.OVERVIEW_CWC.keys())
        
        if self.machine == 'AUG':
            self.rel_CU_Out =  [x for x in self.REL_CWC if 'TGE' not in x and 'PT' not in x] 
                                 
        if self.machine == 'AUG':
            self.cut= 25 
            self.limit= 7200 
            self.uncertainty=0.06*2      
        
        print('class to read in and evaluate cooling water calorimetry data is ready!')
        log.info('class to read in and evaluate cooling water calorimetry data is ready!')
        return   
        
    def read_in(self):
        if self.machine=='AUG':
            try:
                import aug_sfutils as sf
                self.shotnumber_offset = sf.previousshot('KWU', self.shotnumber, 'KWU')
                Output=sf.SFREAD(self.shotnumber, 'KWU', experiment='AUGD') 
                Output_offset=sf.SFREAD(self.shotnumber_offset, 'KWU', experiment='KWU') 
            except:
                import pickle
                Output = pickle.load(open(os.environ["homeDir"]+'/Input/'+str(int(self.shotnumber))+'_CWC.p', "rb" ))   
                Output_offset = pickle.load(open(os.environ["homeDir"]+'/Input/'+str(int(self.shotnumber))+'_CWC_offset.p', "rb" ))      
        self.data_file={}
        self.data_file_offset={}
        try:
            self.data_file['TimeBase']=Output('TimeBase')[:int(self.array_length*60)]
            self.data_file_offset['TimeBase']=Output_offset('TimeBase')
        except:
            self.data_file['TimeBase']=Output['TimeBase'][:int(self.array_length*60)]
            self.data_file_offset['TimeBase']=Output_offset['TimeBase']
        for i, iVal in enumerate(self.REL_CWC):
           print('Reading in data of cooling unit '+str(iVal)+'')
           log.info('Reading in data of cooling unit '+str(iVal)+'')
           try:
               self.data_file[iVal]=Output(iVal)[:int(self.array_length*60)]
               self.data_file_offset[iVal]=Output_offset(iVal)
           except:           
               self.data_file[iVal]=Output[iVal][:int(self.array_length*60)]
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
            p_TG_offset=interp1d(timebase_CWC_offset, signal_offset_TG, kind='linear', fill_value='extrapolate') 
            time_offset = np.linspace(timebase_CWC_offset[0]-diff*mean_step, timebase_CWC_offset[-1]-diff*mean_step, len(timebase_CWC_offset))
            offset=p_TG_offset(time_offset)-signal_offset  
            data_signal=data_signal+np.mean(offset)
            
            p_TG_interp=interp1d(timebase_CWC, data_signal_TG, kind='linear', fill_value='extrapolate') 
            time_TG=np.linspace(timebase_CWC[0]-diff*mean_step, timebase_CWC[-1]-diff*mean_step, len(timebase_CWC))  
            p_TG = p_TG_interp(time_TG) 

            #Cutting off the first seconds due to some annoying peaks
            data_shortened,p_TG_shortened,time_shortened=data_signal[timebase_CWC>0.0],p_TG[timebase_CWC>0.0], timebase_CWC[timebase_CWC>0.0]                      
            fit_data_shortened=data_shortened-p_TG_shortened
            if fit_data_shortened[0]<0.0:
                fit_data_shortened=fit_data_shortened-fit_data_shortened[0]+0.01   
                
            ref_point=np.where(fit_data_shortened==fit_data_shortened.max())[0][-1]   
            #Defining cooling down function 
            if self.machine=='AUG':
                def cooling_func(t,a,b,c):
                    return a*(np.exp(-b*(t-t_0))+np.exp(-c*(t-t_0)))     
                if Signal[:3] in ('PHS'):
                    def cooling_func(t,a,b):
                        return a*np.exp(-b*(t-t_0))   
                                    
            j=0
            score = []
            fitop = [] 
            fitsuc = []  
            cut_scan=600
            reso = 10.0
            while (self.cut+float(j)*reso)<=cut_scan and int(ref_point+self.cut+float(j)*reso)<= (len(fit_data_shortened)-1):    
                temp_short,time_short=fit_data_shortened[(ref_point+int(self.cut+j*reso)):], time_shortened[(ref_point+int(self.cut+j*reso)):]
                temp_short_pre,time_short_pre,p_TG_shortened_pre=fit_data_shortened[:(ref_point+int(self.cut+j*reso)+1)], time_shortened[:(ref_point+int(self.cut+j*reso)+1)],p_TG_shortened[:(ref_point+int(self.cut+j*reso)+1)]
               
                #Defining initial parameter
                t_0=time_short[0]
                u_0=temp_short_pre[temp_short_pre>0.0].min()
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
                val = int(np.where(score==max(score))[0][0])
                self.output_score[''+str(Signal)+'score_time'] = np.linspace(0, (cut_scan-self.cut)/reso, int((cut_scan-self.cut)/reso+1))
                self.output_score[''+str(Signal)+'score_trace'] = score
                self.output_score[''+str(Signal)+'score_max_index'] = val
                self.output_score[''+str(Signal)+'ref_point'] = ref_point
                  
                new_time_extra=np.linspace(time_shortened[ref_point+int(self.cut+val*reso)], self.limit, int(self.limit-ref_point+int(self.cut+val*reso)+1))               
                temp_short_pre,time_short_pre,p_TG_shortened_pre=fit_data_shortened[:(ref_point+int(self.cut+val*10.0)+1)], time_shortened[:(ref_point+int(self.cut+val*10.0)+1)],p_TG_shortened[:(ref_point+int(self.cut+val*10.0)+1)]   
                
                p_1 = fitop[val]
                if self.mode == 'interpulse':
                    Success = fitsuc[val]
                    P_1 = p_1 - 3*Success  
                elif self.mode == 'offline':
                    P_1 = p_1            
                else:      
                    print('wrong mode is chosen!') 
                    
                t_0 = new_time_extra[0]       
                new_time_extra=new_time_extra[cooling_func(new_time_extra,*P_1)>=u_0]    
                   
                #Calculating CU energy deposition and storaging in dict
                energy_post=trapz((cooling_func(new_time_extra,*P_1)*c_v(p_TG_interp(new_time_extra))*dm_dt),new_time_extra)      
                energy_pre=trapz(temp_short_pre*c_v(p_TG_shortened_pre)*dm_dt,time_short_pre)  
                self.output_CWC[Signal] = (energy_post+energy_pre)

            except:
                #Calculating CU energy deposition and storaging in dict
                new_time_extra=time_shortened[fit_data_shortened>=fit_data_shortened[fit_data_shortened>0.0].min()]     
                p_TG_shortened_extra=p_TG_shortened[fit_data_shortened>=fit_data_shortened[fit_data_shortened>0.0].min()]
                new_temp_extra=fit_data_shortened[fit_data_shortened>=fit_data_shortened[fit_data_shortened>0.0].min()] 
                energy=trapz(new_temp_extra*c_v(p_TG_shortened_extra)*dm_dt,new_time_extra)  
                self.output_CWC[Signal] = (energy)   
                self.output_score[''+str(Signal)+'score_time'] = 0
                self.output_score[''+str(Signal)+'score_trace'] = 0
                self.output_score[''+str(Signal)+'score_max_index'] = 0
                self.output_score[''+str(Signal)+'ref_point'] = 0    
              
        return 
        
    def output(self):
        if self.tag =='nan':
            print('CUs based energy distribution is stored under %s' %(''+str(os.environ["storagePath"])+'CWC_results.p'))      
            log.info('CUs based energy distribution is stored under %s' %(''+str(os.environ["storagePath"])+'CWC_results.p'))
            pickle.dump(self.output_CWC , open(''+str(os.environ["storagePath"])+'CWC_results.p', "wb"))
            pickle.dump(self.output_score , open(''+str(os.environ["storagePath"])+'CWC_score_results.p', "wb"))   
        else:
            print('CUs based energy distribution is stored under %s' %(''+str(os.environ["storagePath"])+''+str(self.tag)+'_CWC_results.p'))      
            log.info('CUs based energy distribution is stored under %s' %(''+str(os.environ["storagePath"])+''+str(self.tag)+'_CWC_results.p'))        
            pickle.dump(self.output_CWC , open(''+str(os.environ["storagePath"])+''+str(self.tag)+'_CWC_results.p', "wb"))
            pickle.dump(self.output_score , open(''+str(os.environ["storagePath"])+''+str(self.tag)+'_CWC_score_results.p', "wb"))                   
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

def meanfil(data, it=5, res=5):
    for j in range(it):
      for i in range(len(data)-res):
          arr = np.zeros(int(2*res+1))
          for k in range(res):
            arr[-int(k+1)]=data[i-int(k+1)]
            arr[k]=data[i+int(k+1)]   
          arr[int(res+1)]=data[i]              
          mean=np.mean(arr)
          std=np.std(arr)
          if data[i]>(mean+std) or data[i]<[mean-std]:
              data[i]=mean  
    return data                
