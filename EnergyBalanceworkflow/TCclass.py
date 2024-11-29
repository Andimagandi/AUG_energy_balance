import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.integrate import trapz
from scipy.interpolate import interp1d
import os
import csv
import pickle
import logging
from sklearn.metrics import r2_score
log = logging.getLogger(__name__)
import matplotlib.pyplot as plt

class TCclass():
    def __init__(self, shotnumber, machine, Overview_TC, array_length, tag, mode):
        self.machine = machine
        self.shotnumber = shotnumber
        self.array_length = array_length
        self.tag = tag
        self.mode = mode
        file1 = pd.read_csv(Overview_TC, index_col=0)
        file2 = file1.to_dict("split")
        self.OVERVIEW_TC = dict(zip(file2["index"], file2["data"]))
        self.REL_TC=list(self.OVERVIEW_TC.keys())   
        self.REL_TC=[x for x in self.REL_TC if 'time' not in x]  
        
        #machine-specific parameters, must be set before workflow use
        if self.machine == 'AUG':
            self.cut = 50 
            self.limit = 7200 
            self.uncertainty = 0.5*2  
                  
        print('class to read in and evaluate thermocouple data is ready!')
        log.info('class to read in and evaluate thermocouple data is ready!')
        return    

    def read_in(self):
        #read in of data is machine-specific, must be set before workflow use
        if self.machine=='AUG':
            self.data_file={}
            try:
                import aug_sfutils as sf
                Output=sf.SFREAD(self.shotnumber, 'KWU') 
                self.data_file['TimeBase']=Output('TimeBase')[:int(self.array_length*60)]
            except:
                import pickle
                Output = pickle.load(open(os.environ["homeDir"]+'/Input/'+str(int(self.shotnumber))+'_TC.p', "rb" ))   
            try:   
                self.data_file['TimeBase']=Output['TimeBase'][:int(self.array_length*60)]   
            except:
                self.data_file['TimeBase']=Output('TimeBase')[:int(self.array_length*60)]                            
            for i, iVal in enumerate(self.REL_TC):
               print('Reading in data of thermocouple'+str(iVal)+'')
               log.info('Reading in data of thermocouple'+str(iVal)+'')
               if iVal in ('TD04_03/04', 'TD08_03/04', 'TD10_03/04', 'TD13_03/04'):
                    try:
                        self.data_file[iVal]=np.mean(np.array([Output(iVal[:7]), Output(iVal[:6]+iVal[9:])]), axis=0)[:int(self.array_length*60)]
                    except:
                        self.data_file[iVal]=np.mean(np.array([Output[iVal[:7]], Output[iVal[:6]+iVal[9:]]]), axis=0)[:int(self.array_length*60)]
               elif iVal == 'TD08_07':
                    try:
                        self.data_file[iVal]=np.mean(np.array([Output('TD04_07'), Output('TD10_07'), Output('TD13_07')]), axis=0)[:int(self.array_length*60)]
                    except:
                        self.data_file[iVal]=np.mean(np.array([Output['TD04_07'], Output['TD10_07'], Output['TD13_07']]), axis=0)[:int(self.array_length*60)]
               else:
                    try:
                        self.data_file[iVal]=Output(iVal)[:int(self.array_length*60)]
                    except:
                        self.data_file[iVal]=Output[iVal][:int(self.array_length*60)]                    
        return
        
    def evaluate_data(self):  
        self.output_TC={} 
        self.output_score = {}
        timebase_TC = self.data_file['TimeBase']
        
        for i, Signal in enumerate(self.REL_TC):
            print('Evaluating thermocouple '+str(Signal)+'')      
            log.info('Evaluating thermocouple '+str(Signal)+'')
            
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
                          
            #Fudge factors for BGR1, required only AUG
            ratio=1
            if self.machine == 'AUG':
                if Signal in ('TD04_03/04','TD10_03/04','TD08_03/04','TD13_03/04'):
                    alpha=fieldangles(self.shotnumber, tshot=3.00)
                    alpha2=fieldangles(self.shotnumber, xtinp=0.0, tshot=3.00)
                    ratio=(alpha[0]/alpha2[0])[0][0] 
                       
            #Cutting off the first seconds due to some annoying peaks 
            TC_data=data_signal[timebase_TC>=0.0]    
            TC_time=timebase_TC[timebase_TC>=0.0]
            
            #Defining cooling down function     
            def cooling_func(t,a):
                return (a_0-u_0)*np.exp(-a*(t-t_0))+u_0
            
            j = 0
            score = []
            fitop = []  
            fitsuc = [] 
            cut_scan = 600
            reso = 10.0  
            while (self.cut+float(j)*reso)<= cut_scan and int((self.cut+float(j)*reso))<= (len(TC_time)-1):
                #Determining the maximum peak and decay profile
                ref_point=np.where(np.trunc(TC_time)==(self.cut+j*reso))[0][0]
                temp_short,time_short=TC_data[ref_point:],TC_time[ref_point:]
                
                #Defining initial parameter, uncertainty and fitting cooling down coefficient   
                t_0=time_short[0]
                a_0=temp_short[0]
                c_0=data_signal[0]
                if data_signal[0]>data_signal[-1]: 
                    u_0=data_signal[-1]
                else:
                    u_0=data_signal[0] 

                NAN=np.isnan(temp_short)
                for i,iVal in enumerate(NAN):
                    if NAN[i]==True:
                        temp_short[i]=temp_short[int(i-1)]
                Uncertainty = np.ones(len(temp_short))*self.uncertainty
                p_1, success = curve_fit(cooling_func, time_short, temp_short, maxfev=100000, sigma=Uncertainty)
                Success=np.sqrt(np.diagonal(success)) 
                if Success[0] > p_1[0]:
                    Success[0]=0.6827*p_1[0] #alternative value motivate by sigma definition: within the range of 1 sigma 68,27% of all data are found -> 68.27% as alternative value
                fitsuc.append(Success)
                    
                fitop.append(p_1)
                r2_score_fit = r2_score(temp_short, cooling_func(time_short,*p_1))
                score.append(r2_score_fit)
                j=j+int(1)
     
            #Determining full decay profile
            val = np.where(score==max(score))[0][0]
            p_1 = fitop[val]
            if self.mode == 'interpulse':
                Success = fitsuc[val]
                P_1 = p_1 + 3*Success  
            elif self.mode == 'offline':
                P_1 = p_1
            else:      
                print('wrong mode is chosen!') 
                     
            ref_point=np.where(np.trunc(TC_time)==(self.cut+val*reso))[0][0]
            self.output_score[''+str(Signal)+'score_time'] = np.linspace(0, (cut_scan-self.cut)/reso, int((cut_scan-self.cut)/reso+1))
            self.output_score[''+str(Signal)+'score_trace'] = score
            self.output_score[''+str(Signal)+'score_max_index'] = val
            self.output_score[''+str(Signal)+'ref_point'] =ref_point
            
            a_0,t_0=TC_data[ref_point], TC_time[ref_point]
            TC_time_extra = np.linspace(TC_time[0], self.limit, int(self.limit+1))
            TC_time_extra=TC_time_extra[cooling_func(TC_time_extra,*P_1)>=c_0]  
                            
            self.output_TC[Signal] = (np.abs(trapz((c_p(cooling_func(TC_time_extra,*P_1)+273.15)*mass),(cooling_func(TC_time_extra,*P_1)+273.15)))/ratio)*number   
            
        return          

    def output(self):
        if self.tag =='nan':
            print('TCs based energy distribution is stored under %s' %(''+str(os.environ["storagePath"])+'TC_results.p'))      
            log.info('TCs based energy distribution is stored under %s' %(''+str(os.environ["storagePath"])+'TC_results.p'))
            pickle.dump(self.output_TC , open(''+str(os.environ["storagePath"])+'TC_results.p', "wb"))
            pickle.dump(self.output_score , open(''+str(os.environ["storagePath"])+'TC_score_results.p', "wb"))        
        else:
            print('TCs based energy distribution is stored under %s' %(''+str(os.environ["storagePath"])+''+str(self.tag)+'_TC_results.p'))      
            log.info('TCs based energy distribution is stored under %s' %(''+str(os.environ["storagePath"])+''+str(self.tag)+'_TC_results.p'))
            pickle.dump(self.output_TC , open(''+str(os.environ["storagePath"])+''+str(self.tag)+'_TC_results.p', "wb"))
            pickle.dump(self.output_score , open(''+str(os.environ["storagePath"])+''+str(self.tag)+'_TC_score_results.p', "wb"))  
        return

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

#needed for AUG evaluation
def fieldangles(nshot, tshot, xtinp=0.019315, stinp=1.0698, anorm=0):
    '''
    ; Input
    ;
    ; nshot     - discharge number
    ; tshot     - time point to extract angle (s)
    ; xtinp     - toroidal distance from poloidal centre line of tile (m)
    ; stinp     - positions along cartesian axis in poloidal direction along
    ;             tile surface; it runs from bottom to top (s-coordinate)
    ; alpha     - vector of angle of intersecting field line to its projection on surface of interest
    ; bxt       - vector of field strength along tile surface in horizontal direction
    ; byt       - vector of field strength along tile surface in vertical   direction (direction of s)
    ; bzt       - vector of field strength perpendicular to tile surface
    ;
    ; Keyword
    ;
    ; anorm     - angle of surface of interest to target surface normal
    ;             0=target surface, 90=leading edge etc
    '''
    import aug_sfutils as sf
    xtinp=np.array([xtinp])
    stinp=np.array([stinp])
    
    slo = 0.9872    # lower edge of tile in s-coordinate (m)
    shi = 1.2028    # upper edge
    xlo = -0.038    # left edge
    xhi =  0.038    # right edge
    
    if min(stinp) <= slo or max(stinp) >= shi or min(xtinp) <= xlo or max(xtinp) >= xhi:
        print("Incompatible input: some points outside tile surface!")
    
    npol=len(stinp) #number of points to read
    ntor=len(xtinp) #number of points in toroidal direction (+-38mm tile width)
    
    #convert s to local tile coordinate
    ytinp = stinp-(shi+slo)/2.
    
    #crearting grid in x and y coordinate
    xtile, ytile=np.meshgrid(xtinp, ytinp)
    
    #rotation angles to rotate tile coordinate system
    phi = -14.99/180*np.pi #vertical inclination of target plate
    psi =  -0.60/180*np.pi #tilt angle of target plate to provide shadowing
    
    #torus coordinate of tile centre at surface
    r0 =  1.6073
    z0 = -1.1035
    
    #compute coordinates in Cartesian torus coordinates
    x = (np.cos(phi)*np.sin(psi)*xtile - np.sin(phi)*ytile) + r0
    y =          -np.cos(psi)*xtile
    z = (np.sin(phi)*np.sin(psi)*xtile + np.cos(phi)*ytile) + z0
    
    #radius in cylindrical coordinate system
    r = np.sqrt(x**2+y**2)
    
    #now we flatten the arrays to a 1D vector so that they can be digested by the kk-routines
    zflat=z.flatten()
    rflat=r.flatten()
    
    equ = sf.EQU(nshot, diag='EQI')
    output = sf.rz2brzt(equ, r_in=rflat, z_in=zflat, t_in=tshot)

    #transform B in cylindrical torus coordinates to B in Cartesian torus coordinates
    bx = output[0].reshape(npol,ntor)*x/r - output[2].reshape(npol,ntor)*y/r
    by = output[2].reshape(npol,ntor)*x/r + output[0].reshape(npol,ntor)*y/r
    
    #rotation angles to rotate tile coordinate system; np.cos in rad
    phi = +14.99/180*np.pi #vertical inclination of target plate
    psi =  -0.60/180*np.pi #tilt angle of target plate to provide shadowing
    
    #transform B-field back to tile coordinate system.
    bxt =  np.cos(phi)*np.sin(psi)*bx - np.cos(psi)*by - np.sin(phi)*np.sin(psi)*output[1].reshape(npol,ntor)
    byt =  np.sin(phi)         *bx               + np.cos(phi)         *output[1].reshape(npol,ntor)
    bzt = -np.cos(phi)*np.cos(psi)*bx - np.sin(psi)*by + np.sin(phi)*np.cos(psi)*output[1].reshape(npol,ntor)
    
    normb = np.sqrt(output[0].reshape(npol,ntor)**2+output[2].reshape(npol,ntor)**2+output[1].reshape(npol,ntor)**2)
    
    print("Total difference in B-norm for check:"+str(np.sum(normb-np.sqrt(bx**2+by**2+(output[1].reshape(npol,ntor))**2)))+", "+str(np.sum(normb-np.sqrt(bxt**2+byt**2+bzt**2)))+"")
    
    #alpha - angle of intersecting field line to its projection on tile surface
    cosalpha=np.cos(anorm/180*np.pi)*bzt/normb-np.sin(anorm/180*np.pi)*bxt/normb
    alpha = abs(90.-np.arccos(cosalpha)/np.pi*180)
    
    alpha = np.reshape(alpha,(npol,ntor))
    bxt = np.reshape(bxt,(npol,ntor))
    byt = np.reshape(byt,(npol,ntor))
    bzt = np.reshape(bzt,(npol,ntor))
    
    return alpha, bxt,byt,bzt
