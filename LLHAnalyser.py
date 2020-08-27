import numpy as np
from scipy import optimize, interpolate
from scipy.interpolate import UnivariateSpline
import scipy.special as sps

from iminuit import Minuit


class Profile_Analyser:

    def __init__(self):
        
        self.LLHtype = None

        self.ready = False
        self.signalPDF = None
        self.backgroundPDF = None

        self.signalPDF_uncert2 = None
        self.backgroundPDF_uncert2 = None

        self.nbins = 0
        
        self.livetime = -1.
        self.signalNormalization = -1.

        self.observation = None

        self.computedBestFit = False
        self.bestFit = None
        self.TS = None
        
        self.samplingMethod = 'default'
        
    def setLivetime(self,lt):
        self.livetime = lt

    def setLLHtype(self,type):
        availableTypes = ['Poisson']
        if type not in availableTypes:
            raise ValueError('LLH type not implemented yet. Choose amongst: '+availableTypes)
        else:
            self.LLHtype = type

    def loadBackgroundPDF(self,pdf):
        if self.livetime < 0:
            raise ValueError('Livetime of the analysis is not defined yet. Please do this first!')
        self.backgroundPDF = pdf.flatten()*self.livetime
        print 'total background events:', np.sum(self.backgroundPDF)
        self.nbins = len(pdf)

    def loadSignalPDF(self,pdf,norm):
        if self.livetime < 0:
            raise ValueError('Livetime of the analysis is not defined yet. Please do this first!')
        self.signalPDF = pdf.flatten()*self.livetime*norm
        self.signalNormalization = norm
        print 'total signal events:', np.sum(self.signalPDF)
        if self.nbins == len(pdf):
            self.ready = True
        else:
            raise ValueError('Shape of signal pdf does not match the background pdf! Did you initialize the background pdf first?')
            
    def loadUncertaintyPDFs(self,bkg_pdf,sig_pdf):
        self.backgroundPDF_uncert2 = bkg_pdf.flatten()*self.livetime*self.livetime        
        self.signalPDF_uncert2 = sig_pdf.flatten()*self.livetime*self.livetime

        if self.nbins != len(bkg_pdf):
            raise ValueError('Shape of background uncertainty pdf does not match the background pdf!')
        if self.nbins != len(sig_pdf):
            raise ValueError('Shape of background uncertainty pdf does not match the background pdf!')
 
    def sampleObservation(self,n1,nsig):

        if not self.ready:
            raise ValueError('Not all pdfs are correctly loaded!')

        observationPDF = n1*self.backgroundPDF + nsig*self.signalPDF
        self.observation=np.zeros(np.shape(self.backgroundPDF))
           
        for i in range(len(self.observation)):
            self.observation[i]=np.random.poisson(observationPDF[i])
                
        self.computedBestFit = False

        
    def evaluateLLH(self, n1, nsig):
        modelPDF = n1*self.backgroundPDF + nsig*self.signalPDF
        bins_to_use = (modelPDF>0.)

        if self.LLHtype == 'Poisson':
            values = self.observation[bins_to_use]*np.log(modelPDF[bins_to_use])-modelPDF[bins_to_use]

        elif self.LLHtype == 'Effective':
            modelPDF_uncert2 = n1*self.backgroundPDF_uncert2 + nsig*self.signalPDF_uncert2
    
            alpha = modelPDF[bins_to_use]**2/modelPDF_uncert2[bins_to_use] +1.
            beta  = modelPDF[bins_to_use]/modelPDF_uncert2[bins_to_use]

            values = [
              alpha*np.log(beta),
              sps.loggamma(self.observation[bins_to_use]+alpha).real,
              -(self.observation[bins_to_use]+alpha)*np.log1p(beta),
              -sps.loggamma(alpha).real,
            ]

        else:
            raise ValueError('No valid LLH type defined!')

        return -np.sum(values)

    
    def ComputeBestFit(self):
        LLHmin_DM=Minuit(self.evaluateLLH,
             nsig=1e-3,n1=1.,
             error_nsig=.1,error_n1=.1,
             limit_nsig=(-1.,100.),limit_n1=(0.,10.),
             errordef=.5,print_level=0)  
        LLHmin_DM.migrad()
        
        DM_fitarg_2=LLHmin_DM.fitarg
        LLHmin_DM_2=Minuit(self.evaluateLLH, errordef=.5, print_level=0,pedantic=True, **DM_fitarg_2)
        LLHmin_DM_2.migrad()

        self.bestFit = {}
        self.bestFit['n1']=LLHmin_DM_2.fitarg['n1']
        self.bestFit['nsig']=LLHmin_DM_2.fitarg['nsig']
        self.bestFit['LLH']=self.evaluateLLH(self.bestFit['n1'],self.bestFit['nsig'])
                
        self.computedBestFit = True
        
        
    def ComputeTestStatistics(self):
        
        if not self.computedBestFit:
            self.ComputeBestFit()

        LLHmin_ref=Minuit(self.evaluateLLH,
             nsig=0.,n1=1.,
             fix_nsig = True,
             error_nsig=.1,error_n1=.1,
             limit_nsig=(-1.,100.),limit_n1=(0.,10.),
             errordef=.5,print_level=0)  
        LLHmin_ref.migrad()
            
        self.TS = 0.

        self.bestFit['LLH_ref'] = self.evaluateLLH(LLHmin_ref.fitarg['n1'],LLHmin_ref.fitarg['nsig'])
        if self.bestFit['nsig'] > 0.:
            self.TS = 2*(self.bestFit['LLH_ref']-self.bestFit['LLH'])
        

    def CalculateUpperLimit(self,conf_level):

        nIterations = 0
        eps_TS=0.005
        eps_param=0.05

        deltaTS = 2.71
        if conf_level==90:
            deltaTS = 1.64
        elif conf_level==95:
            deltaTS = 2.71
            
        param_low=self.bestFit['nsig']
        param_up=self.bestFit['nsig']
        param_mean=self.bestFit['nsig']
        
        dTS=0
        cc=1
        while((dTS<deltaTS) and (nIterations<100)):
            nIterations += 1 

            param_up=param_up+3.*np.abs(param_up)

            LLHmin_fix=Minuit(self.evaluateLLH,
                 nsig=param_up,fix_nsig = True,
                 n1=1.,
                 error_nsig=.1,error_n1=.1,
                 limit_nsig=(-10.,100.),limit_n1=(0.,10.),
                 errordef=.5,print_level=0)  
            LLHmin_fix.migrad()

            if param_up <0.:
                TS_fix = 0.
            else:
                TS_fix = 2*(self.bestFit['LLH_ref']-self.evaluateLLH(LLHmin_fix.fitarg['n1'],param_up))

            dTS = self.TS - TS_fix
            
            
        nIterations = 0
        param_low=param_up/4.
        while((cc>0.)  and (nIterations<100)):
            
            nIterations += 1

            param_mean=(param_low+param_up)/2.
            LLHmin_fix=Minuit(self.evaluateLLH,
                 nsig=param_mean,fix_nsig = True,
                 n1=1., error_nsig=.1,error_n1=.1,
                 limit_nsig=(-10.,100.),limit_n1=(0.,10.),
                 errordef=.5,print_level=0)  
            LLHmin_fix.migrad()

            if param_mean <0.:
                TS_fix = 0.
            else:
                TS_fix = 2*(self.bestFit['LLH_ref']-self.evaluateLLH(LLHmin_fix.fitarg['n1'],param_mean))
            
            dTS = self.TS - TS_fix
            
            if(dTS<deltaTS):
                param_low=param_mean
                delta_param=(param_up-param_low)/(param_up)
                
                if((dTS>deltaTS-eps_TS) and (delta_param < eps_param)):
                    cc = 0
                    
            if(dTS>deltaTS):
                param_up=param_mean
                delta_param=(param_up-param_low)/(param_up)
                
                if((dTS<deltaTS+eps_TS) and (delta_param < eps_param)):
                    cc=0
                    
        return param_up
       
    
    def CalculateSensitivity(self,nTrials, conf_level):

        if self.LLHtype == None:
            raise ValueError('LLH type not defined yet!')

        upperlimits = []
        TS = []
        
        for i in tqdm(range(nTrials)):
            self.sampleObservation(1.,0.)
            self.ComputeTestStatistics()
            TS.append(self.TS)
            upperlimits.append(self.CalculateUpperLimit(conf_level))
                    
        p_median = np.percentile(upperlimits, 50)*self.signalNormalization
        p_95_low = np.percentile(upperlimits, 2.5)*self.signalNormalization
        p_95_high = np.percentile(upperlimits, 97.5)*self.signalNormalization
        p_68_low = np.percentile(upperlimits, 16.)*self.signalNormalization
        p_68_high = np.percentile(upperlimits, 84.)*self.signalNormalization

        dic_brazilian = {}
        dic_brazilian['TS_dist'] = TS
        dic_brazilian['error_68_low'] = p_68_low
        dic_brazilian['error_68_high'] = p_68_high
        dic_brazilian['error_95_low'] = p_95_low
        dic_brazilian['error_95_high'] = p_95_high   
        dic_brazilian['median'] = p_median

        return dic_brazilian
