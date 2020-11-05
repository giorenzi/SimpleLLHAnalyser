import numpy as np
import scipy.special as sps
from iminuit import Minuit

class Profile_Analyser:

    def __init__(self):
        
        self.LLHtype = None

        self.Nevents = 0.
        
        self.ready = False
        self.norm = True
        self.signalPDF = None
        self.backgroundPDF = None
        self.PDFs = dict()

        self.signalPDF_uncert2 = None
        self.backgroundPDF_uncert2 = None
        self.PDFs_uncert2 = dict()

        self.nbins = 0
        
        self.livetime = -1.
        self.observation = None

        self.computedBestFit = False
        self.bestFit = None
        self.TS = None
        
        self.samplingMethod = 'default'
        self.moreOutput = False
        
    def setLivetime(self,lt):
        self.livetime = lt
        
    def setNevents(self, N):
        self.Nevents = N

    def setLLHtype(self,type):
        availableTypes = ['Poisson', 'Effective']
        if type not in availableTypes:
            raise ValueError('LLH type not implemented yet. Choose amongst: '+availableTypes)
        else:
            self.LLHtype = type
            
    def DoNotNormalisePDFs(self):
        print("PDFs will not be normalised")
        self.norm = False
            
    def saveMoreOutput(self):
        self.moreOutput = True

    def loadPDF(self,pdf,name):
        self.ready=False
        if self.livetime < 0:
            raise ValueError('Livetime of the analysis is not defined yet. Please do this first!')
        self.PDFs[name] = pdf.flatten()*self.livetime
        if self.norm:
            print("PDFs are being normalised")
            self.PDFs[name]/=np.sum(self.PDFs[name])
        print('total {0} events: {1}'.format(name, np.sum(self.PDFs[name])))
        if self.nbins == 0:
            self.nbins = len(self.PDFs[name])
            print('using number {0} of bins'.format(self.nbins))
        else:
            if self.nbins == len(self.PDFs[name]):
                self.ready = True
            else:
                raise ValueError('Shape of {0} pdf does not match the background pdf!'.format(name)+\
                                 'Did you initialize the background pdf first?')
            
    def loadPDFandUncertainties(self, pdf, name):
        self.ready=False
        if self.livetime < 0:
            raise ValueError('Livetime of the analysis is not defined yet. Please do this first!')
        if type(pdf) != tuple:
            raise ValueError('PDF must be a tuple of type (pdf, uncertainty_pdf)')
        if len(pdf) != 2:
            raise ValueError('PDF must be a tuple of type (pdf, uncertainty_pdf)') 
        self.PDFs[name] = pdf[0].flatten()*self.livetime
        self.PDFs_uncert2[name] = pdf[1].flatten()*self.livetime*self.livetime
        if self.norm:
            print("PDFs are being normalised")
            self.PDFs[name]/=np.sum(self.PDFs[name])
            self.PDFs_uncert2[name]/=np.sum(self.PDFs[name])**2
        #if self.nbins = len(pdf[0]):
        #    self.ready = True
        #else:
        #    raise ValueError('Shape of {0} pdf does not match the background pdf!'.format(name)+\
        #                     'Did you initialize the background pdf first?')
        self.ready = True
            
    def loadBackgroundPDF(self,pdf):
        self.loadPDF(pdf,'bkg')
        '''
        if self.livetime < 0:
            raise ValueError('Livetime of the analysis is not defined yet. Please do this first!')
        self.backgroundPDF = pdf.flatten()*self.livetime
        print('total background events:', np.sum(self.backgroundPDF))
        self.nbins = len(pdf)
        '''
        
    def loadSignalPDF(self,pdf):
        self.loadPDF(pdf,'sig')
        '''
        if self.livetime < 0:
            raise ValueError('Livetime of the analysis is not defined yet. Please do this first!')
        self.signalPDF = pdf.flatten()*self.livetime
        print('total signal events:', np.sum(self.signalPDF))
        if self.nbins == len(pdf):
            self.ready = True
        else:
            raise ValueError('Shape of signal pdf does not match the background pdf!'+\
                             'Did you initialize the background pdf first?')
        '''
           
    def loadUncertaintyPDFs(self, sig_pdf, bkg_pdf, **morePDFs):
        self.PDF_uncert2['bkg'] = bkg_pdf.flatten()*self.livetime*self.livetime        
        self.PDF_uncert2['sig'] = sig_pdf.flatten()*self.livetime*self.livetime
        
        if norm:
            self.PDF_uncert2['bkg'] /= np.sum(self.PDFs_uncert2['bkg'])**2
            self.PDF_uncert2['sig'] /= np.sum(self.PDFs_uncert2['sig'])**2
        #if self.nbins != len(bkg_pdf):
        #    raise ValueError('Shape of background uncertainty pdf does not match the background pdf!')
        #if self.nbins != len(sig_pdf):
        #    raise ValueError('Shape of background uncertainty pdf does not match the background pdf!')
            
        for name in morePDFs.keys():
            self.PDFs_uncert2[name] = morePDFs[name].flatten()*self.livetime*self.livetime
            if norm:
                self.PDFs_uncert2[name] /= np.sum(self.PDF_uncert2[name])**2
            #if self.nbins != len(morePDFs[name]):
            #    raise ValueError('Shape of {0} uncertainty pdf does not match the background pdf!'.format(name))
 

    def sampleObservation(self, pars):
        '''
        Usage:
        sampleObservation(pars)
        pars : tuple of values. The first one is the signal fraction, the other ones must be the relative abundances of
            the different background components
        '''

        if not self.ready:
            raise ValueError('Not all pdfs are correctly loaded!')
            
        if self.Nevents == 0.:
            raise ValueError('Nevents not set!')

        if not 'sig' in self.PDFs.keys():
            raise ValueError("Signal PDF not loaded")
        if len(self.PDFs.keys()) < 2:
            raise ValueError(f"There must be at least 2 PDFs loaded. Currently {len(PDFs.keys())}")
        if len(pars) != len(self.PDFs.keys()):
            raise ValueError('Number of normalisation factors is not the same as number of PDFs loaded!')
        
        xi = pars[0]
        observationPDF = xi*self.PDFs['sig']
        
        bkg_norm = np.sum(pars[1:])
        for i, name in enumerate(self.PDFs.keys()):
            if name == 'sig': continue
            observationPDF += pars[i]*self.PDFs[name]/bkg_norm*(1-xi)
            
        observationPDF *= self.Nevents
        self.observation=np.zeros(np.shape(self.PDFs['sig']))
           
        for i in range(len(self.observation)):
            self.observation[i]=np.random.poisson(observationPDF[i])
                
        self.computedBestFit = False

        
    def evaluateLLH(self, pars):
        #print("pars=",pars)
        xi = pars[0]
        modelPDF = xi*self.PDFs['sig']
        
        bkg_norm = np.sum(pars[1:])
        for i, name in enumerate(self.PDFs.keys()):
            if name == 'sig': continue
            modelPDF += pars[i]*self.PDFs[name]/bkg_norm*(1-xi)
            
        modelPDF *= self.Nevents

        if np.isnan(modelPDF).any():
            print('nan in model array with sig fraction, background components=',xi,pars[1:],self.computedBestFit)

        if self.LLHtype == 'Poisson':
            bins_to_use = (modelPDF>0.)
            values = self.observation[bins_to_use]*np.log(modelPDF[bins_to_use])-modelPDF[bins_to_use]

        elif self.LLHtype == 'Effective':
            modelPDF_uncert2 = xi*self.PDFs_uncert2['sig']
            
            for i, name in enumerate(self.PDFs.keys()):
                if name == 'sig': continue
                modelPDF_uncert2 += pars[i]*self.PDFs_uncert2[name]/bkg_norm*(1-xi)
            
            bins_to_use = (modelPDF>0.)&(modelPDF_uncert2>0.)

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
        
        values = [0.]
        fix    = [False]
        error  = [.01]
        limit  = [(0.,1.)]
        name   = ['xi']
        
        for i, pdfname in enumerate(self.PDFs.keys()):
            if pdfname == 'sig': continue
            values.append(.5)
            fix.append(False)
            error.append(.01)
            limit.append((0., 1.))
            name.append(f'n_{pdfname}')
       
        #values = np.array(values)
        #fix    = np.array(fix)
        #error  = np.array(error)
        #limit  = np.array(limit)
        #name   = tuple(name)
        #print(values)
       
        kwds = dict()
        kwds['errordef']=.5
        kwds['print_level']=0
        
        LLHmin_DM=Minuit.from_array_func(self.evaluateLLH, values, fix=fix,
                                         error=error, limit=limit,
                                         name=name,
                                         **kwds)
        LLHmin_DM.migrad()
        
        self.bestFit = {}
        self.bestFit['params']=[LLHmin_DM.fitarg['xi']]
        for i, pdfname in enumerate(self.PDFs.keys()):
            if pdfname == 'sig': continue
            self.bestFit['params'].append(LLHmin_DM.fitarg[f'n_{pdfname}'])
        #self.bestFit['xi']=LLHmin_DM.fitarg['x1']
        #self.bestFit['n1']=LLHmin_DM.fitarg['n1']
        #if len(self.morePDFs.keys()) > 0:
        #    self.bestFit['more_n']=[]
        #    for key in more_n.keys():
        #        self.bestFit['more_n'].append(LLHmin_DM.fitarg[key])
        #    self.bestFit['LLH']=self.evaluateLLH(self.bestFit['nsig'],self.bestFit['n1'],*self.bestFit['more_n'])
        #else:
        self.bestFit['LLH']=self.evaluateLLH(self.bestFit['params'])
        self.bestFit['param_names']=list(self.PDFs.keys())
                
        self.computedBestFit = True
        
        
    def ComputeTestStatistics(self):
        
        if not self.computedBestFit:
            self.ComputeBestFit()
                
        values = [0.]
        fix    = [True]
        error  = [.01]
        limit  = [(0.,1.)]
        name   = ['xi']
        
        for i, pdfname in enumerate(self.PDFs.keys()):
            if pdfname == 'sig': continue
            values.append(.5)
            fix.append(False)
            error.append(.01)
            limit.append((0.,1.))
            name.append(f'n_{pdfname}')
       
        #values = tuple(values)
        #fix = tuple(fix)
        #error = tuple(error)
        #limit = tuple(limit)
        #name = tuple(name)
        
        kwds = dict()
        kwds['errordef']=.5
        kwds['print_level']=0

        LLHmin_ref=Minuit.from_array_func(self.evaluateLLH, values, fix=fix,
                                          error=error, limit=limit,
                                          name=name,
                                          **kwds)
        LLHmin_ref.migrad()
            
        self.TS = 0.
        ref_params = [LLHmin_ref.fitarg['xi']]
        for i, pdfname in enumerate(self.PDFs.keys()):
            if pdfname == 'sig': continue
            ref_params.append(LLHmin_ref.fitarg[f'n_{pdfname}'])
        
        self.bestFit['LLH_ref'] =  self.evaluateLLH(ref_params)                     
        #if len(more_n.keys()) > 0:
        #    ref_more_n = []
        #    for key in more_n.keys():
        #        ref_more_n.append(LLHmin_ref.fitarg[key])
        #        
        #    self.bestFit['LLH_ref'] = self.evaluateLLH(LLHmin_ref.fitarg['nsig'], LLHmin_ref.fitarg['n1'], *ref_more_n)
        #else:
        #    self.bestFit['LLH_ref'] = self.evaluateLLH(LLHmin_ref.fitarg['nsig'], LLHmin_ref.fitarg['n1'])
            
        if self.bestFit['params'][0] > 0.:
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
            
        param_low=self.bestFit['params'][0]
        param_up=self.bestFit['params'][0]
        param_mean=self.bestFit['params'][0]
        
        dTS=0
        cc=1
        while((dTS<deltaTS) and (nIterations<100)):
            nIterations += 1 

            param_up=param_up+3.*np.abs(param_up)
            
            values = [param_up]
            fix    = [True]
            error  = [.01]
            limit  = [(0.,1.)]
            name   = ['xi']
        
            for i, pdfname in enumerate(self.PDFs.keys()):
                if pdfname == 'sig': continue
                values.append(.5)
                fix.append(False)
                error.append(.01)
                limit.append((0., 1.))
                name.append(f'n_{pdfname}')
       
            #values = tuple(values)
            #fix = tuple(fix)
            #error = tuple(error)
            #limit = tuple(limit)
            #name = tuple(name)
        
            kwds = dict()
            kwds['errordef']=.5
            kwds['print_level']=0
            
            LLHmin_fix=Minuit.from_array_func(self.evaluateLLH, values, fix=fix,
                                              error=error, limit=limit,
                                              name=name,
                                              **kwds)
            LLHmin_fix.migrad()
            
            up_params = [param_up]
            for i, pdfname in enumerate(self.PDFs.keys()):
                if pdfname == 'sig': continue
                up_params.append(LLHmin_fix.fitarg[f'n_{pdfname}'])

            if param_up <0.:
                TS_fix = 0.
            else:
                TS_fix = 2*(self.bestFit['LLH_ref']-self.evaluateLLH(up_params))

            dTS = self.TS - TS_fix
            
            
        nIterations = 0
        param_low=param_up/4.
        while((cc>0.)  and (nIterations<100)):
            
            nIterations += 1

            param_mean=(param_low+param_up)/2.
            
            values = [param_mean]
            fix    = [True]
            error  = [.01]
            limit  = [(0.,1.)]
            name   = ['xi']
        
            for i, pdfname in enumerate(self.PDFs.keys()):
                if pdfname == 'sig': continue
                values.append(.5)
                fix.append(False)
                error.append(0.01)
                limit.append((0., 1.))
                name.append(f'n_{pdfname}')
       
            #values = tuple(values)
            #fix = tuple(fix)
            #error = tuple(error)
            #limit = tuple(limit)
            #name = tuple(name)
        
            kwds = dict()
            kwds['errordef']=.5
            kwds['print_level']=0
            
            LLHmin_fix=Minuit.from_array_func(self.evaluateLLH, values, fix=fix,
                                              error=error, limit=limit,
                                              name=name,
                                              **kwds)
            LLHmin_fix.migrad()

            mean_params = [param_mean]
            for i, pdfname in enumerate(self.PDFs.keys()):
                if pdfname == 'sig': continue
                mean_params.append(LLHmin_fix.fitarg[f'n_{pdfname}'])
            
            if param_mean <0.:
                TS_fix = 0.
            else:
                TS_fix = 2*(self.bestFit['LLH_ref']-self.evaluateLLH(mean_params))
                
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

        TS = []
        upperlimits = []
        if self.moreOutput:
            fits = []
        
        params = [0.]
        for i in range(len(self.PDFs.keys())-1):
            params.append(1.)
        
        for i in range(nTrials):
            self.sampleObservation(params)
            self.ComputeTestStatistics()
            TS.append(self.TS)
            #temporary fix for NaNs
            ul = self.CalculateUpperLimit(conf_level)
            if np.isnan(ul):
                print("Warning: NaN upper limit at trial {i}.\nRepeating trial.".format(i=i))
                i-=1
                continue
            upperlimits.append(ul)
              
            
            if self.moreOutput:
                fits.append(self.bestFit)
            
        p_median = np.percentile(upperlimits, 50)
        p_95_low = np.percentile(upperlimits, 2.5)
        p_95_high = np.percentile(upperlimits, 97.5)
        p_68_low = np.percentile(upperlimits, 16.)
        p_68_high = np.percentile(upperlimits, 84.)

        dic_brazilian = {}
        dic_brazilian['TS_dist'] = TS
        dic_brazilian['error_68_low'] = p_68_low
        dic_brazilian['error_68_high'] = p_68_high
        dic_brazilian['error_95_low'] = p_95_low
        dic_brazilian['error_95_high'] = p_95_high   
        dic_brazilian['median'] = p_median
        if self.moreOutput:
            dic_brazilian['upperlimits'] = upperlimits
            dic_brazilian['bestFits'] = fits

        return dic_brazilian
