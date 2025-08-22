import ROOT
import os 

# Question: A lot of the samples have _PSweights_.  The samples we were working with before didn't have this.  Is it good or bad?
# Sample Dictionaries: samples, samples_2016APVUL, samples_2016UL, samples_2017UL, samples_2018UL, samples_test, samples_QCD

targetlumi = {'2016APV':19500, '2016':16800, '2017':41480, '2018':59830}
lumiStr = '138fbfb' #str(targetlumi/1000).replace('.','p') # 1/fb
systListShort = ['elRecoSF', 'elIdSF', 'elIsoSF', 'TrigEffEl', 'muRecoSF', 'muIdSF', 'muIsoSF', 'TrigEffMu', 'Pileup', 'Prefire', 'jsf', 'toppt', 'muRFcorrd', 'btagHFCO', 'btagHFUC', 'btagLFCO', 'btagLFUC', 'jer', 'jec', 'PuJetSF']
systListFull = ['elRecoSF', 'elIdSF', 'elIsoSF', 'TrigEffEl', 'muRecoSF', 'muIdSF', 'muIsoSF', 'TrigEffMu', 'Pileup', 'Prefire', 'jsf', 'toppt', 'muRFcorrd', 'muR', 'muF', 'btagHFCO', 'btagHFUC', 'btagLFCO', 'btagLFUC', 'jer', 'jec', 'pNetTtag', 'pNetWtag', 'PuJetSF']
uncorrList_sf = ['TrigEffEl', 'TrigEffMu', 'jer', 'jec', 'btagHFUC', 'btagLFUC']
yearList = ["2016APV", "2016", "2017", "2018"]        
systListShortPlots = systListShort.copy()
systListFullPlots = systListFull.copy()
for syst in uncorrList_sf:
    systListShortPlots.remove(syst)
    systListFullPlots.remove(syst)
    for year in yearList:
        systListShortPlots.append(syst+year)
        systListFullPlots.append(syst+year)


systListABCDnn = ['val','train','pNetTtag','pNetWtag']#'param0','param1','param2','param3','param4','param5','param6','param7','lastbin']

# out of date
factorABCDnn = {'allWlep':'0.034045477',
                'allTlep':'0.08270128', 
                'tagTjet':'0.024456696', # '0.027892638',
                'tagWjet':'0.025098598', # '0.028440602',
                'untagWlep':'0.097960876', # '0.047838701',
                'untagTlep':'0.035287068', # '0.135501722'
}

yieldUncertABCDnn = {'tagTjet':0.02, #'0.078577386',
                     'tagWjet':0.02, #'0.061932256',
                     'untagTlep':0.10, #'0.018414285'
                     'untagWlep':0.08, #'0.080574995',
}

# all with direct alpha-ratio. sqrt(sys^2+stat^2+closure^2)

class sample:
    def __init__(self, prefix, xsec, year, textlist, samplename): #, color
        self.prefix = prefix
        self.year = year
        self.textlist = textlist
        self.samplename = samplename
        self.nrun = 1 # dummy
        self.kfactor = 1 # dummy
        self.xsec = xsec # in pb
        self.color = ROOT.kBlack

# We will use these for plotting in this "xsec" folder, but keep them as 1 in the actual sample definitions for ROOT files
# From Xanda, for "singlet" B prod with a b quark, for 1% width, for 50% tW
xsec = {'800':0.1187124, '900':0.0640113, '1000':0.0362987, '1100':0.0215009, '1200':0.0131348, '1300':0.0082629, '1400':0.0053213, '1500':0.0035078, '1600':0.0022829, '1700':0.0014947, '1800':0.0009898, '1900':0.0006519, '2000':0.0004499}
# extrapolating from ln(xsec) fit I get 0.000238 for 2200

Bprime_M1000_2016APV = sample("Bprime_M1000_2016APV", 1.0, "2016APV", "Bprime_M1000_2016APVULNanoList.txt", "/BprimeBtoTW_M-1000_NWALO_TuneCP5_13TeV-madgraph-pythia8/RunIISummer20UL16NanoAODAPVv9-106X_mcRun2_asymptotic_preVFP_v11-v2/NANOAODSIM")
Bprime_M1200_2016APV = sample("Bprime_M1200_2016APV", 1.0, "2016APV", "Bprime_M1200_2016APVULNanoList.txt", "/BprimeBtoTW_M-1200_NWALO_TuneCP5_13TeV-madgraph-pythia8/RunIISummer20UL16NanoAODAPVv9-106X_mcRun2_asymptotic_preVFP_v11-v2/NANOAODSIM")
Bprime_M1300_2016APV = sample("Bprime_M1300_2016APV", 1.0, "2016APV", "Bprime_M1300_2016APVULNanoList.txt", "/BprimeBtoTW_M-1300_NWALO_TuneCP5_13TeV-madgraph-pythia8/RunIISummer20UL16NanoAODAPVv9-106X_mcRun2_asymptotic_preVFP_v11-v2/NANOAODSIM")
Bprime_M1400_2016APV = sample("Bprime_M1400_2016APV", 1.0, "2016APV", "Bprime_M1400_2016APVULNanoList.txt", "/BprimeBtoTW_M-1400_NWALO_TuneCP5_13TeV-madgraph-pythia8/RunIISummer20UL16NanoAODAPVv9-106X_mcRun2_asymptotic_preVFP_v11-v2/NANOAODSIM")
Bprime_M1500_2016APV = sample("Bprime_M1500_2016APV", 1.0, "2016APV", "Bprime_M1500_2016APVULNanoList.txt", "/BprimeBtoTW_M-1500_NWALO_TuneCP5_13TeV-madgraph-pythia8/RunIISummer20UL16NanoAODAPVv9-106X_mcRun2_asymptotic_preVFP_v11-v2/NANOAODSIM")
Bprime_M1600_2016APV = sample("Bprime_M1600_2016APV", 1.0, "2016APV", "Bprime_M1600_2016APVULNanoList.txt", "/BprimeBtoTW_M-1600_NWALO_TuneCP5_13TeV-madgraph-pythia8/RunIISummer20UL16NanoAODAPVv9-106X_mcRun2_asymptotic_preVFP_v11-v2/NANOAODSIM")
Bprime_M1700_2016APV = sample("Bprime_M1700_2016APV", 1.0, "2016APV", "Bprime_M1700_2016APVULNanoList.txt", "/BprimeBtoTW_M-1700_NWALO_TuneCP5_13TeV-madgraph-pythia8/RunIISummer20UL16NanoAODAPVv9-106X_mcRun2_asymptotic_preVFP_v11-v2/NANOAODSIM")
Bprime_M1800_2016APV = sample("Bprime_M1800_2016APV", 1.0, "2016APV", "Bprime_M1800_2016APVULNanoList.txt", "/BprimeBtoTW_M-1800_NWALO_TuneCP5_13TeV-madgraph-pythia8/RunIISummer20UL16NanoAODAPVv9-106X_mcRun2_asymptotic_preVFP_v11-v2/NANOAODSIM")
Bprime_M2000_2016APV = sample("Bprime_M2000_2016APV", 1.0, "2016APV", "Bprime_M2000_2016APVULNanoList.txt", "/BprimeBtoTW_M-2000_NWALO_TuneCP5_13TeV-madgraph-pythia8/RunIISummer20UL16NanoAODAPVv9-106X_mcRun2_asymptotic_preVFP_v11-v2/NANOAODSIM")
Bprime_M2200_2016APV = sample("Bprime_M2200_2016APV", 1.0, "2016APV", "Bprime_M2200_2016APVULNanoList.txt", "/BprimeBtoTW_M-2200_NWALO_TuneCP5_13TeV-madgraph-pythia8/RunIISummer20UL16NanoAODAPVv9-106X_mcRun2_asymptotic_preVFP_v11-v2/NANOAODSIM")
Bprime_M800_2016APV = sample("Bprime_M800_2016APV",   1.0, "2016APV", "Bprime_M800_2016APVULNanoList.txt", "/BprimeBtoTW_M-800_NWALO_TuneCP5_13TeV-madgraph-pythia8/RunIISummer20UL16NanoAODAPVv9-106X_mcRun2_asymptotic_preVFP_v11-v2/NANOAODSIM")
Bprime_M1000_2016 = sample("Bprime_M1000_2016", 1.0, "2016", "Bprime_M1000_2016ULNanoList.txt", "/BprimeBtoTW_M-1000_NWALO_TuneCP5_13TeV-madgraph-pythia8/RunIISummer20UL16NanoAODv9-106X_mcRun2_asymptotic_v17-v2/NANOAODSIM")
Bprime_M1000_2017 = sample("Bprime_M1000_2017", 1.0, "2017", "Bprime_M1000_2017ULNanoList.txt", "/BprimeBtoTW_M-1000_NWALO_TuneCP5_13TeV-madgraph-pythia8/RunIISummer20UL17NanoAODv9-106X_mc2017_realistic_v9-v2/NANOAODSIM")
Bprime_M1000_2018 = sample("Bprime_M1000_2018", 1.0, "2018", "Bprime_M1000_2018ULNanoList.txt", "/BprimeBtoTW_M-1000_NWALO_TuneCP5_13TeV-madgraph-pythia8/RunIISummer20UL18NanoAODv9-106X_upgrade2018_realistic_v16_L1v1-v2/NANOAODSIM")
Bprime_M1200_2016 = sample("Bprime_M1200_2016", 1.0, "2016", "Bprime_M1200_2016ULNanoList.txt", "/BprimeBtoTW_M-1200_NWALO_TuneCP5_13TeV-madgraph-pythia8/RunIISummer20UL16NanoAODv9-106X_mcRun2_asymptotic_v17-v2/NANOAODSIM")
Bprime_M1200_2017 = sample("Bprime_M1200_2017", 1.0, "2017", "Bprime_M1200_2017ULNanoList.txt", "/BprimeBtoTW_M-1200_NWALO_TuneCP5_13TeV-madgraph-pythia8/RunIISummer20UL17NanoAODv9-106X_mc2017_realistic_v9-v2/NANOAODSIM")
Bprime_M1200_2018 = sample("Bprime_M1200_2018", 1.0, "2018", "Bprime_M1200_2018ULNanoList.txt", "/BprimeBtoTW_M-1200_NWALO_TuneCP5_13TeV-madgraph-pythia8/RunIISummer20UL18NanoAODv9-106X_upgrade2018_realistic_v16_L1v1-v2/NANOAODSIM")
Bprime_M1300_2016 = sample("Bprime_M1300_2016", 1.0, "2016", "Bprime_M1300_2016ULNanoList.txt", "/BprimeBtoTW_M-1300_NWALO_TuneCP5_13TeV-madgraph-pythia8/RunIISummer20UL16NanoAODv9-106X_mcRun2_asymptotic_v17-v2/NANOAODSIM")
Bprime_M1300_2017 = sample("Bprime_M1300_2017", 1.0, "2017", "Bprime_M1300_2017ULNanoList.txt", "/BprimeBtoTW_M-1300_NWALO_TuneCP5_13TeV-madgraph-pythia8/RunIISummer20UL17NanoAODv9-106X_mc2017_realistic_v9-v2/NANOAODSIM")
Bprime_M1300_2018 = sample("Bprime_M1300_2018", 1.0, "2018", "Bprime_M1300_2018ULNanoList.txt", "/BprimeBtoTW_M-1300_NWALO_TuneCP5_13TeV-madgraph-pythia8/RunIISummer20UL18NanoAODv9-106X_upgrade2018_realistic_v16_L1v1-v2/NANOAODSIM")
Bprime_M1400_2016 = sample("Bprime_M1400_2016", 1.0, "2016", "Bprime_M1400_2016ULNanoList.txt", "/BprimeBtoTW_M-1400_NWALO_TuneCP5_13TeV-madgraph-pythia8/RunIISummer20UL16NanoAODv9-106X_mcRun2_asymptotic_v17-v2/NANOAODSIM")
Bprime_M1400_2017 = sample("Bprime_M1400_2017", 1.0, "2017", "Bprime_M1400_2017ULNanoList.txt", "/BprimeBtoTW_M-1400_NWALO_TuneCP5_13TeV-madgraph-pythia8/RunIISummer20UL17NanoAODv9-106X_mc2017_realistic_v9-v2/NANOAODSIM")
Bprime_M1400_2018 = sample("Bprime_M1400_2018", 1.0, "2018", "Bprime_M1400_2018ULNanoList.txt", "/BprimeBtoTW_M-1400_NWALO_TuneCP5_13TeV-madgraph-pythia8/RunIISummer20UL18NanoAODv9-106X_upgrade2018_realistic_v16_L1v1-v2/NANOAODSIM")
Bprime_M1500_2016 = sample("Bprime_M1500_2016", 1.0, "2016", "Bprime_M1500_2016ULNanoList.txt", "/BprimeBtoTW_M-1500_NWALO_TuneCP5_13TeV-madgraph-pythia8/RunIISummer20UL16NanoAODv9-106X_mcRun2_asymptotic_v17-v2/NANOAODSIM")
Bprime_M1500_2017 = sample("Bprime_M1500_2017", 1.0, "2017", "Bprime_M1500_2017ULNanoList.txt", "/BprimeBtoTW_M-1500_NWALO_TuneCP5_13TeV-madgraph-pythia8/RunIISummer20UL17NanoAODv9-106X_mc2017_realistic_v9-v2/NANOAODSIM")
Bprime_M1500_2018 = sample("Bprime_M1500_2018", 1.0, "2018", "Bprime_M1500_2018ULNanoList.txt", "/BprimeBtoTW_M-1500_NWALO_TuneCP5_13TeV-madgraph-pythia8/RunIISummer20UL18NanoAODv9-106X_upgrade2018_realistic_v16_L1v1-v2/NANOAODSIM")
Bprime_M1600_2016 = sample("Bprime_M1600_2016", 1.0, "2016", "Bprime_M1600_2016ULNanoList.txt", "/BprimeBtoTW_M-1600_NWALO_TuneCP5_13TeV-madgraph-pythia8/RunIISummer20UL16NanoAODv9-106X_mcRun2_asymptotic_v17-v2/NANOAODSIM")
Bprime_M1600_2017 = sample("Bprime_M1600_2017", 1.0, "2017", "Bprime_M1600_2017ULNanoList.txt", "/BprimeBtoTW_M-1600_NWALO_TuneCP5_13TeV-madgraph-pythia8/RunIISummer20UL17NanoAODv9-106X_mc2017_realistic_v9-v2/NANOAODSIM")
Bprime_M1600_2018 = sample("Bprime_M1600_2018", 1.0, "2018", "Bprime_M1600_2018ULNanoList.txt", "/BprimeBtoTW_M-1600_NWALO_TuneCP5_13TeV-madgraph-pythia8/RunIISummer20UL18NanoAODv9-106X_upgrade2018_realistic_v16_L1v1-v2/NANOAODSIM")
Bprime_M1700_2016 = sample("Bprime_M1700_2016", 1.0, "2016", "Bprime_M1700_2016ULNanoList.txt", "/BprimeBtoTW_M-1700_NWALO_TuneCP5_13TeV-madgraph-pythia8/RunIISummer20UL16NanoAODv9-106X_mcRun2_asymptotic_v17-v2/NANOAODSIM")
Bprime_M1700_2017 = sample("Bprime_M1700_2017", 1.0, "2017", "Bprime_M1700_2017ULNanoList.txt", "/BprimeBtoTW_M-1700_NWALO_TuneCP5_13TeV-madgraph-pythia8/RunIISummer20UL17NanoAODv9-106X_mc2017_realistic_v9-v2/NANOAODSIM")
Bprime_M1700_2018 = sample("Bprime_M1700_2018", 1.0, "2018", "Bprime_M1700_2018ULNanoList.txt", "/BprimeBtoTW_M-1700_NWALO_TuneCP5_13TeV-madgraph-pythia8/RunIISummer20UL18NanoAODv9-106X_upgrade2018_realistic_v16_L1v1-v2/NANOAODSIM")
Bprime_M1800_2016 = sample("Bprime_M1800_2016", 1.0, "2016", "Bprime_M1800_2016ULNanoList.txt", "/BprimeBtoTW_M-1800_NWALO_TuneCP5_13TeV-madgraph-pythia8/RunIISummer20UL16NanoAODv9-106X_mcRun2_asymptotic_v17-v2/NANOAODSIM")
Bprime_M1800_2017 = sample("Bprime_M1800_2017", 1.0, "2017", "Bprime_M1800_2017ULNanoList.txt", "/BprimeBtoTW_M-1800_NWALO_TuneCP5_13TeV-madgraph-pythia8/RunIISummer20UL17NanoAODv9-106X_mc2017_realistic_v9-v2/NANOAODSIM")
Bprime_M1800_2018 = sample("Bprime_M1800_2018", 1.0, "2018", "Bprime_M1800_2018ULNanoList.txt", "/BprimeBtoTW_M-1800_NWALO_TuneCP5_13TeV-madgraph-pythia8/RunIISummer20UL18NanoAODv9-106X_upgrade2018_realistic_v16_L1v1-v2/NANOAODSIM")
Bprime_M2000_2016 = sample("Bprime_M2000_2016", 1.0, "2016", "Bprime_M2000_2016ULNanoList.txt", "/BprimeBtoTW_M-2000_NWALO_TuneCP5_13TeV-madgraph-pythia8/RunIISummer20UL16NanoAODv9-106X_mcRun2_asymptotic_v17-v2/NANOAODSIM")
Bprime_M2000_2017 = sample("Bprime_M2000_2017", 1.0, "2017", "Bprime_M2000_2017ULNanoList.txt", "/BprimeBtoTW_M-2000_NWALO_TuneCP5_13TeV-madgraph-pythia8/RunIISummer20UL17NanoAODv9-106X_mc2017_realistic_v9-v2/NANOAODSIM")
Bprime_M2000_2018 = sample("Bprime_M2000_2018", 1.0, "2018", "Bprime_M2000_2018ULNanoList.txt", "/BprimeBtoTW_M-2000_NWALO_TuneCP5_13TeV-madgraph-pythia8/RunIISummer20UL18NanoAODv9-106X_upgrade2018_realistic_v16_L1v1-v2/NANOAODSIM")
Bprime_M2200_2016 = sample("Bprime_M2200_2016", 1.0, "2016", "Bprime_M2200_2016ULNanoList.txt", "/BprimeBtoTW_M-2200_NWALO_TuneCP5_13TeV-madgraph-pythia8/RunIISummer20UL16NanoAODv9-106X_mcRun2_asymptotic_v17-v2/NANOAODSIM")
Bprime_M2200_2017 = sample("Bprime_M2200_2017", 1.0, "2017", "Bprime_M2200_2017ULNanoList.txt", "/BprimeBtoTW_M-2200_NWALO_TuneCP5_13TeV-madgraph-pythia8/RunIISummer20UL17NanoAODv9-106X_mc2017_realistic_v9-v2/NANOAODSIM")
Bprime_M2200_2018 = sample("Bprime_M2200_2018", 1.0, "2018", "Bprime_M2200_2018ULNanoList.txt", "/BprimeBtoTW_M-2200_NWALO_TuneCP5_13TeV-madgraph-pythia8/RunIISummer20UL18NanoAODv9-106X_upgrade2018_realistic_v16_L1v1-v2/NANOAODSIM")
Bprime_M800_2016  = sample("Bprime_M800_2016", 1.0, "2016", "Bprime_M800_2016ULNanoList.txt", "/BprimeBtoTW_M-800_NWALO_TuneCP5_13TeV-madgraph-pythia8/RunIISummer20UL16NanoAODv9-106X_mcRun2_asymptotic_v17-v2/NANOAODSIM")
Bprime_M800_2017  = sample("Bprime_M800_2017", 1.0, "2017", "Bprime_M800_2017ULNanoList.txt", "/BprimeBtoTW_M-800_NWALO_TuneCP5_13TeV-madgraph-pythia8/RunIISummer20UL17NanoAODv9-106X_mc2017_realistic_v9-v2/NANOAODSIM")
Bprime_M800_2018  = sample("Bprime_M800_2018", 1.0, "2018", "Bprime_M800_2018ULNanoList.txt", "/BprimeBtoTW_M-800_NWALO_TuneCP5_13TeV-madgraph-pythia8/RunIISummer20UL18NanoAODv9-106X_upgrade2018_realistic_v16_L1v1-v2/NANOAODSIM")
DYMHT12002016APV  = sample("DYMHT12002016APV", 0.1514*1.23, "2016APV", "DYMHT12002016APVULNanoList.txt", "/DYJetsToLL_M-50_HT-1200to2500_TuneCP5_PSweights_13TeV-madgraphMLM-pythia8/RunIISummer20UL16NanoAODAPVv9-106X_mcRun2_asymptotic_preVFP_v11-v2/NANOAODSIM")
DYMHT12002016     = sample("DYMHT12002016", 0.1514*1.23, "2016", "DYMHT12002016ULNanoList.txt", "/DYJetsToLL_M-50_HT-1200to2500_TuneCP5_PSweights_13TeV-madgraphMLM-pythia8/RunIISummer20UL16NanoAODv9-106X_mcRun2_asymptotic_v17-v2/NANOAODSIM")
DYMHT12002017     = sample("DYMHT12002017", 0.1514*1.23, "2017", "DYMHT12002017ULNanoList.txt", "/DYJetsToLL_M-50_HT-1200to2500_TuneCP5_PSweights_13TeV-madgraphMLM-pythia8/RunIISummer20UL17NanoAODv9-106X_mc2017_realistic_v9-v1/NANOAODSIM")
DYMHT12002018     = sample("DYMHT12002018", 0.1514*1.23, "2018", "DYMHT12002018ULNanoList.txt", "/DYJetsToLL_M-50_HT-1200to2500_TuneCP5_PSweights_13TeV-madgraphMLM-pythia8/RunIISummer20UL18NanoAODv9-106X_upgrade2018_realistic_v16_L1v1-v1/NANOAODSIM")
DYMHT2002016APV   = sample("DYMHT2002016APV", 40.99*1.23, "2016APV", "DYMHT2002016APVULNanoList.txt", "/DYJetsToLL_M-50_HT-200to400_TuneCP5_PSweights_13TeV-madgraphMLM-pythia8/RunIISummer20UL16NanoAODAPVv9-106X_mcRun2_asymptotic_preVFP_v11-v2/NANOAODSIM")
DYMHT2002016      = sample("DYMHT2002016", 40.99*1.23, "2016", "DYMHT2002016ULNanoList.txt", "/DYJetsToLL_M-50_HT-200to400_TuneCP5_PSweights_13TeV-madgraphMLM-pythia8/RunIISummer20UL16NanoAODv9-106X_mcRun2_asymptotic_v17-v2/NANOAODSIM")
DYMHT2002017      = sample("DYMHT2002017", 40.99*1.23, "2017", "DYMHT2002017ULNanoList.txt", "/DYJetsToLL_M-50_HT-200to400_TuneCP5_PSweights_13TeV-madgraphMLM-pythia8/RunIISummer20UL17NanoAODv9-106X_mc2017_realistic_v9-v1/NANOAODSIM")
DYMHT2002018      = sample("DYMHT2002018", 40.99*1.23, "2018", "DYMHT2002018ULNanoList.txt", "/DYJetsToLL_M-50_HT-200to400_TuneCP5_PSweights_13TeV-madgraphMLM-pythia8/RunIISummer20UL18NanoAODv9-106X_upgrade2018_realistic_v16_L1v1-v1/NANOAODSIM")
DYMHT25002016APV  = sample("DYMHT25002016APV", 0.003565*1.23, "2016APV", "DYMHT25002016APVULNanoList.txt", "/DYJetsToLL_M-50_HT-2500toInf_TuneCP5_PSweights_13TeV-madgraphMLM-pythia8/RunIISummer20UL16NanoAODAPVv9-106X_mcRun2_asymptotic_preVFP_v11-v2/NANOAODSIM")
DYMHT25002016     = sample("DYMHT25002016", 0.003565*1.23, "2016", "DYMHT25002016ULNanoList.txt", "/DYJetsToLL_M-50_HT-2500toInf_TuneCP5_PSweights_13TeV-madgraphMLM-pythia8/RunIISummer20UL16NanoAODv9-106X_mcRun2_asymptotic_v17-v2/NANOAODSIM")
DYMHT25002017     = sample("DYMHT25002017", 0.003565*1.23, "2017", "DYMHT25002017ULNanoList.txt", "/DYJetsToLL_M-50_HT-2500toInf_TuneCP5_PSweights_13TeV-madgraphMLM-pythia8/RunIISummer20UL17NanoAODv9-106X_mc2017_realistic_v9-v1/NANOAODSIM")
DYMHT25002018     = sample("DYMHT25002018", 0.003565*1.23, "2018", "DYMHT25002018ULNanoList.txt", "/DYJetsToLL_M-50_HT-2500toInf_TuneCP5_PSweights_13TeV-madgraphMLM-pythia8/RunIISummer20UL18NanoAODv9-106X_upgrade2018_realistic_v16_L1v1-v1/NANOAODSIM")
DYMHT4002016APV   = sample("DYMHT4002016APV", 5.678*1.23, "2016APV", "DYMHT4002016APVULNanoList.txt", "/DYJetsToLL_M-50_HT-400to600_TuneCP5_PSweights_13TeV-madgraphMLM-pythia8/RunIISummer20UL16NanoAODAPVv9-106X_mcRun2_asymptotic_preVFP_v11-v2/NANOAODSIM")
DYMHT4002016      = sample("DYMHT4002016", 5.678*1.23, "2016", "DYMHT4002016ULNanoList.txt", "/DYJetsToLL_M-50_HT-400to600_TuneCP5_PSweights_13TeV-madgraphMLM-pythia8/RunIISummer20UL16NanoAODv9-106X_mcRun2_asymptotic_v17-v2/NANOAODSIM")
DYMHT4002017      = sample("DYMHT4002017", 5.678*1.23, "2017", "DYMHT4002017ULNanoList.txt", "/DYJetsToLL_M-50_HT-400to600_TuneCP5_PSweights_13TeV-madgraphMLM-pythia8/RunIISummer20UL17NanoAODv9-106X_mc2017_realistic_v9-v1/NANOAODSIM")
DYMHT4002018      = sample("DYMHT4002018", 5.678*1.23, "2018", "DYMHT4002018ULNanoList.txt", "/DYJetsToLL_M-50_HT-400to600_TuneCP5_PSweights_13TeV-madgraphMLM-pythia8/RunIISummer20UL18NanoAODv9-106X_upgrade2018_realistic_v16_L1v1-v1/NANOAODSIM")
DYMHT6002016APV   = sample("DYMHT6002016APV", 1.367*1.23, "2016APV", "DYMHT6002016APVULNanoList.txt", "/DYJetsToLL_M-50_HT-600to800_TuneCP5_PSweights_13TeV-madgraphMLM-pythia8/RunIISummer20UL16NanoAODAPVv9-106X_mcRun2_asymptotic_preVFP_v11-v2/NANOAODSIM")
DYMHT6002016      = sample("DYMHT6002016", 1.367*1.23, "2016", "DYMHT6002016ULNanoList.txt", "/DYJetsToLL_M-50_HT-600to800_TuneCP5_PSweights_13TeV-madgraphMLM-pythia8/RunIISummer20UL16NanoAODv9-106X_mcRun2_asymptotic_v17-v2/NANOAODSIM")
DYMHT6002017      = sample("DYMHT6002017", 1.367*1.23, "2017", "DYMHT6002017ULNanoList.txt", "/DYJetsToLL_M-50_HT-600to800_TuneCP5_PSweights_13TeV-madgraphMLM-pythia8/RunIISummer20UL17NanoAODv9-106X_mc2017_realistic_v9-v1/NANOAODSIM")
DYMHT6002018      = sample("DYMHT6002018", 1.367*1.23, "2018", "DYMHT6002018ULNanoList.txt", "/DYJetsToLL_M-50_HT-600to800_TuneCP5_PSweights_13TeV-madgraphMLM-pythia8/RunIISummer20UL18NanoAODv9-106X_upgrade2018_realistic_v16_L1v1-v1/NANOAODSIM")
DYMHT8002016APV   = sample("DYMHT8002016APV", 0.6304*1.23, "2016APV", "DYMHT8002016APVULNanoList.txt", "/DYJetsToLL_M-50_HT-800to1200_TuneCP5_PSweights_13TeV-madgraphMLM-pythia8/RunIISummer20UL16NanoAODAPVv9-106X_mcRun2_asymptotic_preVFP_v11-v2/NANOAODSIM")
DYMHT8002016      = sample("DYMHT8002016", 0.6304*1.23, "2016", "DYMHT8002016ULNanoList.txt", "/DYJetsToLL_M-50_HT-800to1200_TuneCP5_PSweights_13TeV-madgraphMLM-pythia8/RunIISummer20UL16NanoAODv9-106X_mcRun2_asymptotic_v17-v2/NANOAODSIM")
DYMHT8002017      = sample("DYMHT8002017", 0.6304*1.23, "2017", "DYMHT8002017ULNanoList.txt", "/DYJetsToLL_M-50_HT-800to1200_TuneCP5_PSweights_13TeV-madgraphMLM-pythia8/RunIISummer20UL17NanoAODv9-106X_mc2017_realistic_v9-v1/NANOAODSIM")
DYMHT8002018      = sample("DYMHT8002018", 0.6304*1.23, "2018", "DYMHT8002018ULNanoList.txt", "/DYJetsToLL_M-50_HT-800to1200_TuneCP5_PSweights_13TeV-madgraphMLM-pythia8/RunIISummer20UL18NanoAODv9-106X_upgrade2018_realistic_v16_L1v1-v1/NANOAODSIM")
JetHTRun2016APVB    = sample("JetHTRun2016APVB", 1.0, "2016APV", "JetHTRun2016APVBNanoList.txt", "/JetHT/Run2016B-ver2_HIPM_UL2016_MiniAODv2_NanoAODv9-v2/NANOAOD")
JetHTRun2016APVC    = sample("JetHTRun2016APVC", 1.0, "2016APV", "JetHTRun2016APVCNanoList.txt", "/JetHT/Run2016C-HIPM_UL2016_MiniAODv2_NanoAODv9-v2/NANOAOD")
JetHTRun2016APVD    = sample("JetHTRun2016APVD", 1.0, "2016APV", "JetHTRun2016APVDNanoList.txt", "/JetHT/Run2016D-HIPM_UL2016_MiniAODv2_NanoAODv9-v2/NANOAOD")
JetHTRun2016APVE    = sample("JetHTRun2016APVE", 1.0, "2016APV", "JetHTRun2016APVENanoList.txt", "/JetHT/Run2016E-HIPM_UL2016_MiniAODv2_NanoAODv9-v2/NANOAOD")
JetHTRun2016APVF    = sample("JetHTRun2016APVF", 1.0, "2016APV", "JetHTRun2016APVFNanoList.txt", "/JetHT/Run2016F-HIPM_UL2016_MiniAODv2_NanoAODv9-v2/NANOAOD")
JetHTRun2016F       = sample("JetHTRun2016F", 1.0, "2016", "JetHTRun2016FNanoList.txt", "/JetHT/Run2016F-UL2016_MiniAODv2_NanoAODv9-v1/NANOAOD")
JetHTRun2016G       = sample("JetHTRun2016G", 1.0, "2016", "JetHTRun2016GNanoList.txt", "/JetHT/Run2016G-UL2016_MiniAODv2_NanoAODv9-v1/NANOAOD")
JetHTRun2016H       = sample("JetHTRun2016H", 1.0, "2016", "JetHTRun2016HNanoList.txt", "/JetHT/Run2016H-UL2016_MiniAODv2_NanoAODv9-v1/NANOAOD")
JetHTRun2017B       = sample("JetHTRun2017B", 1.0, "2017", "JetHTRun2017BNanoList.txt", "/JetHT/Run2017B-UL2017_MiniAODv2_NanoAODv9-v1/NANOAOD")
JetHTRun2017C       = sample("JetHTRun2017C", 1.0, "2017", "JetHTRun2017CNanoList.txt", "/JetHT/Run2017C-UL2017_MiniAODv2_NanoAODv9-v1/NANOAOD")
JetHTRun2017D       = sample("JetHTRun2017D", 1.0, "2017", "JetHTRun2017DNanoList.txt", "/JetHT/Run2017D-UL2017_MiniAODv2_NanoAODv9-v1/NANOAOD")
JetHTRun2017E       = sample("JetHTRun2017E", 1.0, "2017", "JetHTRun2017ENanoList.txt", "/JetHT/Run2017E-UL2017_MiniAODv2_NanoAODv9-v1/NANOAOD")
JetHTRun2017F       = sample("JetHTRun2017F", 1.0, "2017", "JetHTRun2017FNanoList.txt", "/JetHT/Run2017F-UL2017_MiniAODv2_NanoAODv9-v1/NANOAOD")
JetHTRun2018A       = sample("JetHTRun2018A", 1.0, "2018", "JetHTRun2018ANanoList.txt", "/JetHT/Run2018A-UL2018_MiniAODv2_NanoAODv9-v2/NANOAOD")
JetHTRun2018B       = sample("JetHTRun2018B", 1.0, "2018", "JetHTRun2018BNanoList.txt", "/JetHT/Run2018B-UL2018_MiniAODv2_NanoAODv9-v1/NANOAOD")
JetHTRun2018C       = sample("JetHTRun2018C", 1.0, "2018", "JetHTRun2018CNanoList.txt", "/JetHT/Run2018C-UL2018_MiniAODv2_NanoAODv9-v1/NANOAOD")
JetHTRun2018D       = sample("JetHTRun2018D", 1.0, "2018", "JetHTRun2018DNanoList.txt", "/JetHT/Run2018D-UL2018_MiniAODv2_NanoAODv9-v2/NANOAOD")
QCDHT10002016APV  = sample("QCDHT10002016APV", 1207., "2016APV", "QCDHT10002016APVULNanoList.txt", "/QCD_HT1000to1500_TuneCP5_PSWeights_13TeV-madgraph-pythia8/RunIISummer20UL16NanoAODAPVv9-106X_mcRun2_asymptotic_preVFP_v11-v1/NANOAODSIM")
QCDHT10002016     = sample("QCDHT10002016", 1207., "2016", "QCDHT10002016ULNanoList.txt", "/QCD_HT1000to1500_TuneCP5_PSWeights_13TeV-madgraph-pythia8/RunIISummer20UL16NanoAODv9-106X_mcRun2_asymptotic_v17-v1/NANOAODSIM")
QCDHT10002017     = sample("QCDHT10002017", 1207., "2017", "QCDHT10002017ULNanoList.txt", "/QCD_HT1000to1500_TuneCP5_PSWeights_13TeV-madgraph-pythia8/RunIISummer20UL17NanoAODv9-106X_mc2017_realistic_v9-v1/NANOAODSIM")
QCDHT10002018     = sample("QCDHT10002018", 1207., "2018", "QCDHT10002018ULNanoList.txt", "/QCD_HT1000to1500_TuneCP5_PSWeights_13TeV-madgraph-pythia8/RunIISummer20UL18NanoAODv9-106X_upgrade2018_realistic_v16_L1v1-v1/NANOAODSIM")
QCDHT15002016APV  = sample("QCDHT15002016APV", 119.9, "2016APV", "QCDHT15002016APVULNanoList.txt", "/QCD_HT1500to2000_TuneCP5_PSWeights_13TeV-madgraph-pythia8/RunIISummer20UL16NanoAODAPVv9-106X_mcRun2_asymptotic_preVFP_v11-v1/NANOAODSIM")
QCDHT15002016     = sample("QCDHT15002016", 119.9, "2016", "QCDHT15002016ULNanoList.txt", "/QCD_HT1500to2000_TuneCP5_PSWeights_13TeV-madgraph-pythia8/RunIISummer20UL16NanoAODv9-106X_mcRun2_asymptotic_v17-v1/NANOAODSIM")
QCDHT15002017     = sample("QCDHT15002017", 119.9, "2017", "QCDHT15002017ULNanoList.txt", "/QCD_HT1500to2000_TuneCP5_PSWeights_13TeV-madgraph-pythia8/RunIISummer20UL17NanoAODv9-106X_mc2017_realistic_v9-v1/NANOAODSIM")
QCDHT15002018     = sample("QCDHT15002018", 119.9, "2018", "QCDHT15002018ULNanoList.txt", "/QCD_HT1500to2000_TuneCP5_PSWeights_13TeV-madgraph-pythia8/RunIISummer20UL18NanoAODv9-106X_upgrade2018_realistic_v16_L1v1-v1/NANOAODSIM")
QCDHT20002016APV  = sample("QCDHT20002016APV", 25.24, "2016APV", "QCDHT20002016APVULNanoList.txt", "/QCD_HT2000toInf_TuneCP5_PSWeights_13TeV-madgraph-pythia8/RunIISummer20UL16NanoAODAPVv9-106X_mcRun2_asymptotic_preVFP_v11-v1/NANOAODSIM")
QCDHT20002016     = sample("QCDHT20002016", 25.24, "2016", "QCDHT20002016ULNanoList.txt", "/QCD_HT2000toInf_TuneCP5_PSWeights_13TeV-madgraph-pythia8/RunIISummer20UL16NanoAODv9-106X_mcRun2_asymptotic_v17-v1/NANOAODSIM")
QCDHT20002017     = sample("QCDHT20002017", 25.24, "2017", "QCDHT20002017ULNanoList.txt", "/QCD_HT2000toInf_TuneCP5_PSWeights_13TeV-madgraph-pythia8/RunIISummer20UL17NanoAODv9-106X_mc2017_realistic_v9-v1/NANOAODSIM")
QCDHT20002018     = sample("QCDHT20002018", 25.24, "2018", "QCDHT20002018ULNanoList.txt", "/QCD_HT2000toInf_TuneCP5_PSWeights_13TeV-madgraph-pythia8/RunIISummer20UL18NanoAODv9-106X_upgrade2018_realistic_v16_L1v1-v1/NANOAODSIM")
QCDHT2002016APV   = sample("QCDHT2002016APV", 1712000., "2016APV", "QCDHT2002016APVULNanoList.txt", "/QCD_HT200to300_TuneCP5_PSWeights_13TeV-madgraph-pythia8/RunIISummer20UL16NanoAODAPVv9-106X_mcRun2_asymptotic_preVFP_v11-v1/NANOAODSIM")
QCDHT2002016      = sample("QCDHT2002016", 1712000., "2016", "QCDHT2002016ULNanoList.txt", "/QCD_HT200to300_TuneCP5_PSWeights_13TeV-madgraph-pythia8/RunIISummer20UL16NanoAODv9-106X_mcRun2_asymptotic_v17-v1/NANOAODSIM")
QCDHT2002017      = sample("QCDHT2002017", 1712000., "2017", "QCDHT2002017ULNanoList.txt", "/QCD_HT200to300_TuneCP5_PSWeights_13TeV-madgraph-pythia8/RunIISummer20UL17NanoAODv9-106X_mc2017_realistic_v9-v1/NANOAODSIM")
QCDHT2002018      = sample("QCDHT2002018", 1712000., "2018", "QCDHT2002018ULNanoList.txt", "/QCD_HT200to300_TuneCP5_PSWeights_13TeV-madgraph-pythia8/RunIISummer20UL18NanoAODv9-106X_upgrade2018_realistic_v16_L1v1-v1/NANOAODSIM")
QCDHT3002016APV   = sample("QCDHT3002016APV", 347700., "2016APV", "QCDHT3002016APVULNanoList.txt", "/QCD_HT300to500_TuneCP5_PSWeights_13TeV-madgraph-pythia8/RunIISummer20UL16NanoAODAPVv9-106X_mcRun2_asymptotic_preVFP_v11-v1/NANOAODSIM")
QCDHT3002016      = sample("QCDHT3002016", 347700., "2016", "QCDHT3002016ULNanoList.txt", "/QCD_HT300to500_TuneCP5_PSWeights_13TeV-madgraph-pythia8/RunIISummer20UL16NanoAODv9-106X_mcRun2_asymptotic_v17-v1/NANOAODSIM")
QCDHT3002017      = sample("QCDHT3002017", 347700., "2017", "QCDHT3002017ULNanoList.txt", "/QCD_HT300to500_TuneCP5_PSWeights_13TeV-madgraph-pythia8/RunIISummer20UL17NanoAODv9-106X_mc2017_realistic_v9-v1/NANOAODSIM")
QCDHT3002018      = sample("QCDHT3002018", 347700., "2018", "QCDHT3002018ULNanoList.txt", "/QCD_HT300to500_TuneCP5_PSWeights_13TeV-madgraph-pythia8/RunIISummer20UL18NanoAODv9-106X_upgrade2018_realistic_v16_L1v1-v1/NANOAODSIM")
QCDHT5002016APV   = sample("QCDHT5002016APV", 32100., "2016APV", "QCDHT5002016APVULNanoList.txt", "/QCD_HT500to700_TuneCP5_PSWeights_13TeV-madgraph-pythia8/RunIISummer20UL16NanoAODAPVv9-106X_mcRun2_asymptotic_preVFP_v11-v1/NANOAODSIM")
QCDHT5002016      = sample("QCDHT5002016", 32100., "2016", "QCDHT5002016ULNanoList.txt", "/QCD_HT500to700_TuneCP5_PSWeights_13TeV-madgraph-pythia8/RunIISummer20UL16NanoAODv9-106X_mcRun2_asymptotic_v17-v1/NANOAODSIM")
QCDHT5002017      = sample("QCDHT5002017", 32100., "2017", "QCDHT5002017ULNanoList.txt", "/QCD_HT500to700_TuneCP5_PSWeights_13TeV-madgraph-pythia8/RunIISummer20UL17NanoAODv9-106X_mc2017_realistic_v9-v1/NANOAODSIM")
QCDHT5002018      = sample("QCDHT5002018", 32100., "2018", "QCDHT5002018ULNanoList.txt", "/QCD_HT500to700_TuneCP5_PSWeights_13TeV-madgraph-pythia8/RunIISummer20UL18NanoAODv9-106X_upgrade2018_realistic_v16_L1v1-v1/NANOAODSIM")
QCDHT7002016APV   = sample("QCDHT7002016APV", 6831., "2016APV", "QCDHT7002016APVULNanoList.txt", "/QCD_HT700to1000_TuneCP5_PSWeights_13TeV-madgraph-pythia8/RunIISummer20UL16NanoAODAPVv9-106X_mcRun2_asymptotic_preVFP_v11-v1/NANOAODSIM")
QCDHT7002016      = sample("QCDHT7002016", 6831., "2016", "QCDHT7002016ULNanoList.txt", "/QCD_HT700to1000_TuneCP5_PSWeights_13TeV-madgraph-pythia8/RunIISummer20UL16NanoAODv9-106X_mcRun2_asymptotic_v17-v1/NANOAODSIM")
QCDHT7002017      = sample("QCDHT7002017", 6831., "2017", "QCDHT7002017ULNanoList.txt", "/QCD_HT700to1000_TuneCP5_PSWeights_13TeV-madgraph-pythia8/RunIISummer20UL17NanoAODv9-106X_mc2017_realistic_v9-v1/NANOAODSIM")
QCDHT7002018      = sample("QCDHT7002018", 6831., "2018", "QCDHT7002018ULNanoList.txt", "/QCD_HT700to1000_TuneCP5_PSWeights_13TeV-madgraph-pythia8/RunIISummer20UL18NanoAODv9-106X_upgrade2018_realistic_v16_L1v1-v2/NANOAODSIM")
SingleElecRun2016APVB = sample("SingleElecRun2016APVB", 1.0, "2016APV", "SingleElecRun2016APVB2016APVULNanoList.txt", "/SingleElectron/Run2016B-ver2_HIPM_UL2016_MiniAODv2_NanoAODv9-v2/NANOAOD")
SingleElecRun2016APVC = sample("SingleElecRun2016APVC", 1.0, "2016APV", "SingleElecRun2016APVC2016APVULNanoList.txt", "/SingleElectron/Run2016C-HIPM_UL2016_MiniAODv2_NanoAODv9-v2/NANOAOD")
SingleElecRun2016APVD = sample("SingleElecRun2016APVD", 1.0, "2016APV", "SingleElecRun2016APVD2016APVULNanoList.txt", "/SingleElectron/Run2016D-HIPM_UL2016_MiniAODv2_NanoAODv9-v2/NANOAOD")
SingleElecRun2016APVE = sample("SingleElecRun2016APVE", 1.0, "2016APV", "SingleElecRun2016APVE2016APVULNanoList.txt", "/SingleElectron/Run2016E-HIPM_UL2016_MiniAODv2_NanoAODv9-v2/NANOAOD")
SingleElecRun2016APVF = sample("SingleElecRun2016APVF", 1.0, "2016APV", "SingleElecRun2016APVF2016APVULNanoList.txt", "/SingleElectron/Run2016F-HIPM_UL2016_MiniAODv2_NanoAODv9-v2/NANOAOD")
SingleElecRun2016F  = sample("SingleElecRun2016F", 1.0, "2016", "SingleElecRun2016F2016ULNanoList.txt", "/SingleElectron/Run2016F-UL2016_MiniAODv2_NanoAODv9-v1/NANOAOD")
SingleElecRun2016G  = sample("SingleElecRun2016G", 1.0, "2016", "SingleElecRun2016G2016ULNanoList.txt", "/SingleElectron/Run2016G-UL2016_MiniAODv2_NanoAODv9-v1/NANOAOD")
SingleElecRun2016H  = sample("SingleElecRun2016H", 1.0, "2016", "SingleElecRun2016H2016ULNanoList.txt", "/SingleElectron/Run2016H-UL2016_MiniAODv2_NanoAODv9-v1/NANOAOD")
SingleElecRun2017B  = sample("SingleElecRun2017B", 1.0, "2017", "SingleElecRun2017B2017ULNanoList.txt", "/SingleElectron/Run2017B-UL2017_MiniAODv2_NanoAODv9-v1/NANOAOD")
SingleElecRun2017C  = sample("SingleElecRun2017C", 1.0, "2017", "SingleElecRun2017C2017ULNanoList.txt", "/SingleElectron/Run2017C-UL2017_MiniAODv2_NanoAODv9-v1/NANOAOD")
SingleElecRun2017D  = sample("SingleElecRun2017D", 1.0, "2017", "SingleElecRun2017D2017ULNanoList.txt", "/SingleElectron/Run2017D-UL2017_MiniAODv2_NanoAODv9-v1/NANOAOD")
SingleElecRun2017E  = sample("SingleElecRun2017E", 1.0, "2017", "SingleElecRun2017E2017ULNanoList.txt", "/SingleElectron/Run2017E-UL2017_MiniAODv2_NanoAODv9-v1/NANOAOD")
SingleElecRun2017F  = sample("SingleElecRun2017F", 1.0, "2017", "SingleElecRun2017F2017ULNanoList.txt", "/SingleElectron/Run2017F-UL2017_MiniAODv2_NanoAODv9-v1/NANOAOD")
SingleElecRun2018A  = sample("SingleElecRun2018A", 1.0, "2018", "SingleElecRun2018A2018ULNanoList.txt", "/EGamma/Run2018A-UL2018_MiniAODv2_NanoAODv9-v1/NANOAOD")
SingleElecRun2018B  = sample("SingleElecRun2018B", 1.0, "2018", "SingleElecRun2018B2018ULNanoList.txt", "/EGamma/Run2018B-UL2018_MiniAODv2_NanoAODv9-v1/NANOAOD")
SingleElecRun2018C  = sample("SingleElecRun2018C", 1.0, "2018", "SingleElecRun2018C2018ULNanoList.txt", "/EGamma/Run2018B-UL2018_MiniAODv2_NanoAODv9-v1/NANOAOD")
SingleElecRun2018D  = sample("SingleElecRun2018D", 1.0, "2018", "SingleElecRun2018D2018ULNanoList.txt", "/EGamma/Run2018B-UL2018_MiniAODv2_NanoAODv9-v1/NANOAOD")
SingleMuonRun2016APVB = sample("SingleMuonRun2016APVB", 1.0, "2016APV", "SingleMuonRun2016APVB2016APVULNanoList.txt", "/SingleMuon/Run2016B-ver2_HIPM_UL2016_MiniAODv2_NanoAODv9-v2/NANOAOD")
SingleMuonRun2016APVC = sample("SingleMuonRun2016APVC", 1.0, "2016APV", "SingleMuonRun2016APVC2016APVULNanoList.txt", "/SingleMuon/Run2016C-HIPM_UL2016_MiniAODv2_NanoAODv9-v2/NANOAOD")
SingleMuonRun2016APVD = sample("SingleMuonRun2016APVD", 1.0, "2016APV", "SingleMuonRun2016APVD2016APVULNanoList.txt", "/SingleMuon/Run2016D-HIPM_UL2016_MiniAODv2_NanoAODv9-v2/NANOAOD")
SingleMuonRun2016APVE = sample("SingleMuonRun2016APVE", 1.0, "2016APV", "SingleMuonRun2016APVE2016APVULNanoList.txt", "/SingleMuon/Run2016E-HIPM_UL2016_MiniAODv2_NanoAODv9-v2/NANOAOD")
SingleMuonRun2016APVF = sample("SingleMuonRun2016APVF", 1.0, "2016APV", "SingleMuonRun2016APVF2016APVULNanoList.txt", "/SingleMuon/Run2016F-HIPM_UL2016_MiniAODv2_NanoAODv9-v2/NANOAOD")
SingleMuonRun2016F  = sample("SingleMuonRun2016F", 1.0, "2016", "SingleMuonRun2016F2016ULNanoList.txt", "/SingleMuon/Run2016F-UL2016_MiniAODv2_NanoAODv9-v1/NANOAOD")
SingleMuonRun2016G  = sample("SingleMuonRun2016G", 1.0, "2016", "SingleMuonRun2016G2016ULNanoList.txt", "/SingleMuon/Run2016G-UL2016_MiniAODv2_NanoAODv9-v1/NANOAOD")
SingleMuonRun2016H  = sample("SingleMuonRun2016H", 1.0, "2016", "SingleMuonRun2016H2016ULNanoList.txt", "/SingleMuon/Run2016H-UL2016_MiniAODv2_NanoAODv9-v1/NANOAOD")
SingleMuonRun2017B  = sample("SingleMuonRun2017B", 1.0, "2017", "SingleMuonRun2017B2017ULNanoList.txt", "/SingleMuon/Run2017B-UL2017_MiniAODv2_NanoAODv9-v1/NANOAOD")
SingleMuonRun2017C  = sample("SingleMuonRun2017C", 1.0, "2017", "SingleMuonRun2017C2017ULNanoList.txt", "/SingleMuon/Run2017C-UL2017_MiniAODv2_NanoAODv9-v1/NANOAOD")
SingleMuonRun2017D  = sample("SingleMuonRun2017D", 1.0, "2017", "SingleMuonRun2017D2017ULNanoList.txt", "/SingleMuon/Run2017D-UL2017_MiniAODv2_NanoAODv9-v1/NANOAOD")
SingleMuonRun2017E  = sample("SingleMuonRun2017E", 1.0, "2017", "SingleMuonRun2017E2017ULNanoList.txt", "/SingleMuon/Run2017E-UL2017_MiniAODv2_NanoAODv9-v1/NANOAOD")
SingleMuonRun2017F  = sample("SingleMuonRun2017F", 1.0, "2017", "SingleMuonRun2017F2017ULNanoList.txt", "/SingleMuon/Run2017F-UL2017_MiniAODv2_NanoAODv9-v1/NANOAOD")
SingleMuonRun2018A  = sample("SingleMuonRun2018A", 1.0, "2018", "SingleMuonRun2018A2018ULNanoList.txt", "/SingleMuon/Run2018A-UL2018_MiniAODv2_NanoAODv9-v2/NANOAOD")
SingleMuonRun2018B  = sample("SingleMuonRun2018B", 1.0, "2018", "SingleMuonRun2018B2018ULNanoList.txt", "/SingleMuon/Run2018B-UL2018_MiniAODv2_NanoAODv9-v2/NANOAOD")
SingleMuonRun2018C  = sample("SingleMuonRun2018C", 1.0, "2018", "SingleMuonRun2018C2018ULNanoList.txt", "/SingleMuon/Run2018C-UL2018_MiniAODv2_NanoAODv9-v2/NANOAOD")
SingleMuonRun2018D  = sample("SingleMuonRun2018D", 1.0, "2018", "SingleMuonRun2018D2018ULNanoList.txt", "/SingleMuon/Run2018D-UL2018_MiniAODv2_NanoAODv9-v1/NANOAOD")
STs2016APV        = sample("STs2016APV", 11.073*0.333, "2016APV", "STs2016APVULNanoList.txt", "/ST_s-channel_4f_leptonDecays_TuneCP5_13TeV-amcatnlo-pythia8/RunIISummer20UL16NanoAODAPVv9-106X_mcRun2_asymptotic_preVFP_v11-v1/NANOAODSIM")
STs2016           = sample("STs2016", 11.073*0.333, "2016", "STs2016ULNanoList.txt", "/ST_s-channel_4f_leptonDecays_TuneCP5_13TeV-amcatnlo-pythia8/RunIISummer20UL16NanoAODv9-106X_mcRun2_asymptotic_v17-v1/NANOAODSIM")
STs2017           = sample("STs2017", 11.073*0.333, "2017", "STs2017ULNanoList.txt", "/ST_s-channel_4f_leptonDecays_TuneCP5_13TeV-amcatnlo-pythia8/RunIISummer20UL17NanoAODv9-106X_mc2017_realistic_v9-v1/NANOAODSIM")
STs2018           = sample("STs2018", 11.073*0.333, "2018", "STs2018ULNanoList.txt", "/ST_s-channel_4f_leptonDecays_TuneCP5_13TeV-amcatnlo-pythia8/RunIISummer20UL18NanoAODv9-106X_upgrade2018_realistic_v16_L1v1-v1/NANOAODSIM")
STt2016APV        = sample("STt2016APV", 134.2, "2016APV", "STt2016APVULNanoList.txt", "/ST_t-channel_top_4f_InclusiveDecays_TuneCP5_13TeV-powheg-madspin-pythia8/RunIISummer20UL16NanoAODAPVv9-106X_mcRun2_asymptotic_preVFP_v11-v1/NANOAODSIM")
STt2016           = sample("STt2016", 134.2, "2016", "STt2016ULNanoList.txt", "/ST_t-channel_top_4f_InclusiveDecays_TuneCP5_13TeV-powheg-madspin-pythia8/RunIISummer20UL16NanoAODv9-106X_mcRun2_asymptotic_v17-v1/NANOAODSIM")
STt2017           = sample("STt2017", 134.2, "2017", "STt2017ULNanoList.txt", "/ST_t-channel_top_4f_InclusiveDecays_TuneCP5_13TeV-powheg-madspin-pythia8/RunIISummer20UL17NanoAODv9-106X_mc2017_realistic_v9-v1/NANOAODSIM")
STt2018           = sample("STt2018", 134.2, "2018", "STt2018ULNanoList.txt", "/ST_t-channel_top_4f_InclusiveDecays_TuneCP5_13TeV-powheg-madspin-pythia8/RunIISummer20UL18NanoAODv9-106X_upgrade2018_realistic_v16_L1v1-v1/NANOAODSIM")
STtb2016APV       = sample("STtb2016APV", 80.0, "2016APV", "STtb2016APVULNanoList.txt", "/ST_t-channel_antitop_4f_InclusiveDecays_TuneCP5_13TeV-powheg-madspin-pythia8/RunIISummer20UL16NanoAODAPVv9-106X_mcRun2_asymptotic_preVFP_v11-v1/NANOAODSIM")
STtb2016          = sample("STtb2016", 80.0, "2016", "STtb2016ULNanoList.txt", "/ST_t-channel_antitop_4f_InclusiveDecays_TuneCP5_13TeV-powheg-madspin-pythia8/RunIISummer20UL16NanoAODv9-106X_mcRun2_asymptotic_v17-v1/NANOAODSIM")
STtb2017          = sample("STtb2017", 80.0, "2017", "STtb2017ULNanoList.txt", "/ST_t-channel_antitop_4f_InclusiveDecays_TuneCP5_13TeV-powheg-madspin-pythia8/RunIISummer20UL17NanoAODv9-106X_mc2017_realistic_v9-v1/NANOAODSIM")
STtb2018          = sample("STtb2018", 80.0, "2018", "STtb2018ULNanoList.txt", "/ST_t-channel_antitop_4f_InclusiveDecays_TuneCP5_13TeV-powheg-madspin-pythia8/RunIISummer20UL18NanoAODv9-106X_upgrade2018_realistic_v16_L1v1-v1/NANOAODSIM")
STtW2016APV       = sample("STtW2016APV", 39.65, "2016APV", "STtW2016APVULNanoList.txt", "/ST_tW_top_5f_inclusiveDecays_TuneCP5_13TeV-powheg-pythia8/RunIISummer20UL16NanoAODAPVv9-106X_mcRun2_asymptotic_preVFP_v11-v1/NANOAODSIM")
STtW2016          = sample("STtW2016", 39.65, "2016", "STtW2016ULNanoList.txt", "/ST_tW_top_5f_inclusiveDecays_TuneCP5_13TeV-powheg-pythia8/RunIISummer20UL16NanoAODv9-106X_mcRun2_asymptotic_v17-v2/NANOAODSIM")
STtW2017          = sample("STtW2017", 39.65, "2017", "STtW2017ULNanoList.txt", "/ST_tW_top_5f_inclusiveDecays_TuneCP5_13TeV-powheg-pythia8/RunIISummer20UL17NanoAODv9-106X_mc2017_realistic_v9-v2/NANOAODSIM")
STtW2018          = sample("STtW2018", 39.65, "2018", "STtW2018ULNanoList.txt", "/ST_tW_top_5f_inclusiveDecays_TuneCP5_13TeV-powheg-pythia8/RunIISummer20UL18NanoAODv9-106X_upgrade2018_realistic_v16_L1v1-v2/NANOAODSIM")
STtWb2016APV      = sample("STtWb2016APV", 39.65, "2016APV", "STtWb2016APVULNanoList.txt", "/ST_tW_antitop_5f_inclusiveDecays_TuneCP5_13TeV-powheg-pythia8/RunIISummer20UL16NanoAODAPVv9-106X_mcRun2_asymptotic_preVFP_v11-v1/NANOAODSIM")
STtWb2016         = sample("STtWb2016", 39.65, "2016", "STtWb2016ULNanoList.txt", "/ST_tW_antitop_5f_inclusiveDecays_TuneCP5_13TeV-powheg-pythia8/RunIISummer20UL16NanoAODv9-106X_mcRun2_asymptotic_v17-v2/NANOAODSIM")
STtWb2017         = sample("STtWb2017", 39.65, "2017", "STtWb2017ULNanoList.txt", "/ST_tW_antitop_5f_inclusiveDecays_TuneCP5_13TeV-powheg-pythia8/RunIISummer20UL17NanoAODv9-106X_mc2017_realistic_v9-v2/NANOAODSIM")
STtWb2018         = sample("STtWb2018", 39.65, "2018", "STtWb2018ULNanoList.txt", "/ST_tW_antitop_5f_inclusiveDecays_TuneCP5_13TeV-powheg-pythia8/RunIISummer20UL18NanoAODv9-106X_upgrade2018_realistic_v16_L1v1-v2/NANOAODSIM")
TTHB2016APV       = sample("TTHB2016APV", 0.2926, "2016APV", "TTHB2016APVULNanoList.txt", "/ttHTobb_M125_TuneCP5_13TeV-powheg-pythia8/RunIISummer20UL16NanoAODAPVv9-106X_mcRun2_asymptotic_preVFP_v11-v2/NANOAODSIM")
TTHB2016          = sample("TTHB2016", 0.2926, "2016", "TTHB2016ULNanoList.txt", "/ttHTobb_M125_TuneCP5_13TeV-powheg-pythia8/RunIISummer20UL16NanoAODv9-106X_mcRun2_asymptotic_v17-v2/NANOAODSIM")
TTHB2017          = sample("TTHB2017", 0.2926, "2017", "TTHB2017ULNanoList.txt", "/ttHTobb_M125_TuneCP5_13TeV-powheg-pythia8/RunIISummer20UL17NanoAODv9-106X_mc2017_realistic_v9-v2/NANOAODSIM")
TTHB2018          = sample("TTHB2018", 0.2926, "2018", "TTHB2018ULNanoList.txt", "/ttHTobb_M125_TuneCP5_13TeV-powheg-pythia8/RunIISummer20UL18NanoAODv9-106X_upgrade2018_realistic_v16_L1v1-v2/NANOAODSIM")
TTHnonB2016APV    = sample("TTHnonB2016APV", 0.2126, "2016APV", "TTHnonB2016APVULNanoList.txt", "/ttHToNonbb_M125_TuneCP5_13TeV-powheg-pythia8/RunIISummer20UL16NanoAODAPVv9-106X_mcRun2_asymptotic_preVFP_v11-v2/NANOAODSIM")
TTHnonB2016       = sample("TTHnonB2016", 0.2126, "2016", "TTHnonB2016ULNanoList.txt", "/ttHToNonbb_M125_TuneCP5_13TeV-powheg-pythia8/RunIISummer20UL16NanoAODv9-106X_mcRun2_asymptotic_v17-v2/NANOAODSIM")
TTHnonB2017       = sample("TTHnonB2017", 0.2126, "2017", "TTHnonB2017ULNanoList.txt", "/ttHToNonbb_M125_TuneCP5_13TeV-powheg-pythia8/RunIISummer20UL17NanoAODv9-106X_mc2017_realistic_v9-v2/NANOAODSIM")
TTHnonB2018       = sample("TTHnonB2018", 0.2126, "2018", "TTHnonB2018ULNanoList.txt", "/ttHToNonbb_M125_TuneCP5_13TeV-powheg-pythia8/RunIISummer20UL18NanoAODv9-106X_upgrade2018_realistic_v16_L1v1-v2/NANOAODSIM")
TTMT10002016APV   = sample("TTMT10002016APV", 831.76*0.02474, "2016APV", "TTMT10002016APVULNanoList.txt", "/TT_Mtt-1000toInf_TuneCP5_13TeV-powheg-pythia8/RunIISummer20UL16NanoAODAPVv9-106X_mcRun2_asymptotic_preVFP_v11-v1/NANOAODSIM")
TTMT10002016      = sample("TTMT10002016", 831.76*0.02474, "2016", "TTMT10002016ULNanoList.txt", "/TT_Mtt-1000toInf_TuneCP5_13TeV-powheg-pythia8/RunIISummer20UL16NanoAODv9-106X_mcRun2_asymptotic_v17-v1/NANOAODSIM")
TTMT10002017      = sample("TTMT10002017", 831.76*0.02474, "2017", "TTMT10002017ULNanoList.txt", "/TT_Mtt-1000toInf_TuneCP5_13TeV-powheg-pythia8/RunIISummer20UL17NanoAODv9-106X_mc2017_realistic_v9-v2/NANOAODSIM")
TTMT10002018      = sample("TTMT10002018", 831.76*0.02474, "2018", "TTMT10002018ULNanoList.txt", "/TT_Mtt-1000toInf_TuneCP5_13TeV-powheg-pythia8/RunIISummer20UL18NanoAODv9-106X_upgrade2018_realistic_v16_L1v1-v1/NANOAODSIM")
TTMT7002016APV    = sample("TTMT7002016APV", 831.76*0.0921, "2016APV", "TTMT7002016APVULNanoList.txt", "/TT_Mtt-700to1000_TuneCP5_13TeV-powheg-pythia8/RunIISummer20UL16NanoAODAPVv9-106X_mcRun2_asymptotic_preVFP_v11-v1/NANOAODSIM")
TTMT7002016       = sample("TTMT7002016", 831.76*0.0921, "2016", "TTMT7002016ULNanoList.txt", "/TT_Mtt-700to1000_TuneCP5_13TeV-powheg-pythia8/RunIISummer20UL16NanoAODv9-106X_mcRun2_asymptotic_v17-v1/NANOAODSIM")
TTMT7002017       = sample("TTMT7002017", 831.76*0.0921, "2017", "TTMT7002017ULNanoList.txt", "/TT_Mtt-700to1000_TuneCP5_13TeV-powheg-pythia8/RunIISummer20UL17NanoAODv9-106X_mc2017_realistic_v9-v2/NANOAODSIM")
TTMT7002018       = sample("TTMT7002018", 831.76*0.0921, "2018", "TTMT7002018ULNanoList.txt", "/TT_Mtt-700to1000_TuneCP5_13TeV-powheg-pythia8/RunIISummer20UL18NanoAODv9-106X_upgrade2018_realistic_v16_L1v1-v1/NANOAODSIM")

TTTo2L2Nu2016APV  = sample("TTTo2L2Nu2016APV", 831.76*0.105, "2016APV", "TTTo2L2Nu2016APVULNanoList.txt", "/TTTo2L2Nu_TuneCP5_13TeV-powheg-pythia8/RunIISummer20UL16NanoAODAPVv9-106X_mcRun2_asymptotic_preVFP_v11-v1/NANOAODSIM")
TTTo2L2Nu2016     = sample("TTTo2L2Nu2016", 831.76*0.105, "2016", "TTTo2L2Nu2016ULNanoList.txt", "/TTTo2L2Nu_TuneCP5_13TeV-powheg-pythia8/RunIISummer20UL16NanoAODv9-106X_mcRun2_asymptotic_v17-v1/NANOAODSIM")
TTTo2L2Nu2017     = sample("TTTo2L2Nu2017", 831.76*0.105, "2017", "TTTo2L2Nu2017ULNanoList.txt", "/TTTo2L2Nu_TuneCP5_13TeV-powheg-pythia8/RunIISummer20UL17NanoAODv9-106X_mc2017_realistic_v9-v1/NANOAODSIM")
TTTo2L2Nu2018     = sample("TTTo2L2Nu2018", 831.76*0.105, "2018", "TTTo2L2Nu2018ULNanoList.txt", "/TTTo2L2Nu_TuneCP5_13TeV-powheg-pythia8/RunIISummer20UL18NanoAODv9-106X_upgrade2018_realistic_v16_L1v1-v1/NANOAODSIM")
TTTo2L2Nu2016APV0  = sample("TTTo2L2Nu2016APV0", 831.76*0.105*0.8832, "2016APV", "TTTo2L2Nu2016APVULNanoList.txt", "/TTTo2L2Nu_TuneCP5_13TeV-powheg-pythia8/RunIISummer20UL16NanoAODAPVv9-106X_mcRun2_asymptotic_preVFP_v11-v1/NANOAODSIM")
TTTo2L2Nu20160     = sample("TTTo2L2Nu20160", 831.76*0.105*0.8832, "2016", "TTTo2L2Nu2016ULNanoList.txt", "/TTTo2L2Nu_TuneCP5_13TeV-powheg-pythia8/RunIISummer20UL16NanoAODv9-106X_mcRun2_asymptotic_v17-v1/NANOAODSIM")
TTTo2L2Nu20170     = sample("TTTo2L2Nu20170", 831.76*0.105*0.8832, "2017", "TTTo2L2Nu2017ULNanoList.txt", "/TTTo2L2Nu_TuneCP5_13TeV-powheg-pythia8/RunIISummer20UL17NanoAODv9-106X_mc2017_realistic_v9-v1/NANOAODSIM")
TTTo2L2Nu20180     = sample("TTTo2L2Nu20180", 831.76*0.105*0.8832, "2018", "TTTo2L2Nu2018ULNanoList.txt", "/TTTo2L2Nu_TuneCP5_13TeV-powheg-pythia8/RunIISummer20UL18NanoAODv9-106X_upgrade2018_realistic_v16_L1v1-v1/NANOAODSIM")
TTTo2L2Nu2016APV700  = sample("TTTo2L2Nu2016APV700", 831.76*0.105*0.0921, "2016APV", "TTTo2L2Nu2016APVULNanoList.txt", "/TTTo2L2Nu_TuneCP5_13TeV-powheg-pythia8/RunIISummer20UL16NanoAODAPVv9-106X_mcRun2_asymptotic_preVFP_v11-v1/NANOAODSIM")
TTTo2L2Nu2016700     = sample("TTTo2L2Nu2016700", 831.76*0.105*0.0921, "2016", "TTTo2L2Nu2016ULNanoList.txt", "/TTTo2L2Nu_TuneCP5_13TeV-powheg-pythia8/RunIISummer20UL16NanoAODv9-106X_mcRun2_asymptotic_v17-v1/NANOAODSIM")
TTTo2L2Nu2017700     = sample("TTTo2L2Nu2017700", 831.76*0.105*0.0921, "2017", "TTTo2L2Nu2017ULNanoList.txt", "/TTTo2L2Nu_TuneCP5_13TeV-powheg-pythia8/RunIISummer20UL17NanoAODv9-106X_mc2017_realistic_v9-v1/NANOAODSIM")
TTTo2L2Nu2018700     = sample("TTTo2L2Nu2018700", 831.76*0.105*0.0921, "2018", "TTTo2L2Nu2018ULNanoList.txt", "/TTTo2L2Nu_TuneCP5_13TeV-powheg-pythia8/RunIISummer20UL18NanoAODv9-106X_upgrade2018_realistic_v16_L1v1-v1/NANOAODSIM")
TTTo2L2Nu2016APV1000  = sample("TTTo2L2Nu2016APV1000", 831.76*0.105*0.02474, "2016APV", "TTTo2L2Nu2016APVULNanoList.txt", "/TTTo2L2Nu_TuneCP5_13TeV-powheg-pythia8/RunIISummer20UL16NanoAODAPVv9-106X_mcRun2_asymptotic_preVFP_v11-v1/NANOAODSIM")
TTTo2L2Nu20161000     = sample("TTTo2L2Nu20161000", 831.76*0.105*0.02474, "2016", "TTTo2L2Nu2016ULNanoList.txt", "/TTTo2L2Nu_TuneCP5_13TeV-powheg-pythia8/RunIISummer20UL16NanoAODv9-106X_mcRun2_asymptotic_v17-v1/NANOAODSIM")
TTTo2L2Nu20171000     = sample("TTTo2L2Nu20171000", 831.76*0.105*0.02474, "2017", "TTTo2L2Nu2017ULNanoList.txt", "/TTTo2L2Nu_TuneCP5_13TeV-powheg-pythia8/RunIISummer20UL17NanoAODv9-106X_mc2017_realistic_v9-v1/NANOAODSIM")
TTTo2L2Nu20181000     = sample("TTTo2L2Nu20181000", 831.76*0.105*0.02474, "2018", "TTTo2L2Nu2018ULNanoList.txt", "/TTTo2L2Nu_TuneCP5_13TeV-powheg-pythia8/RunIISummer20UL18NanoAODv9-106X_upgrade2018_realistic_v16_L1v1-v1/NANOAODSIM")

TTToHadronic2016APV = sample("TTToHadronic2016APV", 831.76*0.457, "2016APV", "TTToHadronic2016APVULNanoList.txt", "/TTToHadronic_TuneCP5_13TeV-powheg-pythia8/RunIISummer20UL16NanoAODAPVv9-106X_mcRun2_asymptotic_preVFP_v11-v1/NANOAODSIM")
TTToHadronic2016  = sample("TTToHadronic2016", 831.76*0.457, "2016", "TTToHadronic2016ULNanoList.txt", "/TTToHadronic_TuneCP5_13TeV-powheg-pythia8/RunIISummer20UL16NanoAODv9-106X_mcRun2_asymptotic_v17-v1/NANOAODSIM")
TTToHadronic2017  = sample("TTToHadronic2017", 831.76*0.457, "2017", "TTToHadronic2017ULNanoList.txt", "/TTToHadronic_TuneCP5_13TeV-powheg-pythia8/RunIISummer20UL17NanoAODv9-106X_mc2017_realistic_v9-v1/NANOAODSIM")
TTToHadronic2018  = sample("TTToHadronic2018", 831.76*0.457, "2018", "TTToHadronic2018ULNanoList.txt", "/TTToHadronic_TuneCP5_13TeV-powheg-pythia8/RunIISummer20UL18NanoAODv9-106X_upgrade2018_realistic_v16_L1v1-v1/NANOAODSIM")
TTToHadronic2016APV0 = sample("TTToHadronic2016APV0", 831.76*0.457*0.8832, "2016APV", "TTToHadronic2016APVULNanoList.txt", "/TTToHadronic_TuneCP5_13TeV-powheg-pythia8/RunIISummer20UL16NanoAODAPVv9-106X_mcRun2_asymptotic_preVFP_v11-v1/NANOAODSIM")
TTToHadronic20160  = sample("TTToHadronic20160", 831.76*0.457*0.8832, "2016", "TTToHadronic2016ULNanoList.txt", "/TTToHadronic_TuneCP5_13TeV-powheg-pythia8/RunIISummer20UL16NanoAODv9-106X_mcRun2_asymptotic_v17-v1/NANOAODSIM")
TTToHadronic20170  = sample("TTToHadronic20170", 831.76*0.457*0.8832, "2017", "TTToHadronic2017ULNanoList.txt", "/TTToHadronic_TuneCP5_13TeV-powheg-pythia8/RunIISummer20UL17NanoAODv9-106X_mc2017_realistic_v9-v1/NANOAODSIM")
TTToHadronic20180  = sample("TTToHadronic20180", 831.76*0.457*0.8832, "2018", "TTToHadronic2018ULNanoList.txt", "/TTToHadronic_TuneCP5_13TeV-powheg-pythia8/RunIISummer20UL18NanoAODv9-106X_upgrade2018_realistic_v16_L1v1-v1/NANOAODSIM")
TTToHadronic2016APV700 = sample("TTToHadronic2016APV700", 831.76*0.457*0.0921, "2016APV", "TTToHadronic2016APVULNanoList.txt", "/TTToHadronic_TuneCP5_13TeV-powheg-pythia8/RunIISummer20UL16NanoAODAPVv9-106X_mcRun2_asymptotic_preVFP_v11-v1/NANOAODSIM")
TTToHadronic2016700  = sample("TTToHadronic2016700", 831.76*0.457*0.0921, "2016", "TTToHadronic2016ULNanoList.txt", "/TTToHadronic_TuneCP5_13TeV-powheg-pythia8/RunIISummer20UL16NanoAODv9-106X_mcRun2_asymptotic_v17-v1/NANOAODSIM")
TTToHadronic2017700  = sample("TTToHadronic2017700", 831.76*0.457*0.0921, "2017", "TTToHadronic2017ULNanoList.txt", "/TTToHadronic_TuneCP5_13TeV-powheg-pythia8/RunIISummer20UL17NanoAODv9-106X_mc2017_realistic_v9-v1/NANOAODSIM")
TTToHadronic2018700  = sample("TTToHadronic2018700", 831.76*0.457*0.0921, "2018", "TTToHadronic2018ULNanoList.txt", "/TTToHadronic_TuneCP5_13TeV-powheg-pythia8/RunIISummer20UL18NanoAODv9-106X_upgrade2018_realistic_v16_L1v1-v1/NANOAODSIM")
TTToHadronic2016APV1000 = sample("TTToHadronic2016APV1000", 831.76*0.457*0.02474, "2016APV", "TTToHadronic2016APVULNanoList.txt", "/TTToHadronic_TuneCP5_13TeV-powheg-pythia8/RunIISummer20UL16NanoAODAPVv9-106X_mcRun2_asymptotic_preVFP_v11-v1/NANOAODSIM")
TTToHadronic20161000  = sample("TTToHadronic20161000", 831.76*0.457*0.02474, "2016", "TTToHadronic2016ULNanoList.txt", "/TTToHadronic_TuneCP5_13TeV-powheg-pythia8/RunIISummer20UL16NanoAODv9-106X_mcRun2_asymptotic_v17-v1/NANOAODSIM")
TTToHadronic20171000  = sample("TTToHadronic20171000", 831.76*0.457*0.02474, "2017", "TTToHadronic2017ULNanoList.txt", "/TTToHadronic_TuneCP5_13TeV-powheg-pythia8/RunIISummer20UL17NanoAODv9-106X_mc2017_realistic_v9-v1/NANOAODSIM")
TTToHadronic20181000  = sample("TTToHadronic20181000", 831.76*0.457*0.02474, "2018", "TTToHadronic2018ULNanoList.txt", "/TTToHadronic_TuneCP5_13TeV-powheg-pythia8/RunIISummer20UL18NanoAODv9-106X_upgrade2018_realistic_v16_L1v1-v1/NANOAODSIM")

TTToSemiLeptonic2016APV = sample("TTToSemiLeptonic2016APV", 831.76*0.438, "2016APV", "TTToSemiLeptonic2016APVULNanoList.txt", "/TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8/RunIISummer20UL16NanoAODAPVv9-106X_mcRun2_asymptotic_preVFP_v11-v1/NANOAODSIM")
TTToSemiLeptonic2016 = sample("TTToSemiLeptonic2016", 831.76*0.438, "2016", "TTToSemiLeptonic2016ULNanoList.txt", "/TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8/RunIISummer20UL16NanoAODv9-106X_mcRun2_asymptotic_v17-v1/NANOAODSIM")
TTToSemiLeptonic2017 = sample("TTToSemiLeptonic2017", 831.76*0.438, "2017", "TTToSemiLeptonic2017ULNanoList.txt", "/TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8/RunIISummer20UL17NanoAODv9-106X_mc2017_realistic_v9-v1/NANOAODSIM")
TTToSemiLeptonic2018 = sample("TTToSemiLeptonic2018", 831.76*0.438, "2018", "TTToSemiLeptonic2018ULNanoList.txt", "/TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8/RunIISummer20UL18NanoAODv9-106X_upgrade2018_realistic_v16_L1v1-v1/NANOAODSIM")
TTToSemiLeptonic2016APV0 = sample("TTToSemiLeptonic2016APV0", 831.76*0.438*0.8832, "2016APV", "TTToSemiLeptonic2016APVULNanoList.txt", "/TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8/RunIISummer20UL16NanoAODAPVv9-106X_mcRun2_asymptotic_preVFP_v11-v1/NANOAODSIM")
TTToSemiLeptonic20160 = sample("TTToSemiLeptonic20160", 831.76*0.438*0.8832, "2016", "TTToSemiLeptonic2016ULNanoList.txt", "/TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8/RunIISummer20UL16NanoAODv9-106X_mcRun2_asymptotic_v17-v1/NANOAODSIM")
TTToSemiLeptonic20170 = sample("TTToSemiLeptonic20170", 831.76*0.438*0.8832, "2017", "TTToSemiLeptonic2017ULNanoList.txt", "/TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8/RunIISummer20UL17NanoAODv9-106X_mc2017_realistic_v9-v1/NANOAODSIM")
TTToSemiLeptonic20180 = sample("TTToSemiLeptonic20180", 831.76*0.438*0.8832, "2018", "TTToSemiLeptonic2018ULNanoList.txt", "/TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8/RunIISummer20UL18NanoAODv9-106X_upgrade2018_realistic_v16_L1v1-v1/NANOAODSIM")
TTToSemiLeptonic2016APV700 = sample("TTToSemiLeptonic2016APV700", 831.76*0.438*0.0921, "2016APV", "TTToSemiLeptonic2016APVULNanoList.txt", "/TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8/RunIISummer20UL16NanoAODAPVv9-106X_mcRun2_asymptotic_preVFP_v11-v1/NANOAODSIM")
TTToSemiLeptonic2016700 = sample("TTToSemiLeptonic2016700", 831.76*0.438*0.0921, "2016", "TTToSemiLeptonic2016ULNanoList.txt", "/TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8/RunIISummer20UL16NanoAODv9-106X_mcRun2_asymptotic_v17-v1/NANOAODSIM")
TTToSemiLeptonic2017700 = sample("TTToSemiLeptonic2017700", 831.76*0.438*0.0921, "2017", "TTToSemiLeptonic2017ULNanoList.txt", "/TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8/RunIISummer20UL17NanoAODv9-106X_mc2017_realistic_v9-v1/NANOAODSIM")
TTToSemiLeptonic2018700 = sample("TTToSemiLeptonic2018700", 831.76*0.438*0.0921, "2018", "TTToSemiLeptonic2018ULNanoList.txt", "/TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8/RunIISummer20UL18NanoAODv9-106X_upgrade2018_realistic_v16_L1v1-v1/NANOAODSIM")
TTToSemiLeptonic2016APV1000 = sample("TTToSemiLeptonic2016APV1000", 831.76*0.438*0.02474, "2016APV", "TTToSemiLeptonic2016APVULNanoList.txt", "/TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8/RunIISummer20UL16NanoAODAPVv9-106X_mcRun2_asymptotic_preVFP_v11-v1/NANOAODSIM")
TTToSemiLeptonic20161000 = sample("TTToSemiLeptonic20161000", 831.76*0.438*0.02474, "2016", "TTToSemiLeptonic2016ULNanoList.txt", "/TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8/RunIISummer20UL16NanoAODv9-106X_mcRun2_asymptotic_v17-v1/NANOAODSIM")
TTToSemiLeptonic20171000 = sample("TTToSemiLeptonic20171000", 831.76*0.438*0.02474, "2017", "TTToSemiLeptonic2017ULNanoList.txt", "/TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8/RunIISummer20UL17NanoAODv9-106X_mc2017_realistic_v9-v1/NANOAODSIM")
TTToSemiLeptonic20181000 = sample("TTToSemiLeptonic20181000", 831.76*0.438*0.02474, "2018", "TTToSemiLeptonic2018ULNanoList.txt", "/TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8/RunIISummer20UL18NanoAODv9-106X_upgrade2018_realistic_v16_L1v1-v1/NANOAODSIM")

TTWl2016APV       = sample("TTWl2016APV", 0.2043, "2016APV", "TTWl2016APVULNanoList.txt", "/TTWJetsToLNu_TuneCP5_13TeV-amcatnloFXFX-madspin-pythia8/RunIISummer20UL16NanoAODAPVv9-106X_mcRun2_asymptotic_preVFP_v11-v2/NANOAODSIM")
TTWl2016          = sample("TTWl2016", 0.2043, "2016", "TTWl2016ULNanoList.txt", "/TTWJetsToLNu_TuneCP5_13TeV-amcatnloFXFX-madspin-pythia8/RunIISummer20UL16NanoAODv9-106X_mcRun2_asymptotic_v17-v1/NANOAODSIM")
TTWl2017          = sample("TTWl2017", 0.2043, "2017", "TTWl2017ULNanoList.txt", "/TTWJetsToLNu_TuneCP5_13TeV-amcatnloFXFX-madspin-pythia8/RunIISummer20UL17NanoAODv9-106X_mc2017_realistic_v9-v1/NANOAODSIM")
TTWl2018          = sample("TTWl2018", 0.2043, "2018", "TTWl2018ULNanoList.txt", "/TTWJetsToLNu_TuneCP5_13TeV-amcatnloFXFX-madspin-pythia8/RunIISummer20UL18NanoAODv9-106X_upgrade2018_realistic_v16_L1v1-v1/NANOAODSIM")
TTWq2016APV       = sample("TTWq2016APV", 0.4062, "2016APV", "TTWq2016APVULNanoList.txt", "/TTWJetsToQQ_TuneCP5_13TeV-amcatnloFXFX-madspin-pythia8/RunIISummer20UL16NanoAODAPVv9-106X_mcRun2_asymptotic_preVFP_v11-v2/NANOAODSIM")
TTWq2016          = sample("TTWq2016", 0.4062, "2016", "TTWq2016ULNanoList.txt", "/TTWJetsToQQ_TuneCP5_13TeV-amcatnloFXFX-madspin-pythia8/RunIISummer20UL16NanoAODv9-106X_mcRun2_asymptotic_v17-v1/NANOAODSIM")
TTWq2017          = sample("TTWq2017", 0.4062, "2017", "TTWq2017ULNanoList.txt", "/TTWJetsToQQ_TuneCP5_13TeV-amcatnloFXFX-madspin-pythia8/RunIISummer20UL17NanoAODv9-106X_mc2017_realistic_v9-v1/NANOAODSIM")
TTWq2018          = sample("TTWq2018", 0.4062, "2018", "TTWq2018ULNanoList.txt", "/TTWJetsToQQ_TuneCP5_13TeV-amcatnloFXFX-madspin-pythia8/RunIISummer20UL18NanoAODv9-106X_upgrade2018_realistic_v16_L1v1-v1/NANOAODSIM")
TTZM102016APV     = sample("TTZM102016APV", 0.2529, "2016APV", "TTZM102016APVULNanoList.txt", "/TTZToLLNuNu_M-10_TuneCP5_13TeV-amcatnlo-pythia8/RunIISummer20UL16NanoAODAPVv9-106X_mcRun2_asymptotic_preVFP_v11-v1/NANOAODSIM")
TTZM102016        = sample("TTZM102016", 0.2529, "2016", "TTZM102016ULNanoList.txt", "/TTZToLLNuNu_M-10_TuneCP5_13TeV-amcatnlo-pythia8/RunIISummer20UL16NanoAODv9-106X_mcRun2_asymptotic_v17-v1/NANOAODSIM")
TTZM102017        = sample("TTZM102017", 0.2529, "2017", "TTZM102017ULNanoList.txt", "/TTZToLLNuNu_M-10_TuneCP5_13TeV-amcatnlo-pythia8/RunIISummer20UL17NanoAODv9-106X_mc2017_realistic_v9-v1/NANOAODSIM")
TTZM102018        = sample("TTZM102018", 0.2529, "2018", "TTZM102018ULNanoList.txt", "/TTZToLLNuNu_M-10_TuneCP5_13TeV-amcatnlo-pythia8/RunIISummer20UL18NanoAODv9-106X_upgrade2018_realistic_v16_L1v1-v1/NANOAODSIM")
TTZM1to102016APV  = sample("TTZM1to102016APV", 0.0537, "2016APV", "TTZM1to102016APVULNanoList.txt", "/TTZToLL_M-1to10_TuneCP5_13TeV-amcatnlo-pythia8/RunIISummer20UL16NanoAODv9-106X_mcRun2_asymptotic_v17-v1/NANOAODSIM")
TTZM1to102016     = sample("TTZM1to102016", 0.0537, "2016", "TTZM1to102016ULNanoList.txt", "/TTZToLL_M-1to10_TuneCP5_13TeV-amcatnlo-pythia8/RunIISummer20UL16NanoAODv9-106X_mcRun2_asymptotic_v17-v1/NANOAODSIM")
TTZM1to102017     = sample("TTZM1to102017", 0.0537, "2017", "TTZM1to102017ULNanoList.txt", "/TTZToLL_M-1to10_TuneCP5_13TeV-amcatnlo-pythia8/RunIISummer20UL17NanoAODv9-106X_mc2017_realistic_v9-v1/NANOAODSIM")
TTZM1to102018     = sample("TTZM1to102018", 0.0537, "2018", "TTZM1to102018ULNanoList.txt", "/TTZToLL_M-1to10_TuneCP5_13TeV-amcatnlo-pythia8/RunIISummer20UL18NanoAODv9-106X_upgrade2018_realistic_v16_L1v1-v1/NANOAODSIM")
WJetsHT12002016APV = sample("WJetsHT12002016APV", 1.329*1.21, "2016APV", "WJetsHT12002016APVULNanoList.txt", "/WJetsToLNu_HT-1200To2500_TuneCP5_13TeV-madgraphMLM-pythia8/RunIISummer20UL16NanoAODAPVv9-106X_mcRun2_asymptotic_preVFP_v11-v1/NANOAODSIM")
WJetsHT12002016   = sample("WJetsHT12002016", 1.329*1.21, "2016", "WJetsHT12002016ULNanoList.txt", "/WJetsToLNu_HT-1200To2500_TuneCP5_13TeV-madgraphMLM-pythia8/RunIISummer20UL16NanoAODv9-106X_mcRun2_asymptotic_v17-v1/NANOAODSIM")
WJetsHT12002017   = sample("WJetsHT12002017", 1.329*1.21, "2017", "WJetsHT12002017ULNanoList.txt", "/WJetsToLNu_HT-1200To2500_TuneCP5_13TeV-madgraphMLM-pythia8/RunIISummer20UL17NanoAODv9-106X_mc2017_realistic_v9-v1/NANOAODSIM")
WJetsHT12002018   = sample("WJetsHT12002018", 1.329*1.21, "2018", "WJetsHT12002018ULNanoList.txt", "/WJetsToLNu_HT-1200To2500_TuneCP5_13TeV-madgraphMLM-pythia8/RunIISummer20UL18NanoAODv9-106X_upgrade2018_realistic_v16_L1v1-v1/NANOAODSIM")
WJetsHT2002016APV = sample("WJetsHT2002016APV", 359.7*1.21, "2016APV", "WJetsHT2002016APVULNanoList.txt", "/WJetsToLNu_HT-200To400_TuneCP5_13TeV-madgraphMLM-pythia8/RunIISummer20UL16NanoAODAPVv9-106X_mcRun2_asymptotic_preVFP_v11-v1/NANOAODSIM")
WJetsHT2002016    = sample("WJetsHT2002016", 359.7*1.21, "2016", "WJetsHT2002016ULNanoList.txt", "/WJetsToLNu_HT-200To400_TuneCP5_13TeV-madgraphMLM-pythia8/RunIISummer20UL16NanoAODv9-106X_mcRun2_asymptotic_v17-v1/NANOAODSIM")
WJetsHT2002017    = sample("WJetsHT2002017", 359.7*1.21, "2017", "WJetsHT2002017ULNanoList.txt", "/WJetsToLNu_HT-200To400_TuneCP5_13TeV-madgraphMLM-pythia8/RunIISummer20UL17NanoAODv9-106X_mc2017_realistic_v9-v1/NANOAODSIM")
WJetsHT2002018    = sample("WJetsHT2002018", 359.7*1.21, "2018", "WJetsHT2002018ULNanoList.txt", "/WJetsToLNu_HT-200To400_TuneCP5_13TeV-madgraphMLM-pythia8/RunIISummer20UL18NanoAODv9-106X_upgrade2018_realistic_v16_L1v1-v1/NANOAODSIM")
WJetsHT25002016APV = sample("WJetsHT25002016APV", 0.03216*1.21, "2016APV", "WJetsHT25002016APVULNanoList.txt", "/WJetsToLNu_HT-2500ToInf_TuneCP5_13TeV-madgraphMLM-pythia8/RunIISummer20UL16NanoAODAPVv9-106X_mcRun2_asymptotic_preVFP_v11-v2/NANOAODSIM")
WJetsHT25002016   = sample("WJetsHT25002016", 0.03216*1.21, "2016", "WJetsHT25002016ULNanoList.txt", "/WJetsToLNu_HT-2500ToInf_TuneCP5_13TeV-madgraphMLM-pythia8/RunIISummer20UL16NanoAODv9-106X_mcRun2_asymptotic_v17-v2/NANOAODSIM")
WJetsHT25002017   = sample("WJetsHT25002017", 0.03216*1.21, "2017", "WJetsHT25002017ULNanoList.txt", "/WJetsToLNu_HT-2500ToInf_TuneCP5_13TeV-madgraphMLM-pythia8/RunIISummer20UL17NanoAODv9-106X_mc2017_realistic_v9-v2/NANOAODSIM")
WJetsHT25002018   = sample("WJetsHT25002018", 0.03216*1.21, "2018", "WJetsHT25002018ULNanoList.txt", "/WJetsToLNu_HT-2500ToInf_TuneCP5_13TeV-madgraphMLM-pythia8/RunIISummer20UL18NanoAODv9-106X_upgrade2018_realistic_v16_L1v1-v2/NANOAODSIM")
WJetsHT4002016APV = sample("WJetsHT4002016APV", 48.91*1.21, "2016APV", "WJetsHT4002016APVULNanoList.txt", "/WJetsToLNu_HT-400To600_TuneCP5_13TeV-madgraphMLM-pythia8/RunIISummer20UL16NanoAODAPVv9-106X_mcRun2_asymptotic_preVFP_v11-v1/NANOAODSIM")
WJetsHT4002016    = sample("WJetsHT4002016", 48.91*1.21, "2016", "WJetsHT4002016ULNanoList.txt", "/WJetsToLNu_HT-400To600_TuneCP5_13TeV-madgraphMLM-pythia8/RunIISummer20UL16NanoAODv9-106X_mcRun2_asymptotic_v17-v1/NANOAODSIM")
WJetsHT4002017    = sample("WJetsHT4002017", 48.91*1.21, "2017", "WJetsHT4002017ULNanoList.txt", "/WJetsToLNu_HT-400To600_TuneCP5_13TeV-madgraphMLM-pythia8/RunIISummer20UL17NanoAODv9-106X_mc2017_realistic_v9-v1/NANOAODSIM")
WJetsHT4002018    = sample("WJetsHT4002018", 48.91*1.21, "2018", "WJetsHT4002018ULNanoList.txt", "/WJetsToLNu_HT-400To600_TuneCP5_13TeV-madgraphMLM-pythia8/RunIISummer20UL18NanoAODv9-106X_upgrade2018_realistic_v16_L1v1-v1/NANOAODSIM")
WJetsHT6002016APV = sample("WJetsHT6002016APV", 12.05*1.21, "2016APV", "WJetsHT6002016APVULNanoList.txt", "/WJetsToLNu_HT-600To800_TuneCP5_13TeV-madgraphMLM-pythia8/RunIISummer20UL16NanoAODAPVv9-106X_mcRun2_asymptotic_preVFP_v11-v1/NANOAODSIM")
WJetsHT6002016    = sample("WJetsHT6002016", 12.05*1.21, "2016", "WJetsHT6002016ULNanoList.txt", "/WJetsToLNu_HT-600To800_TuneCP5_13TeV-madgraphMLM-pythia8/RunIISummer20UL16NanoAODv9-106X_mcRun2_asymptotic_v17-v1/NANOAODSIM")
WJetsHT6002017    = sample("WJetsHT6002017", 12.05*1.21, "2017", "WJetsHT6002017ULNanoList.txt", "/WJetsToLNu_HT-600To800_TuneCP5_13TeV-madgraphMLM-pythia8/RunIISummer20UL17NanoAODv9-106X_mc2017_realistic_v9-v1/NANOAODSIM")
WJetsHT6002018    = sample("WJetsHT6002018", 12.05*1.21, "2018", "WJetsHT6002018ULNanoList.txt", "/WJetsToLNu_HT-600To800_TuneCP5_13TeV-madgraphMLM-pythia8/RunIISummer20UL18NanoAODv9-106X_upgrade2018_realistic_v16_L1v1-v1/NANOAODSIM")
WJetsHT8002016APV = sample("WJetsHT8002016APV", 5.501*1.21, "2016APV", "WJetsHT8002016APVULNanoList.txt", "/WJetsToLNu_HT-800To1200_TuneCP5_13TeV-madgraphMLM-pythia8/RunIISummer20UL16NanoAODAPVv9-106X_mcRun2_asymptotic_preVFP_v11-v1/NANOAODSIM")
WJetsHT8002016    = sample("WJetsHT8002016", 5.501*1.21, "2016", "WJetsHT8002016ULNanoList.txt", "/WJetsToLNu_HT-800To1200_TuneCP5_13TeV-madgraphMLM-pythia8/RunIISummer20UL16NanoAODv9-106X_mcRun2_asymptotic_v17-v1/NANOAODSIM")
WJetsHT8002017    = sample("WJetsHT8002017", 5.501*1.21, "2017", "WJetsHT8002017ULNanoList.txt", "/WJetsToLNu_HT-800To1200_TuneCP5_13TeV-madgraphMLM-pythia8/RunIISummer20UL17NanoAODv9-106X_mc2017_realistic_v9-v1/NANOAODSIM")
WJetsHT8002018    = sample("WJetsHT8002018", 5.501*1.21, "2018", "WJetsHT8002018ULNanoList.txt", "/WJetsToLNu_HT-800To1200_TuneCP5_13TeV-madgraphMLM-pythia8/RunIISummer20UL18NanoAODv9-106X_upgrade2018_realistic_v16_L1v1-v1/NANOAODSIM")
WW2016APV         = sample("WW2016APV", 118.7, "2016APV", "WW2016APVULNanoList.txt", "/WW_TuneCP5_13TeV-pythia8/RunIISummer20UL16NanoAODAPVv9-106X_mcRun2_asymptotic_preVFP_v11-v1/NANOAODSIM")
WW2016            = sample("WW2016", 118.7, "2016", "WW2016ULNanoList.txt", "/WW_TuneCP5_13TeV-pythia8/RunIISummer20UL16NanoAODv9-106X_mcRun2_asymptotic_v17-v1/NANOAODSIM")
WW2017            = sample("WW2017", 118.7, "2017", "WW2017ULNanoList.txt", "/WW_TuneCP5_13TeV-pythia8/RunIISummer20UL17NanoAODv9-106X_mc2017_realistic_v9-v1/NANOAODSIM")
WW2018            = sample("WW2018", 118.7, "2018", "WW2018ULNanoList.txt", "/WW_TuneCP5_13TeV-pythia8/RunIISummer20UL18NanoAODv9-106X_upgrade2018_realistic_v16_L1v1-v1/NANOAODSIM")
WZ2016APV         = sample("WZ2016APV", 47.13, "2016APV", "WZ2016APVULNanoList.txt", "/WZ_TuneCP5_13TeV-pythia8/RunIISummer20UL16NanoAODAPVv9-106X_mcRun2_asymptotic_preVFP_v11-v1/NANOAODSIM")
WZ2016            = sample("WZ2016", 47.13, "2016", "WZ2016ULNanoList.txt", "/WZ_TuneCP5_13TeV-pythia8/RunIISummer20UL16NanoAODv9-106X_mcRun2_asymptotic_v17-v1/NANOAODSIM")
WZ2017            = sample("WZ2017", 47.13, "2017", "WZ2017ULNanoList.txt", "/WZ_TuneCP5_13TeV-pythia8/RunIISummer20UL17NanoAODv9-106X_mc2017_realistic_v9-v1/NANOAODSIM")
WZ2018            = sample("WZ2018", 47.13, "2018", "WZ2018ULNanoList.txt", "/WZ_TuneCP5_13TeV-pythia8/RunIISummer20UL18NanoAODv9-106X_upgrade2018_realistic_v16_L1v1-v1/NANOAODSIM")
ZZ2016APV         = sample("ZZ2016APV", 16.523, "2016APV", "ZZ2016APVULNanoList.txt", "/ZZ_TuneCP5_13TeV-pythia8/RunIISummer20UL16NanoAODAPVv9-106X_mcRun2_asymptotic_preVFP_v11-v1/NANOAODSIM")
ZZ2016            = sample("ZZ2016", 16.523, "2016", "ZZ2016ULNanoList.txt", "/ZZ_TuneCP5_13TeV-pythia8/RunIISummer20UL16NanoAODv9-106X_mcRun2_asymptotic_v17-v1/NANOAODSIM")
ZZ2017            = sample("ZZ2017", 16.523, "2017", "ZZ2017ULNanoList.txt", "/ZZ_TuneCP5_13TeV-pythia8/RunIISummer20UL17NanoAODv9-106X_mc2017_realistic_v9-v1/NANOAODSIM")
ZZ2018            = sample("ZZ2018", 16.523, "2018", "ZZ2018ULNanoList.txt", "/ZZ_TuneCP5_13TeV-pythia8/RunIISummer20UL18NanoAODv9-106X_upgrade2018_realistic_v16_L1v1-v1/NANOAODSIM")

## Number of processed events (sum of weights)
Bprime_M1000_2016APV.nrun = 500000.0 # from integral 500000, file Bprime_M1000_2016APV
Bprime_M1200_2016APV.nrun = 498158.0 # from integral 498158, file Bprime_M1200_2016APV
Bprime_M1300_2016APV.nrun = 499079.0 # from integral 499079, file Bprime_M1300_2016APV
Bprime_M1400_2016APV.nrun = 500000.0 # from integral 500000, file Bprime_M1400_2016APV
Bprime_M1500_2016APV.nrun = 497234.0 # from integral 497234, file Bprime_M1500_2016APV
Bprime_M1600_2016APV.nrun = 500000.0 # from integral 500000, file Bprime_M1600_2016APV
Bprime_M1700_2016APV.nrun = 500000.0 # from integral 500000, file Bprime_M1700_2016APV
Bprime_M1800_2016APV.nrun = 499078.0 # from integral 499078, file Bprime_M1800_2016APV
Bprime_M2000_2016APV.nrun = 497234.0 # from integral 497234, file Bprime_M2000_2016APV
Bprime_M2200_2016APV.nrun = 500000.0 # from integral 500000, file Bprime_M2200_2016APV
Bprime_M800_2016APV.nrun = 500000.0 # from integral 500000, file Bprime_M800_2016APV
DYMHT12002016APV.nrun = 2189664.0 # from integral 2189664, file DYMHT12002016APV
DYMHT2002016APV.nrun = 5862631.0 # from integral 5862631, file DYMHT2002016APV
DYMHT25002016APV.nrun = 721404.0 # from integral 721404, file DYMHT25002016APV
DYMHT4002016APV.nrun = 2716892.0 # from integral 2716892, file DYMHT4002016APV
DYMHT6002016APV.nrun = 2681650.0 # from integral 2681650, file DYMHT6002016APV
DYMHT8002016APV.nrun = 2411091.0 # from integral 2411091, file DYMHT8002016APV
QCDHT10002016APV.nrun = 4773503.0 # from integral 4773503, file QCDHT10002016APV
QCDHT15002016APV.nrun = 3503675.0 # from integral 3503675, file QCDHT15002016APV
QCDHT20002016APV.nrun = 1629000.0 # from integral 1629000, file QCDHT20002016APV
QCDHT2002016APV.nrun = 16524587.0 # from integral 16524587, file QCDHT2002016APV
QCDHT3002016APV.nrun = 15341307.0 # from integral 15341307, file QCDHT3002016APV
QCDHT5002016APV.nrun = 15775001.0 # from integral 15775001, file QCDHT5002016APV
QCDHT7002016APV.nrun = 15808790.0 # from integral 15808790, file QCDHT7002016APV
STs2016APV.nrun = 3592771.999926734 # from integral 5518000, file STs2016APV
STt2016APV.nrun = 52437431.99999999 # from integral 55961000, file STt2016APV
STtb2016APV.nrun = 29205918.0 # from integral 31024000, file STtb2016APV
STtW2016APV.nrun = 2299880.0 # from integral 2300000, file STtW2016APV
STtWb2016APV.nrun = 2299866.000000002 # from integral 2300000, file STtWb2016APV
TTHB2016APV.nrun = 4525710.0 # from integral 4622000, file TTHB2016APV
TTHnonB2016APV.nrun = 1936276.0 # from integral 1977996, file TTHnonB2016APV
TTTo2L2Nu2016APV.nrun = 37202074.0 # from integral 37505000, file TTTo2L2Nu2016APV
TTToHadronic2016APV.nrun = 96474574.0 # from integral 97260000, file TTToHadronic2016APV
TTToSemiLeptonic2016APV.nrun = 131106830.0 # from integral 132178000, file TTToSemiLeptonic2016APV
TTMT10002016APV.nrun = 22004252.0 + 0.02474*(TTToHadronic2016APV.nrun+TTToSemiLeptonic2016APV.nrun+TTTo2L2Nu2016APV.nrun)# from integral 23208356, file TTMT10002016APV
TTMT7002016APV.nrun = 23342259.0 + 0.0921*(TTToHadronic2016APV.nrun+TTToSemiLeptonic2016APV.nrun+TTTo2L2Nu2016APV.nrun)# from integral 23785283, file TTMT7002016APV
TTTo2L2Nu2016APV0.nrun = TTTo2L2Nu2016APV.nrun*0.8832
TTToHadronic2016APV0.nrun = TTToHadronic2016APV.nrun*0.8832
TTToSemiLeptonic2016APV0.nrun = TTToSemiLeptonic2016APV.nrun*0.8832
TTTo2L2Nu2016APV700.nrun = TTTo2L2Nu2016APV.nrun*0.0921 + TTMT7002016APV.nrun*0.105
TTToHadronic2016APV700.nrun = TTToHadronic2016APV.nrun*0.0921 + TTMT7002016APV.nrun*0.457
TTToSemiLeptonic2016APV700.nrun = TTToSemiLeptonic2016APV.nrun*0.0921 + TTMT7002016APV.nrun*0.438
TTTo2L2Nu2016APV1000.nrun = TTTo2L2Nu2016APV.nrun*0.02474 + TTMT10002016APV.nrun*0.105
TTToHadronic2016APV1000.nrun = TTToHadronic2016APV.nrun*0.02474 + TTMT10002016APV.nrun*0.457
TTToSemiLeptonic2016APV1000.nrun = TTToSemiLeptonic2016APV.nrun*0.02474 + TTMT10002016APV.nrun*0.438
TTWl2016APV.nrun = 1543290.0 # from integral 2850164, file TTWl2016APV
TTWq2016APV.nrun = 148842.00000000003 # from integral 271496, file TTWq2016APV
TTZM102016APV.nrun = 2856626.0000000014 # from integral 5792000, file TTZM102016APV
TTZM1to102016APV.nrun = 177656.00000000003 # from integral 320000, file TTZM1to102016APV
WJetsHT12002016APV.nrun = 2119975.0 # from integral 2119975, file WJetsHT12002016APV
WJetsHT2002016APV.nrun = 17870845.0 # from integral 17870845, file WJetsHT2002016APV
WJetsHT25002016APV.nrun = 808649.0 # from integral 808649, file WJetsHT25002016APV
WJetsHT4002016APV.nrun = 2467498.0 # from integral 2467498, file WJetsHT4002016APV
WJetsHT6002016APV.nrun = 2344130.0 # from integral 2344130, file WJetsHT6002016APV
WJetsHT8002016APV.nrun = 2510487.0 # from integral 2510487, file WJetsHT8002016APV
WW2016APV.nrun = 15858980.29168732 # from integral 15859000, file WW2016APV
WZ2016APV.nrun = 7934000.0 # from integral 7934000, file WZ2016APV
ZZ2016APV.nrun = 1282000.0 # from integral 1282000, file ZZ2016APV

Bprime_M1000_2016.nrun = 500000.0 # from integral 500000, file Bprime_M1000_2016
Bprime_M1200_2016.nrun = 500000.0 # from integral 500000, file Bprime_M1200_2016
Bprime_M1300_2016.nrun = 500000.0 # from integral 500000, file Bprime_M1300_2016
Bprime_M1400_2016.nrun = 500000.0 # from integral 500000, file Bprime_M1400_2016
Bprime_M1500_2016.nrun = 500000.0 # from integral 500000, file Bprime_M1500_2016
Bprime_M1600_2016.nrun = 500000.0 # from integral 500000, file Bprime_M1600_2016
Bprime_M1700_2016.nrun = 500000.0 # from integral 500000, file Bprime_M1700_2016
Bprime_M1800_2016.nrun = 498140.0 # from integral 498140, file Bprime_M1800_2016
Bprime_M2000_2016.nrun = 500000.0 # from integral 500000, file Bprime_M2000_2016
Bprime_M2200_2016.nrun = 500000.0 # from integral 500000, file Bprime_M2200_2016
Bprime_M800_2016.nrun = 499071.0 # from integral 499071, file Bprime_M800_2016
DYMHT12002016.nrun = 1970857.0 # from integral 1970857, file DYMHT12002016
DYMHT2002016.nrun = 5653782.0 # from integral 5653782, file DYMHT2002016
DYMHT25002016.nrun = 696811.0 # from integral 696811, file DYMHT25002016
DYMHT4002016.nrun = 2491416.0 # from integral 2491416, file DYMHT4002016
DYMHT6002016.nrun = 2299853.0 # from integral 2299853, file DYMHT6002016
DYMHT8002016.nrun = 2393976.0 # from integral 2393976, file DYMHT8002016
QCDHT10002016.nrun = 4365993.0 # from integral 4365993, file QCDHT10002016
QCDHT15002016.nrun = 3003707.0 # from integral 3003707, file QCDHT15002016
QCDHT20002016.nrun = 1847781.0 # from integral 1847781, file QCDHT20002016
QCDHT2002016.nrun = 17569141.0 # from integral 17569141, file QCDHT2002016
QCDHT3002016.nrun = 16747056.0 # from integral 16747056, file QCDHT3002016
QCDHT5002016.nrun = 14212819.0 # from integral 14212819, file QCDHT5002016
QCDHT7002016.nrun = 13750537.0 # from integral 13750537, file QCDHT7002016
STs2016.nrun = 3562865.999989305 # from integral 5471000, file STs2016
STt2016.nrun = 59099221.99999999 # from integral 63073000, file STt2016
STtb2016.nrun = 28814596.0 # from integral 30609000, file STtb2016
STtW2016.nrun = 2490860.000000001 # from integral 2491000, file STtW2016
STtWb2016.nrun = 2553882.0000000005 # from integral 2554000, file STtWb2016
TTHB2016.nrun = 4834712.0 # from integral 4937000, file TTHB2016
TTHnonB2016.nrun = 2194702.0 # from integral 2240994, file TTHnonB2016
TTTo2L2Nu2016.nrun = 43193956.00000001 # from integral 43546000, file TTTo2L2Nu2016
TTToHadronic2016.nrun = 106200414.0 # from integral 107067000, file TTToHadronic2016
TTToSemiLeptonic2016.nrun = 143553998.00000003 # from integral 144722000, file TTToSemiLeptonic2016
TTMT10002016.nrun = 22444321.999999996 + 0.02474*(TTToHadronic2016.nrun+TTToSemiLeptonic2016.nrun+TTTo2L2Nu2016.nrun)# from integral 23673116, file TTMT10002016
TTMT7002016.nrun = 32878794.999999996 + 0.0921*(TTToHadronic2016.nrun+TTToSemiLeptonic2016.nrun+TTTo2L2Nu2016.nrun) # from integral 33502717, file TTMT7002016
TTTo2L2Nu20160.nrun = TTTo2L2Nu2016.nrun*0.8832
TTToHadronic20160.nrun = TTToHadronic2016.nrun*0.8832
TTToSemiLeptonic20160.nrun = TTToSemiLeptonic2016.nrun*0.8832
TTTo2L2Nu2016700.nrun = TTTo2L2Nu2016.nrun*0.0921 + TTMT7002016.nrun*0.105
TTToHadronic2016700.nrun = TTToHadronic2016.nrun*0.0921 + TTMT7002016.nrun*0.457
TTToSemiLeptonic2016700.nrun = TTToSemiLeptonic2016.nrun*0.0921 + TTMT7002016.nrun*0.438
TTTo2L2Nu20161000.nrun = TTTo2L2Nu2016.nrun*0.02474 + TTMT10002016.nrun*0.105
TTToHadronic20161000.nrun = TTToHadronic2016.nrun*0.02474 + TTMT10002016.nrun*0.457
TTToSemiLeptonic20161000.nrun = TTToSemiLeptonic2016.nrun*0.02474 + TTMT10002016.nrun*0.438
TTWl2016.nrun = 1800822.999999999 # from integral 3322643, file TTWl2016
TTWq2016.nrun = 168951.00000000003 # from integral 308983, file TTWq2016
TTZM102016.nrun = 2962856.0000000005 # from integral 6017000, file TTZM102016
TTZM1to102016.nrun = 177656.00000000003 # from integral 320000, file TTZM1to102016
WJetsHT12002016.nrun = 2090561.0 # from integral 2090561, file WJetsHT12002016
WJetsHT2002016.nrun = 15067621.0 # from integral 15067621, file WJetsHT2002016
WJetsHT25002016.nrun = 709514.0 # from integral 709514, file WJetsHT25002016
WJetsHT4002016.nrun = 2115509.0 # from integral 2115509, file WJetsHT4002016
WJetsHT6002016.nrun = 2251807.0 # from integral 2251807, file WJetsHT6002016
WJetsHT8002016.nrun = 2132228.0 # from integral 2132228, file WJetsHT8002016
WW2016.nrun = 15820978.251753712 # from integral 15821000, file WW2016
WZ2016.nrun = 7584000.0 # from integral 7584000, file WZ2016
ZZ2016.nrun = 1151000.0 # from integral 1151000, file ZZ2016

Bprime_M1000_2017.nrun = 1000000.0 # from integral 1000000, file Bprime_M1000_2017
Bprime_M1200_2017.nrun = 1000000.0 # from integral 1000000, file Bprime_M1200_2017
Bprime_M1300_2017.nrun = 1000000.0 # from integral 1000000, file Bprime_M1300_2017
Bprime_M1400_2017.nrun = 1000000.0 # from integral 1000000, file Bprime_M1400_2017
Bprime_M1500_2017.nrun = 1000000.0 # from integral 1000000, file Bprime_M1500_2017
Bprime_M1600_2017.nrun = 1000000.0 # from integral 1000000, file Bprime_M1600_2017
Bprime_M1700_2017.nrun = 1000000.0 # from integral 1000000, file Bprime_M1700_2017
Bprime_M1800_2017.nrun = 1000000.0 # from integral 1000000, file Bprime_M1800_2017
Bprime_M2000_2017.nrun = 1000000.0 # from integral 1000000, file Bprime_M2000_2017
Bprime_M2200_2017.nrun = 999000.0 # from integral 999000, file Bprime_M2200_2017
Bprime_M800_2017.nrun = 1000000.0 # from integral 1000000, file Bprime_M800_2017
DYMHT12002017.nrun = 4725936.0 # from integral 4725936, file DYMHT12002017
DYMHT2002017.nrun = 12451701.0 # from integral 12451701, file DYMHT2002017
DYMHT25002017.nrun = 1480047.0 # from integral 1480047, file DYMHT25002017
DYMHT4002017.nrun = 5384252.0 # from integral 5384252, file DYMHT4002017
DYMHT6002017.nrun = 5118706.0 # from integral 5118706, file DYMHT6002017
DYMHT8002017.nrun = 4347168.0 # from integral 4347168, file DYMHT8002017
QCDHT10002017.nrun = 10186734.0 # from integral 10186734, file QCDHT10002017
QCDHT15002017.nrun = 7701876.0 # from integral 7701876, file QCDHT15002017
QCDHT20002017.nrun = 4112573.0 # from integral 4112573, file QCDHT20002017
QCDHT2002017.nrun = 42714435.0 # from integral 42714435, file QCDHT2002017
QCDHT3002017.nrun = 43429979.0 # from integral 43429979, file QCDHT3002017
QCDHT5002017.nrun = 36194860.0 # from integral 36194860, file QCDHT5002017
QCDHT7002017.nrun = 32934816.0 # from integral 32934816, file QCDHT7002017
STs2017.nrun = 8866569.999971403 # from integral 13620000, file STs2017
STt2017.nrun = 121728252.0 # from integral 129903000, file STt2017
STtb2017.nrun = 65701154.0 # from integral 69793000, file STtb2017
STtW2017.nrun = 5648712.0 # from integral 5649000, file STtW2017
STtWb2017.nrun = 5673700.000000001 # from integral 5674000, file STtWb2017
TTHB2017.nrun = 7661778.0 # from integral 7825000, file TTHB2017
TTHnonB2017.nrun = 4965389.0 # from integral 5070989, file TTHnonB2017
TTTo2L2Nu2017.nrun = 105859990.0 # from integral 106724000, file TTTo2L2Nu2017
TTToHadronic2017.nrun = 231117295.00000003 # from integral 232999999, file TTToHadronic2017
TTToSemiLeptonic2017.nrun = 343257666.0 # from integral 346052000, file TTToSemiLeptonic2017
TTMT10002017.nrun = 21545730.0 + 0.02474*(TTToHadronic2017.nrun+TTToSemiLeptonic2017.nrun+TTTo2L2Nu2017.nrun) # from integral 22724532, file TTMT10002017
TTMT7002017.nrun = 35196862.0 + 0.0921*(TTToHadronic2017.nrun+TTToSemiLeptonic2017.nrun+TTTo2L2Nu2017.nrun) # from integral 35862238, file TTMT7002017
TTTo2L2Nu20170.nrun = TTTo2L2Nu2017.nrun*0.8832
TTToHadronic20170.nrun = TTToHadronic2017.nrun*0.8832
TTToSemiLeptonic20170.nrun = TTToSemiLeptonic2017.nrun*0.8832
TTTo2L2Nu2017700.nrun = TTTo2L2Nu2017.nrun*0.0921 + TTMT7002017.nrun*0.105
TTToHadronic2017700.nrun = TTToHadronic2017.nrun*0.0921 + TTMT7002017.nrun*0.457
TTToSemiLeptonic2017700.nrun = TTToSemiLeptonic2017.nrun*0.0921 + TTMT7002017.nrun*0.438
TTTo2L2Nu20171000.nrun = TTTo2L2Nu2017.nrun*0.02474 + TTMT10002017.nrun*0.105
TTToHadronic20171000.nrun = TTToHadronic2017.nrun*0.02474 + TTMT10002017.nrun*0.457
TTToSemiLeptonic20171000.nrun = TTToSemiLeptonic2017.nrun*0.02474 + TTMT10002017.nrun*0.438
TTWl2017.nrun = 3871054.9999999963 # from integral 7140411, file TTWl2017
TTWq2017.nrun = 359006.00000000006 # from integral 655018, file TTWq2017
TTZM102017.nrun = 6911466.000000001 # from integral 14036000, file TTZM102017
TTZM1to102017.nrun = 390602.00000000006 # from integral 707000, file TTZM1to102017
WJetsHT12002017.nrun = 4752118.0 # from integral 4752118, file WJetsHT12002017
WJetsHT2002017.nrun = 42281979.0 # from integral 42281979, file WJetsHT2002017
WJetsHT25002017.nrun = 1185699.0 # from integral 1185699, file WJetsHT25002017
WJetsHT4002017.nrun = 5468473.0 # from integral 5468473, file WJetsHT4002017
WJetsHT6002017.nrun = 5545298.0 # from integral 5545298, file WJetsHT6002017
WJetsHT8002017.nrun = 5088483.0 # from integral 5088483, file WJetsHT8002017
WW2017.nrun = 15633982.548347734 # from integral 15634000, file WW2017
WZ2017.nrun = 7889000.0 # from integral 7889000, file WZ2017
ZZ2017.nrun = 2706000.0 # from integral 2706000, file ZZ2017

Bprime_M1000_2018.nrun = 1000000.0 # from integral 1000000, file Bprime_M1000_2018
Bprime_M1200_2018.nrun = 1000000.0 # from integral 1000000, file Bprime_M1200_2018
Bprime_M1300_2018.nrun = 1000000.0 # from integral 1000000, file Bprime_M1300_2018
Bprime_M1400_2018.nrun = 1000000.0 # from integral 1000000, file Bprime_M1400_2018
Bprime_M1500_2018.nrun = 1000000.0 # from integral 1000000, file Bprime_M1500_2018
Bprime_M1600_2018.nrun = 999000.0 # from integral 999000, file Bprime_M1600_2018
Bprime_M1700_2018.nrun = 1000000.0 # from integral 1000000, file Bprime_M1700_2018
Bprime_M1800_2018.nrun = 1000000.0 # from integral 1000000, file Bprime_M1800_2018
Bprime_M2000_2018.nrun = 998000.0 # from integral 998000, file Bprime_M2000_2018
Bprime_M2200_2018.nrun = 1000000.0 # from integral 1000000, file Bprime_M2200_2018
Bprime_M800_2018.nrun = 1000000.0 # from integral 1000000, file Bprime_M800_2018
DYMHT12002018.nrun = 5966661.0 # from integral 5966661, file DYMHT12002018
DYMHT2002018.nrun = 18455718.0 # from integral 18455718, file DYMHT2002018
DYMHT25002018.nrun = 1978203.0 # from integral 1978203, file DYMHT25002018
DYMHT4002018.nrun = 8682257.0 # from integral 8682257, file DYMHT4002018
DYMHT6002018.nrun = 7035971.0 # from integral 7035971, file DYMHT6002018
DYMHT8002018.nrun = 6554679.0 # from integral 6554679, file DYMHT8002018
QCDHT10002018.nrun = 14394786.0 # from integral 14394786, file QCDHT10002018
QCDHT15002018.nrun = 10411831.0 # from integral 10411831, file QCDHT15002018
QCDHT20002018.nrun = 5374711.0 # from integral 5374711, file QCDHT20002018
QCDHT2002018.nrun = 57336623.0 # from integral 57336623, file QCDHT2002018
QCDHT3002018.nrun = 61609663.0 # from integral 61609663, file QCDHT3002018
QCDHT5002018.nrun = 49184771.0 # from integral 49184771, file QCDHT5002018
QCDHT7002018.nrun = 48506751.0 # from integral 48506751, file QCDHT7002018
STs2018.nrun = 12607740.99998877 # from integral 19365999, file STs2018
STt2018.nrun = 167111717.99999997 # from integral 178336000, file STt2018
STtb2018.nrun = 90022642.0 # from integral 95627000, file STtb2018
STtW2018.nrun = 7955614.0 # from integral 7956000, file STtW2018
STtWb2018.nrun = 7748690.000000002 # from integral 7749000, file STtWb2018
TTHB2018.nrun = 9467226.0 # from integral 9668000, file TTHB2018
TTHnonB2018.nrun = 7176599.0 # from integral 7328993, file TTHnonB2018
TTTo2L2Nu2018.nrun = 143848848.00000003 # from integral 145020000, file TTTo2L2Nu2018
TTToHadronic2018.nrun = 331506194.0 # from integral 334206000, file TTToHadronic2018
TTToSemiLeptonic2018.nrun = 471248534.0 # from integral 475088000, file TTToSemiLeptonic2018
TTMT10002018.nrun = 22396890.0 + 0.02474*(TTToHadronic2018.nrun+TTToSemiLeptonic2018.nrun+TTTo2L2Nu2018.nrun) # from integral 23624506, file TTMT10002018
TTMT7002018.nrun = 30084128.0 + 0.0921*(TTToHadronic2018.nrun+TTToSemiLeptonic2018.nrun+TTTo2L2Nu2018.nrun) # from integral 30653714, file TTMT7002018
TTTo2L2Nu20180.nrun = TTTo2L2Nu2018.nrun*0.8832
TTToHadronic20180.nrun = TTToHadronic2018.nrun*0.8832
TTToSemiLeptonic20180.nrun = TTToSemiLeptonic2018.nrun*0.8832
TTTo2L2Nu2018700.nrun = TTTo2L2Nu2018.nrun*0.0921 + TTMT7002018.nrun*0.105
TTToHadronic2018700.nrun = TTToHadronic2018.nrun*0.0921 + TTMT7002018.nrun*0.457
TTToSemiLeptonic2018700.nrun = TTToSemiLeptonic2018.nrun*0.0921 + TTMT7002018.nrun*0.438
TTTo2L2Nu20181000.nrun = TTTo2L2Nu2018.nrun*0.02474 + TTMT10002018.nrun*0.105
TTToHadronic20181000.nrun = TTToHadronic2018.nrun*0.02474 + TTMT10002018.nrun*0.457
TTToSemiLeptonic20181000.nrun = TTToSemiLeptonic2018.nrun*0.02474 + TTMT10002018.nrun*0.438
TTWl2018.nrun = 5666427.999999999 # from integral 10450000, file TTWl2018
TTWq2018.nrun = 530327.0000000001 # from integral 970179, file TTWq2018
TTZM102018.nrun = 9651834.000000002 # from integral 19608000, file TTZM102018
TTZM1to102018.nrun = 550706.0 # from integral 994000, file TTZM1to102018
WJetsHT12002018.nrun = 6481518.0 # from integral 6481518, file WJetsHT12002018
WJetsHT2002018.nrun = 58225632.0 # from integral 58225632, file WJetsHT2002018
WJetsHT25002018.nrun = 2097648.0 # from integral 2097648, file WJetsHT25002018
WJetsHT4002018.nrun = 7444030.0 # from integral 7444030, file WJetsHT4002018
WJetsHT6002018.nrun = 7718765.0 # from integral 7718765, file WJetsHT6002018
WJetsHT8002018.nrun = 7306187.0 # from integral 7306187, file WJetsHT8002018
WW2018.nrun = 15678982.071080217 # from integral 15679000, file WW2018
WZ2018.nrun = 7940000.0 # from integral 7940000, file WZ2018
ZZ2018.nrun = 3526000.0 # from integral 3526000, file ZZ2018

samples_signal={
    "Bprime_M800_2016APV":Bprime_M800_2016APV,
    "Bprime_M1000_2016APV":Bprime_M1000_2016APV,
    "Bprime_M1200_2016APV":Bprime_M1200_2016APV,
    "Bprime_M1300_2016APV":Bprime_M1300_2016APV,
    "Bprime_M1400_2016APV":Bprime_M1400_2016APV,
    "Bprime_M1500_2016APV":Bprime_M1500_2016APV,
    "Bprime_M1600_2016APV":Bprime_M1600_2016APV,
    "Bprime_M1700_2016APV":Bprime_M1700_2016APV,
    "Bprime_M1800_2016APV":Bprime_M1800_2016APV,
    "Bprime_M2000_2016APV":Bprime_M2000_2016APV,
    "Bprime_M2200_2016APV":Bprime_M2200_2016APV,
    "Bprime_M1000_2016":Bprime_M1000_2016,
    "Bprime_M1000_2017":Bprime_M1000_2017,
    "Bprime_M1000_2018":Bprime_M1000_2018,
    "Bprime_M1200_2016":Bprime_M1200_2016,
    "Bprime_M1200_2017":Bprime_M1200_2017,
    "Bprime_M1200_2018":Bprime_M1200_2018,
    "Bprime_M1300_2016":Bprime_M1300_2016,
    "Bprime_M1300_2017":Bprime_M1300_2017,
    "Bprime_M1300_2018":Bprime_M1300_2018,
    "Bprime_M1400_2016":Bprime_M1400_2016,
    "Bprime_M1400_2017":Bprime_M1400_2017,
    "Bprime_M1400_2018":Bprime_M1400_2018,
    "Bprime_M1500_2016":Bprime_M1500_2016,
    "Bprime_M1500_2017":Bprime_M1500_2017,
    "Bprime_M1500_2018":Bprime_M1500_2018,
    "Bprime_M1600_2016":Bprime_M1600_2016,
    "Bprime_M1600_2017":Bprime_M1600_2017,
    "Bprime_M1600_2018":Bprime_M1600_2018,
    "Bprime_M1700_2016":Bprime_M1700_2016,
    "Bprime_M1700_2017":Bprime_M1700_2017,
    "Bprime_M1700_2018":Bprime_M1700_2018,
    "Bprime_M1800_2016":Bprime_M1800_2016,
    "Bprime_M1800_2017":Bprime_M1800_2017,
    "Bprime_M1800_2018":Bprime_M1800_2018,
    "Bprime_M2000_2016":Bprime_M2000_2016,
    "Bprime_M2000_2017":Bprime_M2000_2017,
    "Bprime_M2000_2018":Bprime_M2000_2018,
    "Bprime_M2200_2016":Bprime_M2200_2016,
    "Bprime_M2200_2017":Bprime_M2200_2017,
    "Bprime_M2200_2018":Bprime_M2200_2018,
    "Bprime_M800_2016":Bprime_M800_2016,
    "Bprime_M800_2017":Bprime_M800_2017,
    "Bprime_M800_2018":Bprime_M800_2018,
}
samples_electroweak = {
    "DYMHT12002016APV":DYMHT12002016APV,
    "DYMHT12002016":DYMHT12002016,
    "DYMHT12002017":DYMHT12002017,
    "DYMHT12002018":DYMHT12002018,
    "DYMHT2002016APV":DYMHT2002016APV,
    "DYMHT2002016":DYMHT2002016,
    "DYMHT2002017":DYMHT2002017,
    "DYMHT2002018":DYMHT2002018,
    "DYMHT25002016APV":DYMHT25002016APV,
    "DYMHT25002016":DYMHT25002016,
    "DYMHT25002017":DYMHT25002017,
    "DYMHT25002018":DYMHT25002018,
    "DYMHT4002016APV":DYMHT4002016APV,
    "DYMHT4002016":DYMHT4002016,
    "DYMHT4002017":DYMHT4002017,
    "DYMHT4002018":DYMHT4002018,
    "DYMHT6002016APV":DYMHT6002016APV,
    "DYMHT6002016":DYMHT6002016,
    "DYMHT6002017":DYMHT6002017,
    "DYMHT6002018":DYMHT6002018,
    "DYMHT8002016APV":DYMHT8002016APV,
    "DYMHT8002016":DYMHT8002016,
    "DYMHT8002017":DYMHT8002017,
    "DYMHT8002018":DYMHT8002018,
    "WW2016APV":WW2016APV,
    "WW2016":WW2016,
    "WW2017":WW2017,
    "WW2018":WW2018,
    "WZ2016APV":WZ2016APV,
    "WZ2016":WZ2016,
    "WZ2017":WZ2017,
    "WZ2018":WZ2018,
    "ZZ2016APV":ZZ2016APV,
    "ZZ2016":ZZ2016,
    "ZZ2017":ZZ2017,
    "ZZ2018":ZZ2018,
}
for isamp in samples_electroweak.keys():
    if 'DYMHT2500' in isamp: samples_electroweak[isamp].kfactor = 0.617254
    elif 'DYMHT1200' in isamp: samples_electroweak[isamp].kfactor = 0.749894
    elif 'DYMHT800' in isamp: samples_electroweak[isamp].kfactor = 0.883340
    elif 'DYMHT600' in isamp: samples_electroweak[isamp].kfactor = 0.948367
    elif 'DYMHT400' in isamp: samples_electroweak[isamp].kfactor = 0.974071
    elif 'DYMHT200' in isamp: samples_electroweak[isamp].kfactor = 0.992853

samples_wjets = {
    "WJetsHT12002016APV":WJetsHT12002016APV,
    "WJetsHT12002016":WJetsHT12002016,
    "WJetsHT12002017":WJetsHT12002017,
    "WJetsHT12002018":WJetsHT12002018,
    "WJetsHT2002016APV":WJetsHT2002016APV,
    "WJetsHT2002016":WJetsHT2002016,
    "WJetsHT2002017":WJetsHT2002017,
    "WJetsHT2002018":WJetsHT2002018,
    "WJetsHT25002016APV":WJetsHT25002016APV,
    "WJetsHT25002016":WJetsHT25002016,
    "WJetsHT25002017":WJetsHT25002017,
    "WJetsHT25002018":WJetsHT25002018,
    "WJetsHT4002016APV":WJetsHT4002016APV,
    "WJetsHT4002016":WJetsHT4002016,
    "WJetsHT4002017":WJetsHT4002017,
    "WJetsHT4002018":WJetsHT4002018,
    "WJetsHT6002016APV":WJetsHT6002016APV,
    "WJetsHT6002016":WJetsHT6002016,
    "WJetsHT6002017":WJetsHT6002017,
    "WJetsHT6002018":WJetsHT6002018,
    "WJetsHT8002016APV":WJetsHT8002016APV,
    "WJetsHT8002016":WJetsHT8002016,
    "WJetsHT8002017":WJetsHT8002017,
    "WJetsHT8002018":WJetsHT8002018,
}

for isamp in samples_wjets.keys():
    if 'WJetsHT2500' in isamp: samples_wjets[isamp].kfactor = 0.454246
    elif 'WJetsHT1200' in isamp: samples_wjets[isamp].kfactor = 0.608292
    elif 'WJetsHT800' in isamp: samples_wjets[isamp].kfactor = 0.757463
    elif 'WJetsHT600' in isamp: samples_wjets[isamp].kfactor = 0.856705
    elif 'WJetsHT400' in isamp: samples_wjets[isamp].kfactor = 0.928054
    elif 'WJetsHT200' in isamp: samples_wjets[isamp].kfactor = 0.978569

samples_ttbar_abcdnn = {
    "TTMT10002016APV":TTMT10002016APV,
    "TTMT10002016":TTMT10002016,
    "TTMT10002017":TTMT10002017,
    "TTMT10002018":TTMT10002018,
    "TTMT7002016APV":TTMT7002016APV,
    "TTMT7002016":TTMT7002016,
    "TTMT7002017":TTMT7002017,
    "TTMT7002018":TTMT7002018,
    "TTTo2L2Nu2016APV":TTTo2L2Nu2016APV,
    "TTTo2L2Nu2016":TTTo2L2Nu2016,
    "TTTo2L2Nu2017":TTTo2L2Nu2017,
    "TTTo2L2Nu2018":TTTo2L2Nu2018,
    "TTToHadronic2016APV":TTToHadronic2016APV,
    "TTToHadronic2016":TTToHadronic2016,
    "TTToHadronic2017":TTToHadronic2017,
    "TTToHadronic2018":TTToHadronic2018,
    "TTToSemiLeptonic2016APV":TTToSemiLeptonic2016APV,
    "TTToSemiLeptonic2016":TTToSemiLeptonic2016,
    "TTToSemiLeptonic2017":TTToSemiLeptonic2017,
    "TTToSemiLeptonic2018":TTToSemiLeptonic2018,
}

samples_ttbar = {
    "TTMT10002016APV":TTMT10002016APV,
    "TTMT10002016":TTMT10002016,
    "TTMT10002017":TTMT10002017,
    "TTMT10002018":TTMT10002018,
    "TTMT7002016APV":TTMT7002016APV,
    "TTMT7002016":TTMT7002016,
    "TTMT7002017":TTMT7002017,
    "TTMT7002018":TTMT7002018,
    "TTTo2L2Nu2016APV0":TTTo2L2Nu2016APV0,
    "TTTo2L2Nu20160":TTTo2L2Nu20160,
    "TTTo2L2Nu20170":TTTo2L2Nu20170,
    "TTTo2L2Nu20180":TTTo2L2Nu20180,
    "TTTo2L2Nu2016APV700":TTTo2L2Nu2016APV700,
    "TTTo2L2Nu2016700":TTTo2L2Nu2016700,
    "TTTo2L2Nu2017700":TTTo2L2Nu2017700,
    "TTTo2L2Nu2018700":TTTo2L2Nu2018700,
    "TTTo2L2Nu2016APV1000":TTTo2L2Nu2016APV1000,
    "TTTo2L2Nu20161000":TTTo2L2Nu20161000,
    "TTTo2L2Nu20171000":TTTo2L2Nu20171000,
    "TTTo2L2Nu20181000":TTTo2L2Nu20181000,
    "TTToHadronic2016APV0":TTToHadronic2016APV0,
    "TTToHadronic20160":TTToHadronic20160,
    "TTToHadronic20170":TTToHadronic20170,
    "TTToHadronic20180":TTToHadronic20180,
    "TTToHadronic2016APV700":TTToHadronic2016APV700,
    "TTToHadronic2016700":TTToHadronic2016700,
    "TTToHadronic2017700":TTToHadronic2017700,
    "TTToHadronic2018700":TTToHadronic2018700,
    "TTToHadronic2016APV1000":TTToHadronic2016APV1000,
    "TTToHadronic20161000":TTToHadronic20161000,
    "TTToHadronic20171000":TTToHadronic20171000,
    "TTToHadronic20181000":TTToHadronic20181000,
    "TTToSemiLeptonic2016APV0":TTToSemiLeptonic2016APV0,
    "TTToSemiLeptonic20160":TTToSemiLeptonic20160,
    "TTToSemiLeptonic20170":TTToSemiLeptonic20170,
    "TTToSemiLeptonic20180":TTToSemiLeptonic20180,
    "TTToSemiLeptonic2016APV700":TTToSemiLeptonic2016APV700,
    "TTToSemiLeptonic2016700":TTToSemiLeptonic2016700,
    "TTToSemiLeptonic2017700":TTToSemiLeptonic2017700,
    "TTToSemiLeptonic2018700":TTToSemiLeptonic2018700,
    "TTToSemiLeptonic2016APV1000":TTToSemiLeptonic2016APV1000,
    "TTToSemiLeptonic20161000":TTToSemiLeptonic20161000,
    "TTToSemiLeptonic20171000":TTToSemiLeptonic20171000,
    "TTToSemiLeptonic20181000":TTToSemiLeptonic20181000,
}

samples_singletop = {
    "STs2016APV":STs2016APV,
    "STs2016":STs2016,
    "STs2017":STs2017,
    "STs2018":STs2018,
    "STt2016APV":STt2016APV,
    "STt2016":STt2016,
    "STt2017":STt2017,
    "STt2018":STt2018,
    "STtb2016APV":STtb2016APV,
    "STtb2016":STtb2016,
    "STtb2017":STtb2017,
    "STtb2018":STtb2018,
    "STtW2016APV":STtW2016APV,
    "STtW2016":STtW2016,
    "STtW2017":STtW2017,
    "STtW2018":STtW2018,
    "STtWb2016APV":STtWb2016APV,
    "STtWb2016":STtWb2016,
    "STtWb2017":STtWb2017,
    "STtWb2018":STtWb2018,
}
samples_ttbarx = {
    "TTHB2016APV":TTHB2016APV,
    "TTHB2016":TTHB2016,
    "TTHB2017":TTHB2017,
    "TTHB2018":TTHB2018,
    "TTHnonB2016APV":TTHnonB2016APV,
    "TTHnonB2016":TTHnonB2016,
    "TTHnonB2017":TTHnonB2017,
    "TTHnonB2018":TTHnonB2018,
    "TTWl2016APV":TTWl2016APV,
    "TTWl2016":TTWl2016,
    "TTWl2017":TTWl2017,
    "TTWl2018":TTWl2018,
    "TTWq2016APV":TTWq2016APV,
    "TTWq2016":TTWq2016,
    "TTWq2017":TTWq2017,
    "TTWq2018":TTWq2018,
    "TTZM102016APV":TTZM102016APV,
    "TTZM102016":TTZM102016,
    "TTZM102017":TTZM102017,
    "TTZM102018":TTZM102018,
} 
samples_qcd = {
    "QCDHT10002016APV":QCDHT10002016APV,
    "QCDHT10002016":QCDHT10002016,
    "QCDHT10002017":QCDHT10002017,
    "QCDHT10002018":QCDHT10002018,
    "QCDHT15002016APV":QCDHT15002016APV,
    "QCDHT15002016":QCDHT15002016,
    "QCDHT15002017":QCDHT15002017,
    "QCDHT15002018":QCDHT15002018,
    "QCDHT20002016APV":QCDHT20002016APV,
    "QCDHT20002016":QCDHT20002016,
    "QCDHT20002017":QCDHT20002017,
    "QCDHT20002018":QCDHT20002018,
    #"QCDHT2002016APV":QCDHT2002016APV,
    #"QCDHT2002016":QCDHT2002016,
    #"QCDHT2002017":QCDHT2002017,
    #"QCDHT2002018":QCDHT2002018,
    "QCDHT3002016APV":QCDHT3002016APV,
    "QCDHT3002016":QCDHT3002016,
    "QCDHT3002017":QCDHT3002017,
    "QCDHT3002018":QCDHT3002018,
    "QCDHT5002016APV":QCDHT5002016APV,
    "QCDHT5002016":QCDHT5002016,
    "QCDHT5002017":QCDHT5002017,
    "QCDHT5002018":QCDHT5002018,
    "QCDHT7002016APV":QCDHT7002016APV,
    "QCDHT7002016":QCDHT7002016,
    "QCDHT7002017":QCDHT7002017,
    "QCDHT7002018":QCDHT7002018,
}
samples_data= {
    "SingleElecRun2016APVB":SingleElecRun2016APVB,
    "SingleElecRun2016APVC":SingleElecRun2016APVC,
    "SingleElecRun2016APVD":SingleElecRun2016APVD,
    "SingleElecRun2016APVE":SingleElecRun2016APVE,
    "SingleElecRun2016APVF":SingleElecRun2016APVF,
    "SingleElecRun2016F":SingleElecRun2016F,
    "SingleElecRun2016G":SingleElecRun2016G,
    "SingleElecRun2016H":SingleElecRun2016H,
    "SingleElecRun2017B":SingleElecRun2017B,
    "SingleElecRun2017C":SingleElecRun2017C,
    "SingleElecRun2017D":SingleElecRun2017D,
    "SingleElecRun2017E":SingleElecRun2017E,
    "SingleElecRun2017F":SingleElecRun2017F,
    "SingleElecRun2018A":SingleElecRun2018A,
    "SingleElecRun2018B":SingleElecRun2018B,
    "SingleElecRun2018C":SingleElecRun2018C,
    "SingleElecRun2018D":SingleElecRun2018D,
    "SingleMuonRun2016APVB":SingleMuonRun2016APVB,
    "SingleMuonRun2016APVC":SingleMuonRun2016APVC,
    "SingleMuonRun2016APVD":SingleMuonRun2016APVD,
    "SingleMuonRun2016APVE":SingleMuonRun2016APVE,
    "SingleMuonRun2016APVF":SingleMuonRun2016APVF,
    "SingleMuonRun2016F":SingleMuonRun2016F,
    "SingleMuonRun2016G":SingleMuonRun2016G,
    "SingleMuonRun2016H":SingleMuonRun2016H,
    "SingleMuonRun2017B":SingleMuonRun2017B,
    "SingleMuonRun2017C":SingleMuonRun2017C,
    "SingleMuonRun2017D":SingleMuonRun2017D,
    "SingleMuonRun2017E":SingleMuonRun2017E,
    "SingleMuonRun2017F":SingleMuonRun2017F,
    "SingleMuonRun2018A":SingleMuonRun2018A,
    "SingleMuonRun2018B":SingleMuonRun2018B,
    "SingleMuonRun2018C":SingleMuonRun2018C,
    "SingleMuonRun2018D":SingleMuonRun2018D,
}

samples={
    "Bprime_M1000_2016":Bprime_M1000_2016,
    "Bprime_M1000_2017":Bprime_M1000_2017,
    "Bprime_M1000_2018":Bprime_M1000_2018,
    "Bprime_M1200_2016":Bprime_M1200_2016,
    "Bprime_M1200_2017":Bprime_M1200_2017,
    "Bprime_M1200_2018":Bprime_M1200_2018,
    "Bprime_M1300_2016":Bprime_M1300_2016,
    "Bprime_M1300_2017":Bprime_M1300_2017,
    "Bprime_M1300_2018":Bprime_M1300_2018,
    "Bprime_M1400_2016":Bprime_M1400_2016,
    "Bprime_M1400_2017":Bprime_M1400_2017,
    "Bprime_M1400_2018":Bprime_M1400_2018,
    "Bprime_M1500_2016":Bprime_M1500_2016,
    "Bprime_M1500_2017":Bprime_M1500_2017,
    "Bprime_M1500_2018":Bprime_M1500_2018,
    "Bprime_M1600_2016":Bprime_M1600_2016,
    "Bprime_M1600_2017":Bprime_M1600_2017,
    "Bprime_M1600_2018":Bprime_M1600_2018,
    "Bprime_M1700_2016":Bprime_M1700_2016,
    "Bprime_M1700_2017":Bprime_M1700_2017,
    "Bprime_M1700_2018":Bprime_M1700_2018,
    "Bprime_M1800_2016":Bprime_M1800_2016,
    "Bprime_M1800_2017":Bprime_M1800_2017,
    "Bprime_M1800_2018":Bprime_M1800_2018,
    "Bprime_M2000_2016":Bprime_M2000_2016,
    "Bprime_M2000_2017":Bprime_M2000_2017,
    "Bprime_M2000_2018":Bprime_M2000_2018,
    "Bprime_M2200_2016":Bprime_M2200_2016,
    "Bprime_M2200_2017":Bprime_M2200_2017,
    "Bprime_M2200_2018":Bprime_M2200_2018,
    "Bprime_M800_2016":Bprime_M800_2016,
    "Bprime_M800_2017":Bprime_M800_2017,
    "Bprime_M800_2018":Bprime_M800_2018,
    "DYMHT12002016APV":DYMHT12002016APV,
    "DYMHT12002016":DYMHT12002016,
    "DYMHT12002017":DYMHT12002017,
    "DYMHT12002018":DYMHT12002018,
    "DYMHT2002016APV":DYMHT2002016APV,
    "DYMHT2002016":DYMHT2002016,
    "DYMHT2002017":DYMHT2002017,
    "DYMHT2002018":DYMHT2002018,
    "DYMHT25002016APV":DYMHT25002016APV,
    "DYMHT25002016":DYMHT25002016,
    "DYMHT25002017":DYMHT25002017,
    "DYMHT25002018":DYMHT25002018,
    "DYMHT4002016APV":DYMHT4002016APV,
    "DYMHT4002016":DYMHT4002016,
    "DYMHT4002017":DYMHT4002017,
    "DYMHT4002018":DYMHT4002018,
    "DYMHT6002016APV":DYMHT6002016APV,
    "DYMHT6002016":DYMHT6002016,
    "DYMHT6002017":DYMHT6002017,
    "DYMHT6002018":DYMHT6002018,
    "DYMHT8002016APV":DYMHT8002016APV,
    "DYMHT8002016":DYMHT8002016,
    "DYMHT8002017":DYMHT8002017,
    "DYMHT8002018":DYMHT8002018,
    "JetHTRun2016APVB":JetHTRun2016APVB,
    "JetHTRun2016APVC":JetHTRun2016APVC,
    "JetHTRun2016APVD":JetHTRun2016APVD,
    "JetHTRun2016APVE":JetHTRun2016APVE,
    "JetHTRun2016APVF":JetHTRun2016APVF,
    "JetHTRun2016F":JetHTRun2016F,
    "JetHTRun2016G":JetHTRun2016G,
    "JetHTRun2016H":JetHTRun2016H,
    "JetHTRun2017B":JetHTRun2017B,
    "JetHTRun2017C":JetHTRun2017C,
    "JetHTRun2017D":JetHTRun2017D,
    "JetHTRun2017E":JetHTRun2017E,
    "JetHTRun2017F":JetHTRun2017F,
    "JetHTRun2018A":JetHTRun2018A,
    "JetHTRun2018B":JetHTRun2018B,
    "JetHTRun2018C":JetHTRun2018C,
    "JetHTRun2018D":JetHTRun2018D,
    "QCDHT10002016APV":QCDHT10002016APV,
    "QCDHT10002016":QCDHT10002016,
    "QCDHT10002017":QCDHT10002017,
    "QCDHT10002018":QCDHT10002018,
    "QCDHT15002016APV":QCDHT15002016APV,
    "QCDHT15002016":QCDHT15002016,
    "QCDHT15002017":QCDHT15002017,
    "QCDHT15002018":QCDHT15002018,
    "QCDHT20002016APV":QCDHT20002016APV,
    "QCDHT20002016":QCDHT20002016,
    "QCDHT20002017":QCDHT20002017,
    "QCDHT20002018":QCDHT20002018,
    "QCDHT2002016APV":QCDHT2002016APV,
    "QCDHT2002016":QCDHT2002016,
    "QCDHT2002017":QCDHT2002017,
    "QCDHT2002018":QCDHT2002018,
    "QCDHT3002016APV":QCDHT3002016APV,
    "QCDHT3002016":QCDHT3002016,
    "QCDHT3002017":QCDHT3002017,
    "QCDHT3002018":QCDHT3002018,
    "QCDHT5002016APV":QCDHT5002016APV,
    "QCDHT5002016":QCDHT5002016,
    "QCDHT5002017":QCDHT5002017,
    "QCDHT5002018":QCDHT5002018,
    "QCDHT7002016APV":QCDHT7002016APV,
    "QCDHT7002016":QCDHT7002016,
    "QCDHT7002017":QCDHT7002017,
    "QCDHT7002018":QCDHT7002018,
    "SingleElecRun2016APVB":SingleElecRun2016APVB,
    "SingleElecRun2016APVC":SingleElecRun2016APVC,
    "SingleElecRun2016APVD":SingleElecRun2016APVD,
    "SingleElecRun2016APVE":SingleElecRun2016APVE,
    "SingleElecRun2016APVF":SingleElecRun2016APVF,
    "SingleElecRun2016F":SingleElecRun2016F,
    "SingleElecRun2016G":SingleElecRun2016G,
    "SingleElecRun2016H":SingleElecRun2016H,
    "SingleElecRun2017B":SingleElecRun2017B,
    "SingleElecRun2017C":SingleElecRun2017C,
    "SingleElecRun2017D":SingleElecRun2017D,
    "SingleElecRun2017E":SingleElecRun2017E,
    "SingleElecRun2017F":SingleElecRun2017F,
    "SingleElecRun2018A":SingleElecRun2018A,
    "SingleElecRun2018B":SingleElecRun2018B,
    "SingleElecRun2018C":SingleElecRun2018C,
    "SingleElecRun2018D":SingleElecRun2018D,
    "SingleMuonRun2016APVB":SingleMuonRun2016APVB,
    "SingleMuonRun2016APVC":SingleMuonRun2016APVC,
    "SingleMuonRun2016APVD":SingleMuonRun2016APVD,
    "SingleMuonRun2016APVE":SingleMuonRun2016APVE,
    "SingleMuonRun2016APVF":SingleMuonRun2016APVF,
    "SingleMuonRun2016F":SingleMuonRun2016F,
    "SingleMuonRun2016G":SingleMuonRun2016G,
    "SingleMuonRun2016H":SingleMuonRun2016H,
    "SingleMuonRun2017B":SingleMuonRun2017B,
    "SingleMuonRun2017C":SingleMuonRun2017C,
    "SingleMuonRun2017D":SingleMuonRun2017D,
    "SingleMuonRun2017E":SingleMuonRun2017E,
    "SingleMuonRun2017F":SingleMuonRun2017F,
    "SingleMuonRun2018A":SingleMuonRun2018A,
    "SingleMuonRun2018B":SingleMuonRun2018B,
    "SingleMuonRun2018C":SingleMuonRun2018C,
    "SingleMuonRun2018D":SingleMuonRun2018D,
    "STs2016APV":STs2016APV,
    "STs2016":STs2016,
    "STs2017":STs2017,
    "STs2018":STs2018,
    "STt2016APV":STt2016APV,
    "STt2016":STt2016,
    "STt2017":STt2017,
    "STt2018":STt2018,
    "STtb2016APV":STtb2016APV,
    "STtb2016":STtb2016,
    "STtb2017":STtb2017,
    "STtb2018":STtb2018,
    "STtW2016APV":STtW2016APV,
    "STtW2016":STtW2016,
    "STtW2017":STtW2017,
    "STtW2018":STtW2018,
    "STtWb2016APV":STtWb2016APV,
    "STtWb2016":STtWb2016,
    "STtWb2017":STtWb2017,
    "STtWb2018":STtWb2018,
    "TTHB2016APV":TTHB2016APV,
    "TTHB2016":TTHB2016,
    "TTHB2017":TTHB2017,
    "TTHB2018":TTHB2018,
    "TTHnonB2016APV":TTHnonB2016APV,
    "TTHnonB2016":TTHnonB2016,
    "TTHnonB2017":TTHnonB2017,
    "TTHnonB2018":TTHnonB2018,
    "TTMT10002016APV":TTMT10002016APV,
    "TTMT10002016":TTMT10002016,
    "TTMT10002017":TTMT10002017,
    "TTMT10002018":TTMT10002018,
    "TTMT7002016APV":TTMT7002016APV,
    "TTMT7002016":TTMT7002016,
    "TTMT7002017":TTMT7002017,
    "TTMT7002018":TTMT7002018,
    "TTTo2L2Nu2016APV":TTTo2L2Nu2016APV,
    "TTTo2L2Nu2016":TTTo2L2Nu2016,
    "TTTo2L2Nu2017":TTTo2L2Nu2017,
    "TTTo2L2Nu2018":TTTo2L2Nu2018,
    "TTToHadronic2016APV":TTToHadronic2016APV,
    "TTToHadronic2016":TTToHadronic2016,
    "TTToHadronic2017":TTToHadronic2017,
    "TTToHadronic2018":TTToHadronic2018,
    "TTToSemiLeptonic2016APV":TTToSemiLeptonic2016APV,
    "TTToSemiLeptonic2016":TTToSemiLeptonic2016,
    "TTToSemiLeptonic2017":TTToSemiLeptonic2017,
    "TTToSemiLeptonic2018":TTToSemiLeptonic2018,
    "TTWl2016APV":TTWl2016APV,
    "TTWl2016":TTWl2016,
    "TTWl2017":TTWl2017,
    "TTWl2018":TTWl2018,
    "TTWq2016APV":TTWq2016APV,
    "TTWq2016":TTWq2016,
    "TTWq2017":TTWq2017,
    "TTWq2018":TTWq2018,
    "TTZM102016APV":TTZM102016APV,
    "TTZM102016":TTZM102016,
    "TTZM102017":TTZM102017,
    "TTZM102018":TTZM102018,
    "TTZM1to102016APV":TTZM1to102016APV,
    "TTZM1to102016":TTZM1to102016,
    "TTZM1to102017":TTZM1to102017,
    "TTZM1to102018":TTZM1to102018,
    "WJetsHT12002016APV":WJetsHT12002016APV,
    "WJetsHT12002016":WJetsHT12002016,
    "WJetsHT12002017":WJetsHT12002017,
    "WJetsHT12002018":WJetsHT12002018,
    "WJetsHT2002016APV":WJetsHT2002016APV,
    "WJetsHT2002016":WJetsHT2002016,
    "WJetsHT2002017":WJetsHT2002017,
    "WJetsHT2002018":WJetsHT2002018,
    "WJetsHT25002016APV":WJetsHT25002016APV,
    "WJetsHT25002016":WJetsHT25002016,
    "WJetsHT25002017":WJetsHT25002017,
    "WJetsHT25002018":WJetsHT25002018,
    "WJetsHT4002016APV":WJetsHT4002016APV,
    "WJetsHT4002016":WJetsHT4002016,
    "WJetsHT4002017":WJetsHT4002017,
    "WJetsHT4002018":WJetsHT4002018,
    "WJetsHT6002016APV":WJetsHT6002016APV,
    "WJetsHT6002016":WJetsHT6002016,
    "WJetsHT6002017":WJetsHT6002017,
    "WJetsHT6002018":WJetsHT6002018,
    "WJetsHT8002016APV":WJetsHT8002016APV,
    "WJetsHT8002016":WJetsHT8002016,
    "WJetsHT8002017":WJetsHT8002017,
    "WJetsHT8002018":WJetsHT8002018,
    "WW2016APV":WW2016APV,
    "WW2016":WW2016,
    "WW2017":WW2017,
    "WW2018":WW2018,
    "WZ2016APV":WZ2016APV,
    "WZ2016":WZ2016,
    "WZ2017":WZ2017,
    "WZ2018":WZ2018,
    "ZZ2016APV":ZZ2016APV,
    "ZZ2016":ZZ2016,
    "ZZ2017":ZZ2017,
    "ZZ2018":ZZ2018,
}

mclist_2016APV = [
    Bprime_M1000_2016APV,
    Bprime_M1200_2016APV,
    Bprime_M1300_2016APV,
    Bprime_M1400_2016APV,
    Bprime_M1500_2016APV,
    Bprime_M1600_2016APV,
    Bprime_M1700_2016APV,
    Bprime_M1800_2016APV,
    Bprime_M2000_2016APV,
    Bprime_M2200_2016APV,
    Bprime_M800_2016APV,
    DYMHT12002016APV,
    DYMHT2002016APV,
    DYMHT25002016APV,
    DYMHT4002016APV,
    DYMHT6002016APV,
    DYMHT8002016APV,
    QCDHT10002016APV,
    QCDHT15002016APV,
    QCDHT20002016APV,
    QCDHT2002016APV,
    QCDHT3002016APV,
    QCDHT5002016APV,
    QCDHT7002016APV,
    STs2016APV,
    STt2016APV,
    STtb2016APV,
    STtW2016APV,
    STtWb2016APV,
    TTHB2016APV,
    TTHnonB2016APV,
    TTMT10002016APV,
    TTMT7002016APV,
    TTTo2L2Nu2016APV,
    TTToHadronic2016APV,
    TTToSemiLeptonic2016APV,
    TTWl2016APV,
    TTWq2016APV,
    TTZM102016APV,
    TTZM1to102016APV,
    WJetsHT12002016APV,
    WJetsHT2002016APV,
    WJetsHT25002016APV,
    WJetsHT4002016APV,
    WJetsHT6002016APV,
    WJetsHT8002016APV,
    WW2016APV,
    WZ2016APV,
    ZZ2016APV,
]

mclist_2016 = [
    Bprime_M1000_2016,
    Bprime_M1200_2016,
    Bprime_M1300_2016,
    Bprime_M1400_2016,
    Bprime_M1500_2016,
    Bprime_M1600_2016,
    Bprime_M1700_2016,
    Bprime_M1800_2016,
    Bprime_M2000_2016,
    Bprime_M2200_2016,
    Bprime_M800_2016,
    DYMHT12002016,
    DYMHT2002016,
    DYMHT25002016,
    DYMHT4002016,
    DYMHT6002016,
    DYMHT8002016,
    QCDHT10002016,
    QCDHT15002016,
    QCDHT20002016,
    QCDHT2002016,
    QCDHT3002016,
    QCDHT5002016,
    QCDHT7002016,
    STs2016,
    STt2016,
    STtb2016,
    STtW2016,
    STtWb2016,
    TTHB2016,
    TTHnonB2016,
    TTMT10002016,
    TTMT7002016,
    TTTo2L2Nu2016,
    TTToHadronic2016,
    TTToSemiLeptonic2016,
    TTWl2016,
    TTWq2016,
    TTZM102016,
    TTZM1to102016,
    WJetsHT12002016,
    WJetsHT2002016,
    WJetsHT25002016,
    WJetsHT4002016,
    WJetsHT6002016,
    WJetsHT8002016,
    WW2016,
    WZ2016,
    ZZ2016,
]

mclist_2017 = [
    Bprime_M1000_2017,
    Bprime_M1200_2017,
    Bprime_M1300_2017,
    Bprime_M1400_2017,
    Bprime_M1500_2017,
    Bprime_M1600_2017,
    Bprime_M1700_2017,
    Bprime_M1800_2017,
    Bprime_M2000_2017,
    Bprime_M2200_2017,
    Bprime_M800_2017,
    DYMHT12002017,
    DYMHT2002017,
    DYMHT25002017,
    DYMHT4002017,
    DYMHT6002017,
    DYMHT8002017,
    QCDHT10002017,
    QCDHT15002017,
    QCDHT20002017,
    QCDHT2002017,
    QCDHT3002017,
    QCDHT5002017,
    QCDHT7002017,
    STs2017,
    STt2017,
    STtb2017,
    STtW2017,
    STtWb2017,
    TTHB2017,
    TTHnonB2017,
    TTMT10002017,
    TTMT7002017,
    TTTo2L2Nu2017,
    TTToHadronic2017,
    TTToSemiLeptonic2017,
    TTWl2017,
    TTWq2017,
    TTZM102017,
    TTZM1to102017,
    WJetsHT12002017,
    WJetsHT2002017,
    WJetsHT25002017,
    WJetsHT4002017,
    WJetsHT6002017,
    WJetsHT8002017,
    WW2017,
    WZ2017,
    ZZ2017,
]

mclist_2018 = [
    Bprime_M1000_2018,
    Bprime_M1200_2018,
    Bprime_M1300_2018,
    Bprime_M1400_2018,
    Bprime_M1500_2018,
    Bprime_M1600_2018,
    Bprime_M1700_2018,
    Bprime_M1800_2018,
    Bprime_M2000_2018,
    Bprime_M2200_2018,
    Bprime_M800_2018,
    DYMHT12002018,
    DYMHT2002018,
    DYMHT25002018,
    DYMHT4002018,
    DYMHT6002018,
    DYMHT8002018,
    QCDHT10002018,
    QCDHT15002018,
    QCDHT20002018,
    QCDHT2002018,
    QCDHT3002018,
    QCDHT5002018,
    QCDHT7002018,
    STs2018,
    STt2018,
    STtb2018,
    STtW2018,
    STtWb2018,
    TTHB2018,
    TTHnonB2018,
    TTMT10002018,
    TTMT7002018,
    TTTo2L2Nu2018,
    TTToHadronic2018,
    TTToSemiLeptonic2018,
    TTWl2018,
    TTWq2018,
    TTZM102018,
    TTZM1to102018,
    WJetsHT12002018,
    WJetsHT2002018,
    WJetsHT25002018,
    WJetsHT4002018,
    WJetsHT6002018,
    WJetsHT8002018,
    WW2018,
    WZ2018,
    ZZ2018,
]

