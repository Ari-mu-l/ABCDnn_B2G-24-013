import ROOT
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib import cm
from matplotlib import patches
import os
#from tqdm import tqdm
from samples import *
from utils import readTreeNominal
import itertools

getHistos = False

ROOT.gInterpreter.Declare("""
    float compute_weight( float optionalWeights, float genWeight, float lumi, float xsec, float nRun, float PileupWeights, float leptonRecoSF, float leptonIDSF, float leptonIsoSF, float leptonHLTSF, float btagWeights ){
    return optionalWeights * PileupWeights * leptonRecoSF * leptonIDSF * leptonIsoSF * leptonHLTSF * btagWeights * genWeight * lumi * xsec / (nRun * abs(genWeight));
    }
""")

def extended_ABCD( A, B, C, X, Y ):
    D_yield = B * X * C**2 / ( A**2 * Y )
    D_err   = D_yield * np.sqrt( 1/B + 1/X + 1/Y + 4/C + 4/A )
    return D_yield, D_err

def ABCD( A, B, C ):
    D_yield = B * C / A
    D_err = D_yield * np.sqrt( 1/B + 1/C + 1/D_yield )
    return D_yield, D_err

rPath = "root://cmseos.fnal.gov//store/user/jmanagan/"
rDir = rPath + "BtoTW_Oct2023_fullRun2"
nTree = "Events_Nominal"
fDATA = [
    "SingleElecRun2016APVB",
    "SingleElecRun2016APVC",
    "SingleElecRun2016APVD",
    "SingleElecRun2016APVE",
    "SingleElecRun2016APVF",
    "SingleElecRun2016F",
    "SingleElecRun2016G",
    "SingleElecRun2016H",
    "SingleElecRun2017B",
    "SingleElecRun2017C",
    "SingleElecRun2017D",
    "SingleElecRun2017E",
    "SingleElecRun2017F",
    "SingleElecRun2018A",
    "SingleElecRun2018B",
    #"SingleElecRun2018C",
    #"SingleElecRun2018D",
    "SingleMuonRun2016APVB",
    "SingleMuonRun2016APVC",
    "SingleMuonRun2016APVD",
    "SingleMuonRun2016APVE",
    "SingleMuonRun2016APVF",
    "SingleMuonRun2016F",
    "SingleMuonRun2016G",
    "SingleMuonRun2016H",
    "SingleMuonRun2017B",
    "SingleMuonRun2017C",
    "SingleMuonRun2017D",
    "SingleMuonRun2017E",
    "SingleMuonRun2017F",
    "SingleMuonRun2018A",
    "SingleMuonRun2018B",
    "SingleMuonRun2018C",
    "SingleMuonRun2018D"
]
fSIG = [
    "Bprime_M1400_2016",
    "Bprime_M1400_2017",
    "Bprime_M1400_2018",
]
fBKG = {
    "EWK": [
        "DYMHT12002016APV",
        "DYMHT12002016",
        "DYMHT12002017",
        "DYMHT12002018",
        "DYMHT2002016APV",
        "DYMHT2002016",
        "DYMHT2002017",
        "DYMHT2002018",
        "DYMHT25002016APV",
        "DYMHT25002016",
        "DYMHT25002017",
        "DYMHT25002018",
        "DYMHT4002016APV",
        "DYMHT4002016",
        "DYMHT4002017",
        "DYMHT4002018",
        "DYMHT6002016APV",
        "DYMHT6002016",
        "DYMHT6002017",
        "DYMHT6002018",
        "DYMHT8002016APV",
        "DYMHT8002016",
        "DYMHT8002017",
        "DYMHT8002018",
        "WW2016APV",
        "WW2016",
        "WW2017",
        "WW2018",
        "WZ2016APV",
        "WZ2016",
        "WZ2017",
        "WZ2018",
        "ZZ2016APV",
        "ZZ2016",
        "ZZ2017",
        "ZZ2018",
    ],
    "WJETS": [
        "WJetsHT12002016APV",
        "WJetsHT12002016",
        "WJetsHT12002017",
        "WJetsHT12002018",
        "WJetsHT2002016APV",
        "WJetsHT2002016",
        "WJetsHT2002017",
        "WJetsHT2002018",
        "WJetsHT25002016APV",
        "WJetsHT25002016",
        "WJetsHT25002017",
        "WJetsHT25002018",
        "WJetsHT4002016APV",
        "WJetsHT4002016",
        "WJetsHT4002017",
        "WJetsHT4002018",
        "WJetsHT6002016APV",
        "WJetsHT6002016",
        "WJetsHT6002017",
        "WJetsHT6002018",
        "WJetsHT8002016APV",
        "WJetsHT8002016",
        "WJetsHT8002017",
        "WJetsHT8002018",
    ],
    "TTBAR": [
        "TTMT10002016APV",
        "TTMT10002016",
        "TTMT10002017",
        "TTMT10002018",
        "TTMT7002016APV",
        "TTMT7002016",
        "TTMT7002017",
        "TTMT7002018",
        "TTTo2L2Nu2016APV",
        "TTTo2L2Nu2016",
        "TTTo2L2Nu2017",
        "TTTo2L2Nu2018",
        "TTToHadronic2016APV",
        "TTToHadronic2016",
        "TTToHadronic2017",
        "TTToHadronic2018",
        "TTToSemiLeptonic2016APV",
        "TTToSemiLeptonic2016",
        "TTToSemiLeptonic2017",
        "TTToSemiLeptonic2018",
    ],
    "ST": [
        "STs2016APV",
        "STs2016",
        "STs2017",
        "STs2018",
        "STt2016APV",
        "STt2016",
        "STt2017",
        "STt2018",
        "STtb2016APV",
        "STtb2016",
        "STtb2017",
        "STtb2018",
        "STtW2016APV",
        "STtW2016",
        "STtW2017",
        "STtW2018",
        "STtWb2016APV",
        "STtWb2016",
        "STtWb2017",
        "STtWb2018",
    ],
    "TTBARX": [
        "TTHB2016APV",
        "TTHB2016",
        "TTHB2017",
        "TTHB2018",
        "TTHnonB2016APV",
        "TTHnonB2016",
        "TTHnonB2017",
        "TTHnonB2018",
        "TTWl2016APV",
        "TTWl2016",
        "TTWl2017",
        "TTWl2018",
        "TTWq2016APV",
        "TTWq2016",
        "TTWq2017",
        "TTWq2018",
        "TTZM102016APV",
        "TTZM102016",
        "TTZM102017",
    "TTZM102018"
    ],
    "QCD": [
        "QCDHT10002016APV",
        "QCDHT10002016",
        "QCDHT10002017",
        "QCDHT10002018",
        "QCDHT15002016APV",
        "QCDHT15002016",
        "QCDHT15002017",
        "QCDHT15002018",
        "QCDHT20002016APV",
        "QCDHT20002016",
        "QCDHT20002017",
        "QCDHT20002018",
        #"QCDHT2002016APV",
        #"QCDHT2002016",
        #"QCDHT2002017",
        #"QCDHT2002018",
        "QCDHT3002016APV",
        "QCDHT3002016",
        "QCDHT3002017",
        "QCDHT3002018",
        "QCDHT5002016APV",
        "QCDHT5002016",
        "QCDHT5002017",
        "QCDHT5002018",
        "QCDHT7002016APV",
        "QCDHT7002016",
        "QCDHT7002017",
        "QCDHT7002018",
    ],   
}

fBase = "W_MT <= 200"
variables = [
    "NJets_DeepFlavL",
    "NJets_forward",
    "Bprime_mass",
    "gcJet_ST"
]


hists_Bp = { "DATA": {}, "SIG": {}, "BKG": {} }
yields = { "DATA": {}, "SIG": {}, "BKG": {} }
mc_count = { "BKG": {} }
regions = {"A": "NJets_forward==0 && NJets_DeepFlavL==3",
           "B": "NJets_forward==0 && NJets_DeepFlavL<3" ,
           "C": "NJets_forward>0  && NJets_DeepFlavL==3",
           "D": "NJets_forward>0  && NJets_DeepFlavL<3" ,
           "X": "NJets_forward==0 && NJets_DeepFlavL>3" ,
           "Y": "NJets_forward>0  && NJets_DeepFlavL>3" }

print( "Loading data:" )
if (getHistos):
    outFile = ROOT.TFile.Open( "HistData_Bprime_mass.root", "RECREATE" )
for f in fDATA:
    samplename = samples[f].samplename.split('/')[1]+samples[f].samplename.split('-')[0][-1]
    year = (f.split('Run')[1])[:-1]
    #print(samplename, year)
    fChain = readTreeNominal( samplename, year, rDir)
    rDF_ = ROOT.RDataFrame( fChain ).Filter( fBase ).Define( "weight", "1.0" )
    if getHistos:
        outFile.cd()
    for region in regions:
        rDF_region = rDF_.Filter(regions[region])
        #yields["DATA"][f] = rDF_region.Count().GetValue()
        if (getHistos):
            hist_Bp = rDF_region.Histo1D((f, "Bprime_mass", 51, 0, 5000), "Bprime_mass", "weight")
            hist_Bp.Write()

print( "Loading signal:" )
if (getHistos):
    outFile = ROOT.TFile.Open( "HistSig_Bprime_mass.root", "RECREATE" )
for f in fSIG:
    sample = samples[f]
    samplename = sample.samplename.split('/')[1]
    year = f.split('_')[-1]
    fChain = readTreeNominal( samplename, year, rDir)
    rDF_ = ROOT.RDataFrame( fChain ).Filter( fBase ).Define( "weight", "compute_weight( {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {} )".format("1.0", "genWeight", targetlumi[year], sample.xsec, sample.nrun, "PileupWeights[0]", "leptonRecoSF[0]", "leptonIDSF[0]", "leptonIsoSF[0]", "leptonHLTSF[0]", "btagWeights[17]") )
    if getHistos:
        outFile.cd()
    for region in regions:
        rDF_region = rDF_.Filter(regions[region])
        #yields["SIG"][f] = rDF_region.Count().GetValue()
        if (getHistos):
            hist_Bp = rDF_region.Histo1D((f, "Bprime_mass", 51, 0, 5000), "Bprime_mass", "weight")
            hist_Bp.Write()

if getHistos:
    outFile = ROOT.TFile.Open( "HistBkg_Bprime_mass.root", "RECREATE" )
for g in fBKG:
    yields["BKG"][g] = {}
    print( "Loading background group: " + g )
    for f in fBKG[g]:
        sample = samples[f]
        samplename = sample.samplename.split('/')[1]
        if("APV" in f):
            year = f[-7:]
        else:
            year = f[-4:]
        fChain = readTreeNominal( samplename, year, rDir)
        if "_WJetsHT"  in samplename:
            rDF_ = ROOT.RDataFrame( fChain ).Filter( fBase ).Define( "weight", "compute_weight( {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {} )".format("gcHTCorr_WjetLHE[0]", "genWeight", targetlumi[year], sample.xsec, sample.nrun, "PileupWeights[0]", "leptonRecoSF[0]", "leptonIDSF[0]", "leptonIsoSF[0]", "leptonHLTSF[0]", "btagWeights[17]") )
        elif "TTTo" in samplename or  'TTMT' in samplename:
            rDF_ = ROOT.RDataFrame( fChain ).Filter( fBase ).Define( "weight", "compute_weight( {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {} )".format("gcHTCorr_top[0]", "genWeight", targetlumi[year], sample.xsec, sample.nrun, "PileupWeights[0]", "leptonRecoSF[0]", "leptonIDSF[0]", "leptonIsoSF[0]", "leptonHLTSF[0]", "btagWeights[17]") )
        else:
            rDF_ = ROOT.RDataFrame( fChain ).Filter( fBase ).Define( "weight", "compute_weight( {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {} )".format("1.0", "genWeight", targetlumi[year], sample.xsec, sample.nrun, "PileupWeights[0]", "leptonRecoSF[0]", "leptonIDSF[0]", "leptonIsoSF[0]", "leptonHLTSF[0]", "btagWeights[17]") )
        if getHistos:
            outFile.cd()
        for region in regions:
            rDF_region = rDF_.Filter(regions[region])
            #yields["BKG"][g][f] = rDF_region.Count().GetValue()
            if (getHistos):
                hist_Bp = rDF_region.Histo1D((f, "Bprime_mass", 51, 0, 5000), "Bprime_mass", "weight")
                hist_Bp.Write()

# Get yields
yields_bkg_region = {}
for region in regions:
    print("Region {}: ".format(region))
    yields_data = 0
    for f in fDATA:
        yields_data += yields["DATA"][f]
        print("Data: {}".format(yields_data))
    yields_bkg = 0
    for g in fBKG:
        yields_bkg_group = 0
        for f in fBKG[g]:
            print("  -{}: {}".format(f, yields["BKG"][g][f]))
            yields_bkg_group += yields["BKG"][g][f]
        print("Background group {}: {}".format(g, yields_bkg_group))
        yields_bkg += yields_bkg_group
    print("Total number of background: {}".format(yields_bkg))
    print(" ")
    yields_bkg_region[region] = yields_bkg
    for f in fSIG:
        print("Signal {}: {}".format(f, yields["SIG"][f]))

print("Now printing yield percentage")
# Get yield percentage
for region in regions:
    print("Background breakdown for region {}".format(region))
    for g in fBKG:
        yields_bkg_group = 0
        for f in fBKG[g]:
            print("  -{}: {}%".format(f, 100*yields["BKG"][g][f]/yields_bkg_region[region]))
            yields_bkg_group += yields["BKG"][g][f]
        print("Background group {}: {:}%".format(g, 100*yields_bkg_group/yields_bkg_region[region]))
    print(" ")

