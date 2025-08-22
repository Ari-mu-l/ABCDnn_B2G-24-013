# copy the desired rootfiles to working directory first

import os
import numpy as np
from ROOT import *
import json

gStyle.SetOptFit(1)
gStyle.SetOptStat(0)
gROOT.SetBatch(True) # suppress histogram display
TH1.SetDefaultSumw2(True)

model_case14 = "22"
model_case23 = "33"

rootDir_case14 = f'logBpMlogST_mmd1_case14_random{model_case14}'
rootDir_case23 = f'logBpMlogST_mmd1_case23_random{model_case23}'

binlo = 400
binhi = 2500
bins = 210 #210 #105 for 2016 and 210 for full Run2 # ANv7 2D: 105 bins
year = '' # '', '_2016'

doV2 = True #IMPORTANT: REMEMBER TO TURN ON AND OFF!!
withoutCorrection = False
withFit = False
separateUncertCases = True

if withoutCorrection:
    outDirTag = ''
else:
    outDirTag = f'NoCorrectionByCategory'

tag = {"case1" : "tagTjet",
       "case2" : "tagWjet",
       "case3" : "untagTlep",
       "case4" : "untagWlep",
       }

outDir = '/uscms/home/xshen/nobackup/alma9/CMSSW_13_3_3/src/vlq-BtoTW-SLA/makeTemplates'

def modifyOverflow(hist, bins):
    content = hist.GetBinContent(bins)+hist.GetBinContent(bins+1)
    error = np.sqrt(hist.GetBinError(bins)**2+hist.GetBinError(bins+1)**2)
    
    hist.SetBinContent(bins, content)
    hist.SetBinContent(bins+1, 0)

    hist.SetBinError(bins, error)
    hist.SetBinError(bins+1, 0)
    

alphaFactors = {}
with open("alphaRatio_factors.json","r") as alphaFile:
    alphaFactors = json.load(alphaFile)

def fillHistogram(hist):
    # TEMP: turned of negative bin adjustment
    # Need to turn back on if not using smoothing (smoothing script takes care of negative bins)
    for i in range(bins+1): # deals with negative bins
        if hist.GetBinContent(i)<0:
            hist.SetBinContent(i,0)
    return hist

regionMap = {"A": "A", "B": "B", "C": "C", "D": "D", "V":"V", "BV":"BV","highST": "D2"} # TEMP: named highST as D2 in the intermediate file
def createHist(case, region, histType, shift): # histType: Nominal, pNet, trainUncert, correct (correctDn==original). shift: Up, Dn
    if case=="case1" or case=="case4":
        rootDir = rootDir_case14
    else:
        rootDir = rootDir_case23

    #histFile = TFile.Open(f'{rootDir}/hists_ABCDnn_{case}_400to2500_210_pNet_byCategory.root', "READ")
    histFile = TFile.Open(f'{rootDir}/hists_ABCDnn_{case}_400to2500_210_pNet.root', "READ")

    with open(f'alphaRatio_factors{year}.json',"r") as alphaFile:
        alphaFactors = json.load(alphaFile)
    
    if "Nominal" in histType:
        hist_ttbar     = histFile.Get(f'Bprime_mass_pre_{region}_ttbar').Clone()
        hist_wjets     = histFile.Get(f'Bprime_mass_pre_{region}_wjets').Clone()
        hist_qcd       = histFile.Get(f'Bprime_mass_pre_{region}_qcd').Clone()
        hist_singletop = histFile.Get(f'Bprime_mass_pre_{region}_singletop').Clone()

        totalSamples = hist_ttbar.Integral() + hist_wjets.Integral() + hist_qcd.Integral() + hist_singletop.Integral()
        normalization = alphaFactors[case][region]["prediction"] / totalSamples
        #ttbarFrac     = (hist_ttbar.Integral() / totalSamples) * alphaFactors[case][region][prediction]
        #wjetsFrac     = (hist_wjets.Integral() / totalSamples) * alphaFactors[case][region][prediction]
        #qcdFrac       = (hist_qcd.Integral() / totalSamples) * alphaFactors[case][region][prediction]
        #singletopFrac = (hist_singletop.Integral() / totalSamples) * alphaFactors[case][region][prediction]
        
        hist_ttbar.Scale(normalization)
        hist_wjets.Scale(normalization)
        hist_qcd.Scale(normalization)
        hist_singletop.Scale(normalization)
        
        modifyOverflow(hist_ttbar,bins)
        modifyOverflow(hist_wjets,bins)
        modifyOverflow(hist_qcd,bins)
        modifyOverflow(hist_singletop,bins)
        
        outNameTag = ''
        hist_ttbar_out     = fillHistogram(hist_ttbar)
        hist_wjets_out     = fillHistogram(hist_wjets)
        hist_qcd_out       = fillHistogram(hist_qcd)
        hist_singletop_out = fillHistogram(hist_singletop)
    
    if doV2:
        if (region=="V" and (case=="case1" or case=="case2")) or (region=="D" and (case=="case3" or case=="case4")):
            outFile = TFile.Open(f'{outDir}/templatesV2_Jan2025_{bins}bins{outDirTag}/templates_BpMass_ABCDnn_138fbfb{year}.root', "UPDATE")
            
            hist_ttbar_out.SetTitle("")
            hist_wjets_out.SetTitle("")
            hist_qcd_out.SetTitle("")
            hist_singletop_out.SetTitle("")
            
            hist_ttbar_out.SetName(f'BpMass_ABCDnn_138fbfb_isL_{tag[case]}_V2__ttbar{outNameTag}')
            hist_wjets_out.SetName(f'BpMass_ABCDnn_138fbfb_isL_{tag[case]}_V2__wjets{outNameTag}')
            hist_qcd_out.SetName(f'BpMass_ABCDnn_138fbfb_isL_{tag[case]}_V2__qcd{outNameTag}')
            hist_singletop_out.SetName(f'BpMass_ABCDnn_138fbfb_isL_{tag[case]}_V2__singletop{outNameTag}')
            
            hist_ttbar_out.Write(f'BpMass_ABCDnn_138fbfb_isL_{tag[case]}_V2__ttbar{outNameTag}', TObject.kOverwrite)
            hist_wjets_out.Write(f'BpMass_ABCDnn_138fbfb_isL_{tag[case]}_V2__wjets{outNameTag}', TObject.kOverwrite)
            hist_qcd_out.Write(f'BpMass_ABCDnn_138fbfb_isL_{tag[case]}_V2__qcd{outNameTag}', TObject.kOverwrite)
            hist_singletop_out.Write(f'BpMass_ABCDnn_138fbfb_isL_{tag[case]}_V2__singletop{outNameTag}', TObject.kOverwrite)
            
            outFile.Close()
    else:
        os.exit('Region D not implemented yet!')
        
    # else:
    #     outFile = TFile.Open(f'{outDir}/templates{region}_Jan2025_{bins}bins{outDirTag}/templates_BpMass_ABCDnn_138fbfb{year}.root', "UPDATE")
    #     #if region=="highST":
    #     #    print(hist_out.Integral())
    #     #    print(f'BpMass_ABCDnn_138fbfb_isL_{tag[case]}_{region}__major{outNameTag}')
    #     hist_out.SetTitle("")
    #     hist_out.SetName(f'BpMass_ABCDnn_138fbfb_isL_{tag[case]}_{region}__major{outNameTag}')
    #     hist_out.Write(f'BpMass_ABCDnn_138fbfb_isL_{tag[case]}_{region}__major{outNameTag}', TObject.kOverwrite)
    #     outFile.Close()
    # histFile.Close()

#histList = ["Nominal", "pNet", "trainUncert", "correct", "smooth"]
histList = ["Nominal"]

for case in ["case1", "case2", "case3", "case4"]:
    #createHist(case, "B", "Nominal", "")
    #createHist(case, "BV", "Nominal", "")
    for histType in histList:
        if histType == "Nominal":
            shiftList = [""]
        else:
            shiftList = ["Up", "Dn"]
        for shift in shiftList:
            createHist(case, "D", histType, shift)
            #if year=='':
            createHist(case, "V", histType, shift)
            ##createHist(case, "highST", histType, shift)
