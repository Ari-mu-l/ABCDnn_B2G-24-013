# Makes uncorrected V2 plot in AN7

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
bins = 210 #105 for 2016 and 210 for full Run2
year = '' # '', '_2016'

outDirTag = '_noCorrection'

doV2 = True #IMPORTANT: REMEMBER TO TURN ON AND OFF!!
separateUncertCases = True

tag = {"case1" : "tagTjet",
       "case2" : "tagWjet",
       "case3" : "untagTlep",
       "case4" : "untagWlep",
       }

outDir = '/uscms/home/xshen/nobackup/alma9/CMSSW_13_3_3/src/vlq-BtoTW-SLA/makeTemplates'

def modifyOverflow(hist, bins):
    hist.SetBinContent(bins, hist.GetBinContent(bins)+hist.GetBinContent(bins+1))
    hist.SetBinContent(bins+1, 0)

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
    histFile = TFile.Open(f'{rootDir}/hists_ABCDnn_{case}_BpM400to2500ST0to1500_420bins30bins_pNet{year}_modified.root', "READ")
    if "Nominal" in histType:
        histFile2 = TFile.Open(f'{rootDir}/hists_ABCDnn_{case}_BpM400to2500ST0to1500_420bins30bins_pNet{year}.root', "READ")
        hist2D = histFile2.Get(f'BpMST_pre_{regionMap[region]}').Clone(f'Bprime_mass_pre_{regionMap[region]}_1D')
        hist2D.RebinX(2)
        hist2D.Scale(alphaFactors[case][region]["prediction"]/hist2D.Integral())
        hist = hist2D.ProjectionX(f'Bprime_mass_pre_{regionMap[region]}')
        modifyOverflow(hist,bins)
        outNameTag = ''
        histFile.cd()
    elif "pNet" in histType:
        if case=="case3" or case=="case4":
            return
        elif case=="case1":
            try:
                hist = histFile.Get(f'Bprime_mass_pre_{regionMap[region]}_{histType}{shift}_1D').Clone()
            except:
                print(f'pre_{region} does not have pNet shift stored')
                return
            outNameTag = f'__pNetTtag{shift}'
        else: # case2
            try:
                hist = histFile.Get(f'Bprime_mass_pre_{regionMap[region]}_{histType}{shift}_1D').Clone() # TEMP: name cloned file, because highST vs D2 naming difference
            except:
                print(f'pre_{region} does not have pNet shift stored')
                return
            outNameTag = f'__pNetWtag{shift}'
        modifyOverflow(hist,bins)
    elif "trainUncert" in histType:
        hist = histFile.Get(f'Bprime_mass_pre_{regionMap[region]}_trainUncertfullST{shift}').Clone()
        modifyOverflow(hist,bins)
        outNameTag = f'__train{shift}'
        
    outNameTag = outNameTag.replace('Dn','Down') # naming convention in SLA is Down
    hist_out = fillHistogram(hist)
    
    if doV2:
        if (region=="V" and (case=="case1" or case=="case2")) or (region=="D" and (case=="case3" or case=="case4")):
            outFile = TFile.Open(f'{outDir}/templatesV2_Jan2025_{bins}bins{outDirTag}/templates_BpMass_ABCDnn_138fbfb{year}.root', "UPDATE")
            hist_out.SetTitle("")
            hist_out.SetName(f'BpMass_ABCDnn_138fbfb_isL_{tag[case]}_V2__major{outNameTag}')
            hist_out.Write(f'BpMass_ABCDnn_138fbfb_isL_{tag[case]}_V2__major{outNameTag}', TObject.kOverwrite)
            outFile.Close()
    else:
        outFile = TFile.Open(f'{outDir}/templates{region}_Jan2025_{bins}bins{outDirTag}/templates_BpMass_ABCDnn_138fbfb{year}.root', "UPDATE")
        #if region=="highST":
        #    print(hist_out.Integral())
        #    print(f'BpMass_ABCDnn_138fbfb_isL_{tag[case]}_{region}__major{outNameTag}')
        hist_out.SetTitle("")
        hist_out.SetName(f'BpMass_ABCDnn_138fbfb_isL_{tag[case]}_{region}__major{outNameTag}')
        hist_out.Write(f'BpMass_ABCDnn_138fbfb_isL_{tag[case]}_{region}__major{outNameTag}', TObject.kOverwrite)
        outFile.Close()
    histFile.Close()
    if "Nominal" in histType:
        histFile2.Close()
        

histList = ["Nominal", "pNet", "trainUncert"]

for case in ["case1", "case2", "case3", "case4"]:
    #createHist(case, "B", "Nominal", "")
    #createHist(case, "BV", "Nominal", "")
    for histType in histList:
        if histType == "Nominal":
            shiftList = [""]
        else:
            shiftList =	["Up", "Dn"]
        for shift in shiftList:
            createHist(case, "D", histType, shift)
            createHist(case, "V", histType, shift)
