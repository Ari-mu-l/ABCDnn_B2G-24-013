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

doV2 = False #IMPORTANT: REMEMBER TO TURN ON AND OFF!!
withoutCorrection = False
withFit = False
separateUncertCases = True

if withoutCorrection:
    outDirTag = '_noCorrection'
else:
    #outDirTag = f'BtargetHoleCorrBTrain_smooth_rebin{year}_dynamicST_smoothBUncert'
    #outDirTag = f'BtargetHoleCorrBTrain_smooth_rebin{year}_dynamicST_2DsmoothUncert' #1D
    outDirTag = f'BtargetHoleCorrBTrain_smooth_rebin{year}_dynamicST'
    #outDirTag = f'BtargetHoleCorrABCpABCTrain_2Dsmooth_rebin{year}' #2D

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
        
    ##histFile = TFile.Open(f'{rootDir}/hists_ABCDnn_{case}_BpM0to4000ST0to5000_800bins100bins_pNet{year}_modified.root', "READ")
    ##histFile = TFile.Open(f'{rootDir}/hists_ABCDnn_{case}_BpM300to2600ST0to1600_460bins32bins_pNet{year}_modified.root', "READ")
    histFile = TFile.Open(f'{rootDir}/hists_ABCDnn_{case}_BpM300to3000ST0to2000_540bins40bins_pNet{year}_modified.root', "READ")
    #histFile = TFile.Open(f'{rootDir}/hists_ABCDnn_{case}_BpM400to2500ST0to1500_420bins30bins_pNet{year}_modified.root', "READ")
    
    if "Nominal" in histType:
        if "B" in region:
            hist = histFile.Get(f'Bprime_mass_pre_{regionMap[region]}_withCorrect{regionMap[region]}').Clone()
        else:
            #hist = histFile.Get(f'Bprime_mass_pre_{regionMap[region]}_withCorrect{regionMap[region]}').Clone() # up to preApp action 5
            if case=="case1":
                if region=="D":
                    hist = histFile.Get(f'Bprime_mass_pre_D_withCorrectD').Clone()
                    print(f'Bprime_mass_pre_D_withCorrectB being used')
                elif region=="V":
                    hist = histFile.Get(f'Bprime_mass_pre_V_withCorrectV').Clone()
                else:
                    hist = histFile.Get(f'Bprime_mass_pre_{regionMap[region]}_withCorrect{regionMap[region]}').Clone()
            else:
                hist = histFile.Get(f'Bprime_mass_pre_{regionMap[region]}_withCorrect{regionMap[region]}').Clone()
        modifyOverflow(hist,bins)
        outNameTag = ''
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
    elif "correct" in histType:
        if region=="V" or region=="D":
            if case=="case1":
                if region=="D":
                    hist = histFile.Get(f'Bprime_mass_pre_D_withCorrectD{shift}').Clone()
                elif region=="V":
                    hist = histFile.Get(f'Bprime_mass_pre_V_withCorrectV{shift}').Clone()
                else:
                    hist = histFile.Get(f'Bprime_mass_pre_{regionMap[region]}_withCorrect{regionMap[region]}{shift}').Clone()
            else:
                hist = histFile.Get(f'Bprime_mass_pre_{regionMap[region]}_withCorrect{region}{shift}').Clone()
        else:
            print("Please update region in the code. Exit...")
            exit()
        modifyOverflow(hist,bins)
        outNameTag = f'__correct{shift}'
    elif "smooth" in histType:
        hist = histFile.Get(f'Bprime_mass_pre_{regionMap[region]}_withCorrect{regionMap[region]}_Smoothed').Clone()
        modifyOverflow(hist,bins)
        outNameTag = f'__smooth{shift}' # smoothUp and smoothDown are the same for now
        
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

#histList = ["Nominal", "pNet", "trainUncert", "correct"]
histList = ["Nominal", "pNet", "trainUncert", "correct", "smooth"]

def addWithoutCorrection(case, region):
    if case=="case1" or case=="case4":
        rootDir = rootDir_case14
    else:
        rootDir = rootDir_case23

    histFile = TFile.Open(f'{rootDir}/hists_ABCDnn_{case}_BpM300to3000ST0to2000_540bins40bins_pNet{year}_modified.root', "READ")

    hist = histFile.Get(f'Bprime_mass_pre_{regionMap[region]}_noCorrect')
    modifyOverflow(hist,bins)
    hist_out = fillHistogram(hist)

    if doV2:
        if (region=="V" and (case=="case1" or case=="case2")) or (region=="D" and (case=="case3" or case=="case4")):
            outFile = TFile.Open(f'{outDir}/templatesV2_Jan2025_{bins}bins{outDirTag}/templates_BpMass_ABCDnn_138fbfb{year}.root', "UPDATE")
            hist_out.SetTitle("")
            hist_out.SetName(f'BpMass_ABCDnn_138fbfb_isL_{tag[case]}_V2__major')
            hist_out.Write(f'BpMass_ABCDnn_138fbfb_isL_{tag[case]}_V2__major', TObject.kOverwrite)
            outFile.Close()
    else:
        outFile = TFile.Open(f'{outDir}/templates{region}_Jan2025_{bins}bins{outDirTag}/templates_BpMass_ABCDnn_138fbfb{year}.root', "UPDATE")
        hist_out.SetTitle("")
        hist_out.SetName(f'BpMass_ABCDnn_138fbfb_isL_{tag[case]}_{region}__major')
        hist_out.Write(f'BpMass_ABCDnn_138fbfb_isL_{tag[case]}_{region}__major', TObject.kOverwrite)
        outFile.Close()
    histFile.Close()
    
    

if withoutCorrection:
    for case in ["case1", "case2", "case3", "case4"]:
        addWithoutCorrection(case, "D")
        addWithoutCorrection(case, "V")
else:
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
                #if year=='':
                #createHist(case, "V", histType, shift)
                ##createHist(case, "highST", histType, shift)
