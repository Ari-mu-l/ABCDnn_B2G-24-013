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
bins = 210
year = '' # '', '_2016'

doV2 = True
withFit = False
separateUncertCases = True

# store parameters in dictionaries
# region: A, B, C, D, X, Y
params = {"case1":{"tgt":{},"pre":{}},
          "case2":{"tgt":{},"pre":{}},
          "case3":{"tgt":{},"pre":{}},
          "case4":{"tgt":{},"pre":{}},
          } 
pred_uncert = {"Description":"Prediction uncertainty",
               "case1":{},
               "case2":{},
               "case3":{},
               "case4":{}
               }

# region: "D", "V"
normalization = {
                 "case1":{},
                 "case2":{},
                 "case3":{},
                 "case4":{}
                 }

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

regionMap = {"A": "A", "B": "B", "C": "C", "D": "D", "V":"V", "highST": "D2"} # TEMP: named highST as D2 in the intermediate file
def createHist(case, region, histType): # histType: Nominal, pNetUp, pNetDn, trainUncertUp, trainUncertDn
    histFile = TFile.Open(f'hists_ABCDnn_{case}_{binlo}to{binhi}_{bins_name}_pNet.root', "READ")
    if histType=="Nominal":
        hist = histFile.Get(f'Bprime_mass_pre_{regionMap[region]}').Clone() # TEMP: named highST as D2 in the intermediate file
        outNameTag = ""
        modifyOverflow(hist,bins)
        hist.Scale(alphaFactors[case][region]["prediction"]/hist.Integral())
    elif "pNet" in histType:
        if case=="case3" or case=="case4":
            return
        elif case=="case1":
            try:
                hist = histFile.Get(f'Bprime_mass_pre_{regionMap[region]}_{histType}').Clone() 
            except:
                print(f'pre_{region} does not have pNet shift stored')
                return
            outNameTag = histType.replace('pNet', '__pNetTtag')
        else: # case2
            try:
                hist = histFile.Get(f'Bprime_mass_pre_{regionMap[region]}_{histType}').Clone() # TEMP: name cloned file, because highST vs D2 naming difference
            except:
                print(f'pre_{region} does not have pNet shift stored')
                return
            outNameTag = histType.replace('pNet', '__pNetWtag')
        modifyOverflow(hist,bins)
        hist.Scale(alphaFactors[case][region]["prediction"]/hist.Integral())
    elif "trainUncert" in histType:
        if region=="V":
            hist_shift = histFile.Get(f'Bprime_mass_trainUncertlowST').Clone()
        elif region=="highST":
            hist_shift = histFile.Get(f'Bprime_mass_trainUncerthighST').Clone()
        else:
            hist_shift = histFile.Get(f'Bprime_mass_trainUncertfullST').Clone()

        hist = histFile.Get(f'Bprime_mass_pre_{regionMap[region]}').Clone()
        
        modifyOverflow(hist,bins)
        hist.Scale(alphaFactors[case][region]["prediction"]/hist.Integral())    

        hist_shift.Multiply(hist)
        if "Up" in histType:
            hist.Add(hist_shift, 1.0)
        else:
            hist.Add(hist_shift, -1.0)
        
        outNameTag = histType.replace('trainUncert', '__train')
        

    
    outNameTag = outNameTag.replace('Dn','Down') # naming convention in SLA is Down

    hist_out = fillHistogram(hist)
    
    if doV2:
        if (region=="V" and (case=="case1" or case=="case2")) or (region=="D" and (case=="case3" or case=="case4")):
            outFile = TFile.Open(f'{outDir}/templatesV2_Jan2025_{bins}bins/templates_BpMass_ABCDnn_138fbfb.root', "UPDATE")
            hist_out.Write(f'BpMass_ABCDnn_138fbfb_isL_{tag[case]}_V2__major{outNameTag}')
            outFile.Close()
    else:
        outFile = TFile.Open(f'{outDir}/templates{region}_Jan2025_{bins}bins/templates_BpMass_ABCDnn_138fbfb.root', "UPDATE")
        #if region=="highST":
        #    print(hist_out.Integral())
        #    print(f'BpMass_ABCDnn_138fbfb_isL_{tag[case]}_{region}__major{outNameTag}')
        hist_out.Write(f'BpMass_ABCDnn_138fbfb_isL_{tag[case]}_{region}__major{outNameTag}')
        outFile.Close()

histList = ["Nominal", "pNetUp", "pNetDn", "trainUncertUp", "trainUncertDn"]

for case in ["case1", "case2", "case3", "case4"]:
    for histType in histList:
        createHist(case, "D", histType)
        createHist(case, "V", histType)
        createHist(case, "V2", histType)
        #createHist(case, "highST", histType)
