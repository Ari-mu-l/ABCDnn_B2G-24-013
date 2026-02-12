import ROOT
import os
import json
import numpy as np

ROOT.TH2.SetDefaultSumw2(True) # Question: is this needed?
ROOT.gROOT.SetBatch(True) # suppress histogram display
ROOT.gStyle.SetOptStat(0) # no stat box

# model selection
model_case14 = '22' #'22'
model_case23 = '33'

rootDir_case14 = f'logBpMlogST_mmd1_case14_random{model_case14}'
rootDir_case23 = f'logBpMlogST_mmd1_case23_random{model_case23}'

# histogram settings
bin_lo_BpM = 400 #0
bin_hi_BpM = 2500
bin_lo_ST = 0
bin_hi_ST = 1500 #1530 #1500
Nbins_BpM = 420 # 2100
Nbins_ST  = 30 #18 #30
validationCut = 850
statCutoff = 0 #10

unblind_BpM = 700
unblind_ST = 850

rebinX = 2 #4 for 2016 (105bins) and 2 for full run2 (210bins)
rebinY = 1
Nbins_BpM_actual = int(Nbins_BpM/rebinX)
Nbins_ST_actual = int(Nbins_ST/rebinY)

year = '' # '', '_2016', '_2016APV'
varyBinSize = True

plotDir ='2D_plots/'
if not os.path.exists(plotDir):
    os.makedirs(plotDir)

STTag = {"fullST": "",
         "lowST":  "V",
         "highST": "2"}

STMap = {"fullST": "D",
         "lowST":  "V",
         "highST": "D2"}

def mergeBin(i, j, nEvt, hist):
    while i<Nbins_BpM_actual+1 and nEvt<1:
        i+=1 # the last bin to include
        if hist.GetBinContent(i,j)>0:
            nEvt += hist.GetBinContent(i,j)
    return i, hist.GetXaxis().GetBinLowEdge(i)

def modifyBinning(hist):
    # scan across BpM
    xbinsDict = {}
    nXbins = 0
    shortestBpM = []
    
    for j in range(1,Nbins_ST_actual+1): # bin 0 is underflow
    #for j in [10]:
        i = 0
        nEvt = 0
        xbinsList = [hist.GetXaxis().GetBinLowEdge(0)]
        while i<Nbins_BpM_actual: # TODO: consider the last bin
            i, binLowEdge = mergeBin(i, j, nEvt, hist)
            xbinsList.append(binLowEdge)
        xbinsDict[f'ST bin {j}'] = np.array(xbinsList)

        if nXbins==0 or nXbins>len(xbinsList):
            nXbins = len(xbinsList)
            shortestBpM = xbinsList

    for j in range(1,Nbins_ST_actual+1):
        binTemp = 0
        for k in range(1,nXbins): # the low edge is always the same
            print(shortestBpM[k])
            closestBin = np.argmin(np.abs(xbinsDict[f'ST bin {j}']-shortestBpM[k]))
            if binTemp==0 or closestBin>binTemp:
                binTemp = closestBin
            print(closestBin)
            exit()
    print(shortestBpM)
    exit()

        
def modifyOverflow2D(hist):
    ## Julie comment: ROOT counts bins from 1, I think we need to be using 1 in place of 0 here
    ## Julie comment: I think we actually need to do the entire right EDGE and the entire top EDGE...
    # top edge should be [imass,Nbins_ST] as the bin number, where imass runs 1 through Nbins_BpM_actual
    for imass in range(1,Nbins_BpM_actual+1):
        newtotal = hist.GetBinContent(imass,Nbins_ST_actual)+hist.GetBinContent(imass,Nbins_ST_actual+1)
        hist.SetBinContent(imass,Nbins_ST_actual,newtotal)
        hist.SetBinContent(imass,Nbins_ST_actual+1,0)
    # right edge should be [Nbins_BpM_actual,ist], where ist runs 1 through Nbins_ST
    for ist in range(1,Nbins_ST_actual+1):
        newtotal = hist.GetBinContent(Nbins_BpM_actual,ist)+hist.GetBinContent(Nbins_BpM_actual+1,ist)
        hist.SetBinContent(Nbins_BpM_actual,ist,newtotal)
        hist.SetBinContent(Nbins_BpM_actual+1,ist,0)
        

def getNormalizedTgtPreHists(histFile, histTag, getTgt=True):
    hist_pre = histFile.Get(f'BpMST_pre_{histTag}').Clone(f'BpMST_pre_{histTag}')
    hist_pre.RebinX(rebinX)
    hist_pre.RebinY(rebinY)
    modifyOverflow2D(hist_pre)
    hist_pre.Scale(1/hist_pre.Integral())
    
    if getTgt:
        hist_tgt = histFile.Get(f'BpMST_dat_{histTag}').Clone(f'BpMST_dat_{histTag}')
        hist_mnr = histFile.Get(f'BpMST_mnr_{histTag}').Clone(f'BpMST_mnr_{histTag}')
        hist_tgt.RebinX(rebinX)
        hist_mnr.RebinX(rebinX)
        hist_tgt.RebinY(rebinY)
        hist_mnr.RebinY(rebinY)
        
        # tgt = dat - mnr
        hist_tgt.Add(hist_mnr, -1.0)

        modifyOverflow2D(hist_tgt)
        hist_tgt.Scale(1/hist_tgt.Integral())

        return hist_tgt, hist_pre
    else:
        return hist_pre


alphaFactors = {}
with open(f'alphaRatio_factors{year}.json',"r") as alphaFile:
    alphaFactors = json.load(alphaFile)

counts = {}
with open(f'counts{year}.json',"r") as countsFile:
    counts = json.load(countsFile)
    
def getAlphaRatioTgtPreHists(histFile, histTag, case, getTgt=True):
    #TEMP: remove after the naming convention is changed in getAlphaRatio
    if histTag=="D2":
        region = "highST"
    else:
        region = histTag
        
    if 'pNet' in histTag:
        region = histTag[0]
        
    if getTgt:
        _, hist_pre = getNormalizedTgtPreHists(histFile, histTag, getTgt)
    else:
        hist_pre = getNormalizedTgtPreHists(histFile, histTag, getTgt)

    if region=="B" or region=="BV": # normalize B and BV with MC
        #yield_pred = counts[case][region]["major"]
        yield_pred = counts[case][region]["data"]-counts[case][region]["minor"]
    else:
        yield_pred = alphaFactors[case][region]["prediction"]
        
    hist_pre.Scale(yield_pred)
    
    if getTgt:
        hist_tgt = histFile.Get(f'BpMST_dat_{histTag}').Clone(f'BpMST_dat_{histTag}')
        hist_mnr = histFile.Get(f'BpMST_mnr_{histTag}').Clone(f'BpMST_mnr_{histTag}')
        hist_tgt.RebinX(rebinX)
        hist_mnr.RebinX(rebinX)
        hist_tgt.RebinY(rebinY)
        hist_mnr.RebinY(rebinY)
        hist_tgt.Add(hist_mnr, -1.0)
        modifyOverflow2D(hist_tgt)
                    
        return hist_tgt, hist_pre
    else:
        return hist_pre
    

def plotHists2D_All():
    histFile1 = ROOT.TFile.Open(f'{rootDir_case14}/hists_ABCDnn_case1_BpM{bin_lo_BpM}to{bin_hi_BpM}ST{bin_lo_ST}to{bin_hi_ST}_{Nbins_BpM}bins{Nbins_ST}bins_pNet{year}_modified.root', 'READ')
    histFile2 = ROOT.TFile.Open(f'{rootDir_case23}/hists_ABCDnn_case2_BpM{bin_lo_BpM}to{bin_hi_BpM}ST{bin_lo_ST}to{bin_hi_ST}_{Nbins_BpM}bins{Nbins_ST}bins_pNet{year}_modified.root', 'READ')
    histFile3 = ROOT.TFile.Open(f'{rootDir_case23}/hists_ABCDnn_case3_BpM{bin_lo_BpM}to{bin_hi_BpM}ST{bin_lo_ST}to{bin_hi_ST}_{Nbins_BpM}bins{Nbins_ST}bins_pNet{year}_modified.root', 'READ')
    histFile4 = ROOT.TFile.Open(f'{rootDir_case14}/hists_ABCDnn_case4_BpM{bin_lo_BpM}to{bin_hi_BpM}ST{bin_lo_ST}to{bin_hi_ST}_{Nbins_BpM}bins{Nbins_ST}bins_pNet{year}_modified.root', 'READ')

    for region in ["D", "V", "D2"]:
        hist_tgt1, hist_pre1 = getAlphaRatioTgtPreHists(histFile1, f'{region}', 'case1')
        hist_tgt2, hist_pre2 = getAlphaRatioTgtPreHists(histFile2, f'{region}', 'case2')
        hist_tgt3, hist_pre3 = getAlphaRatioTgtPreHists(histFile3, f'{region}', 'case3')
        hist_tgt4, hist_pre4 = getAlphaRatioTgtPreHists(histFile4, f'{region}', 'case4')

        hist_pre1.Add(hist_pre2)
        hist_pre1.Add(hist_pre3)
        hist_pre1.Add(hist_pre4)
        
        c1 = ROOT.TCanvas(f'c1_{region}', f'ST_ABCDnn vs BpM_ABCDnn in region {region}', 900, 600)
        hist_pre1.SetTitle(f'ST_ABCDnn vs BpM_ABCDnn in region {region}')
        hist_pre1.GetXaxis().SetTitle('B mass (GeV)')
        hist_pre1.GetYaxis().SetTitle('ST (GeV)')
        hist_pre1.GetXaxis().SetRangeUser(400,2500) # set viewing range
        hist_pre1.GetYaxis().SetRangeUser(400,1500)
        hist_pre1.Draw("COLZ")
        c1.SaveAs(f'{plotDir}BpMST_ABCDnn_{region}_all.png')
        c1.Close()

    histFile1.Close()
    histFile2.Close()
    histFile3.Close()
    histFile4.Close()
        
    
def plotHists2D_Separate(histFileIn, histFileOut, case):
    for region in ["D", "B"]: # high/low ST regions are just part of these plots
        hist_tgt, hist_pre = getAlphaRatioTgtPreHists(histFileIn, f'{region}', case)

        # plot ABCDnn prediction
        c1 = ROOT.TCanvas(f'c1_{region}', f'ST_ABCDnn vs BpM_ABCDnn in region {region}', 900, 600)
        
        hist_pre.SetTitle(f'ST_ABCDnn vs BpM_ABCDnn in region {region}')
        hist_pre.GetXaxis().SetTitle('B mass (GeV)')
        hist_pre.GetYaxis().SetTitle('ST (GeV)')
        hist_pre.Draw("COLZ")
        c1.SaveAs(f'{plotDir}BpMST_ABCDnn_{region}_{case}.png')
        c1.Close()
  
        # blind for D and highST case1 and 2
        blind = False
        if year=='' and (region=='D' or region=='D2'):
            if case=='case1' or case=='case2':
                blind = True

        if not blind:
            # plot target hist for non-blinded cases in V,D,VhighST
            c2 = ROOT.TCanvas(f'c2_{region}', f'ST_target vs BpM_target in region {region}', 900, 600)
            hist_tgt.SetTitle(f'ST_target vs BpM_target in region {region}')
            hist_tgt.GetXaxis().SetTitle('B mass (GeV)')
            hist_tgt.GetYaxis().SetTitle('ST (GeV)')
            hist_tgt.Draw("COLZ")
            c2.SaveAs(f'{plotDir}BpMST_target_{region}_{case}.png')
            c2.Close()

    # plot training uncertainty derived from full ST, low ST, highST
    #for STrange in ['fullST','lowST','highST']:
    for STrange in ['fullST']:
        c3 = ROOT.TCanvas(f'c3_{case}_{STrange}', f'Percentage training uncertainty from {STrange} ({case})', 900, 600)
        hist_trainUncert = histFileOut.Get(f'BpMST_trainUncert{STrange}')
        #hist_trainUncert.SetTitle(f'Percentage training uncertainty from {STrange} ({case})')
        hist_trainUncert.SetTitle(f'')
        hist_trainUncert.GetXaxis().SetTitle('B mass (GeV)')
        hist_trainUncert.GetYaxis().SetTitle('ST (GeV)')
        hist_trainUncert.GetYaxis().SetRangeUser(400,1500)
        hist_trainUncert.GetZaxis().SetRangeUser(-1.0,2.0)
        hist_trainUncert.Draw("COLZ")
        c3.SaveAs(f'{plotDir}BpMST_trainUncertPercent{STrange}_{case}.png')
        c3.Close()

    for region in ["B", "D", "V", "BV"]: # general
    #for region in ["B", "D", "V"]: # year-by-year gof
        # plot 2D correction maps
        c4 = ROOT.TCanvas(f'c4_{case}_{region}', f'Percentage correction from {region} ({case})', 900, 600)
        hist_Correct = histFileOut.Get(f'BpMST_Correct{region}').Clone(f'BpMST_Correct{region}_Copy')
        if (case=="case1" or case=="case2") and region=="D": # change in plotting made for ANv7
            unblind_BpM_bin = hist_Correct.GetXaxis().FindFixBin(unblind_BpM)
            unblind_ST_bin = hist_Correct.GetYaxis().FindFixBin(unblind_ST)
            for i in range(unblind_BpM_bin, Nbins_BpM_actual+1):
                for j in range(unblind_ST_bin, Nbins_ST_actual+1):
                    hist_Correct.SetBinContent(i,j,0)
        #hist_Correct.SetTitle(f'Percentage correction from {region} ({case})')
        hist_Correct.SetTitle(f'')
        hist_Correct.GetXaxis().SetTitle('B mass (GeV)')
        hist_Correct.GetYaxis().SetTitle('ST (GeV)')
        hist_Correct.GetYaxis().SetRangeUser(400,1500)
        hist_Correct.GetZaxis().SetRangeUser(-1.0,2.0)
        hist_Correct.Draw("COLZ")
        c4.SaveAs(f'{plotDir}BpMST_correctionPercent{region}_{case}.png')
        c4.Close()

        # plot corrected region D BpM (1D)
        c5 = ROOT.TCanvas(f'c5_{case}_{region}', f'Bprime_mass_ABCDnn corrected with 2D ({case})', 600, 600)
        hist_Corrected = histFileOut.Get(f'Bprime_mass_pre_{region}_withCorrect{region}').Clone()
        if region=="D" or region=="V":
            hist_Corrected.Scale(alphaFactors[case][region]["prediction"]/hist_Corrected.Integral())
        else:
            hist_Corrected.Scale(counts[case][region]["major"]/hist_Corrected.Integral())
        hist_Corrected.SetTitle(f'Bprime_mass_ABCDnn corrected with {region} 2D map ({case})') # fix lowST label
        hist_Corrected.GetXaxis().SetTitle('B mass (GeV)')
        hist_Corrected.Draw("HIST")
        c5.SaveAs(f'{plotDir}Bprime_mass_ABCDnn_correctedfrom{region}_{case}.png')
        c5.Close()

        # plot 1D correction derived from 2D
        c6 = ROOT.TCanvas(f'c6_{case}_{region}', f'Bprime_mass_ABCDnn correction on {region} from {region} ({case})', 600, 600)
        hist1D = histFileOut.Get(f'Bprime_mass_pre_Correct{region}from{region}')
        hist1D.SetTitle(f'Bprime_mass_ABCDnn correction on {region} from {region} map ({case})')
        hist1D.GetXaxis().SetTitle('B mass (GeV)')
        hist1D.Draw("HIST")
        c6.SaveAs(f'{plotDir}Bprime_mass_ABCDnn_1Dcorrect{region}from2D{region}_{region}_{case}.png')
        c6.Close()
        
        if case=="case1" and (region=="D" or region=="V"):
            if region=="D":
                corrType="B"
            else:
                corrType="BV"
            c7 = ROOT.TCanvas(f'c7_{case}_{region}', f'Bprime_mass_ABCDnn correction on {region} from {corrType} ({case})', 600, 600)
            hist1D = histFileOut.Get(f'Bprime_mass_pre_Correct{region}from{corrType}')
            hist1D.SetTitle(f'Bprime_mass_ABCDnn correction on {region} from {corrType} map ({case})')
            hist1D.GetXaxis().SetTitle('B mass (GeV)')
            hist1D.Draw("HIST")
            c7.SaveAs(f'{plotDir}Bprime_mass_ABCDnn_1Dcorrect{region}from2D{corrType}_{region}_{case}.png')
            c7.Close()
        
def addHistograms(histFileIn, histFileOut, case):
    ###################
    # training uncert #
    ###################
    # use A,B,C to calculate train uncert with fullST, lowST,highST
    for STrange in ["fullST"]: #,"lowST","highST"]:

        # # Alternative 1:
        # for region in ["A", "B", "C"]:
        #     hist_tgt, hist_pre = getNormalizedTgtPreHists(histFileIn, f'{region}{STTag[STrange]}')

        #     # PrecentageDiff = (hist_tgt - hist_pre)/hist_pre
        #     hist_tgt.Add(hist_pre, -1.0)
        #     hist_tgt.Divide(hist_pre)

        #     # take absolute value abs(PercentageDiff) for training uncert
        #     # add contribution to training uncert
        #     #for i in range(Nbins_BpM_actual+1):
        #     #    for j in range(Nbins_ST_actual+1):
        #     #        hist_tgt.SetBinError(i,j,0) # set bin error to 0, so that it acts as a pure scale factor
        #     #        if hist_tgt.GetBinContent(i,j)<0:
        #     #            hist_tgt.SetBinContent(i,j,-hist_tgt.GetBinContent(i,j))

        # # average over A,B,C
        # hist_tgt.Scale(1/3)
        # histFileOut.cd()
        # hist_tgt.Write(f'BpMST_trainUncert{STrange}')
        # print(f'Saved BpMST_trainUncert{STrange} to {case}')

        # Alternative 2:
        # weighted average
        hist_tgtABC = ROOT.TH2D(f'BpM_tgt{STrange}_{case}', "BpM", Nbins_BpM_actual, bin_lo_BpM, bin_hi_BpM, Nbins_ST_actual, bin_lo_ST, bin_hi_ST)
        hist_preABC = ROOT.TH2D(f'BpM_pre{STrange}_{case}', "BpM", Nbins_BpM_actual, bin_lo_BpM, bin_hi_BpM, Nbins_ST_actual, bin_lo_ST, bin_hi_ST)
        
        for region in ["B"]: #["A", "B", "C"]:
            hist_tgt = histFileIn.Get(f'BpMST_dat_{region}').Clone(f'dat_{region}')
            hist_mnr = histFileIn.Get(f'BpMST_mnr_{region}').Clone(f'mnr_{region}')
            hist_pre = histFileIn.Get(f'BpMST_pre_{region}').Clone(f'pre_{region}')
            
            hist_tgt.Add(hist_mnr, -1.0)
            
            hist_tgt.RebinX(rebinX)
            hist_pre.RebinX(rebinX)
            
            hist_tgt.RebinY(rebinY)
            hist_pre.RebinY(rebinY)
            
            hist_tgtABC.Add(hist_tgt)
            hist_preABC.Add(hist_pre)
        
        # trainB + trainAC in hole only
        # if case=="case1":
        #     hist_tgtA = histFileIn.Get(f'BpMST_dat_A').Clone(f'dat_A')
        #     hist_mnrA = histFileIn.Get(f'BpMST_mnr_A').Clone(f'mnr_A')
        #     hist_preA = histFileIn.Get(f'BpMST_pre_A').Clone(f'pre_A')

        #     hist_tgtC = histFileIn.Get(f'BpMST_dat_C').Clone(f'dat_C')
        #     hist_mnrC = histFileIn.Get(f'BpMST_mnr_C').Clone(f'mnr_C')
        #     hist_preC = histFileIn.Get(f'BpMST_pre_C').Clone(f'pre_C')

        #     hist_tgtA.Add(hist_tgtC)
        #     hist_mnrA.Add(hist_mnrC)
        #     hist_preA.Add(hist_preC)
	
        #     hist_tgtA.Add(hist_mnrA, -1.0)

        #     hist_tgtA.RebinX(rebinX)
        #     hist_preA.RebinX(rebinX)

        #     hist_tgtA.RebinY(rebinY)
        #     hist_preA.RebinY(rebinY)
            
        #     unblind_BpM_bin = hist_tgtA.GetXaxis().FindFixBin(unblind_BpM)
        #     unblind_ST_bin = hist_tgtA.GetYaxis().FindFixBin(unblind_ST)
            
        #     for i in range(unblind_BpM_bin, Nbins_BpM_actual+1):
        #         for j in range(unblind_ST_bin, Nbins_ST_actual+1):
        #             hist_tgtABC.SetBinContent(i, j, hist_tgtA.GetBinContent(i,j))
        #             hist_preABC.SetBinContent(i, j, hist_preA.GetBinContent(i,j))

        modifyOverflow2D(hist_tgtABC)
        modifyOverflow2D(hist_preABC)
        
        hist_tgtABC.ProjectionX(f'tgt_{region}_1D')
        hist_preABC.ProjectionX(f'pre_{region}_1D')

        hist_tgtABC.Scale(1/hist_tgtABC.Integral())
        hist_preABC.Scale(1/hist_preABC.Integral())

        # PrecentageDiff = (hist_tgt - hist_pre)/hist_pre
        hist_tgtABC.Add(hist_preABC, -1.0)
        hist_tgtABC.Divide(hist_preABC)

        for i in range(Nbins_BpM_actual+1):
            hist_tgtABC.SetBinError(i,0) # set bin error to 0, so that it acts as a pure scale factor

        histFileOut.cd()
        hist_tgtABC.Write(f'BpM_trainUncert{STrange}')
        print(f'Saved BpM_trainUncert{STrange} to {case}')

    ##############
    # Correction #
    ##############
    for region in ["D", "V", "B","BV"]: #general
    #for region in ["D", "V", "B"]: # year-by-year gof
        hist_tgt, hist_pre = getAlphaRatioTgtPreHists(histFileIn, f'{region}', case)
        #modifyBinning(hist_tgt)
        #exit()

        # note: case1 and 2 only apply to low bkg region
        # __________
        #|     | 3&4|
        #|     |    |
        #|      ----|
        #| Derive   |
        # ----------
        ###########################################
        if region=="D" and (case=="case1" or case=="case2"): # VR can be fully unblinded
            if case=="case1":
                histFilePartner = ROOT.TFile.Open(f'{rootDir}/hists_ABCDnn_case4_BpM{bin_lo_BpM}to{bin_hi_BpM}ST{bin_lo_ST}to{bin_hi_ST}_{Nbins_BpM}bins{Nbins_ST}bins_pNet{year}.root', 'READ')
                #hist_tgt_partner, hist_pre_partner = getAlphaRatioTgtPreHists(histFilePartner, f'{region}', 'case4')
                hist_tgt_partner, hist_pre_partner = getAlphaRatioTgtPreHists(histFileIn, 'B', 'case1') # fill the hole with B
            else: # case2 partners with case3
                histFilePartner = ROOT.TFile.Open(f'{rootDir}/hists_ABCDnn_case3_BpM{bin_lo_BpM}to{bin_hi_BpM}ST{bin_lo_ST}to{bin_hi_ST}_{Nbins_BpM}bins{Nbins_ST}bins_pNet{year}.root', 'READ')
                hist_tgt_partner, hist_pre_partner = getAlphaRatioTgtPreHists(histFilePartner, f'{region}', 'case3')

            unblind_BpM_bin = hist_tgt_partner.GetXaxis().FindFixBin(unblind_BpM)
            unblind_ST_bin = hist_tgt_partner.GetYaxis().FindFixBin(unblind_ST)

            # set case1/2 upper right corner to case3/4
            for i in range(unblind_BpM_bin, Nbins_BpM_actual+1):
                for j in range(unblind_ST_bin, Nbins_ST_actual+1):
                    hist_tgt.SetBinContent(i, j, hist_tgt_partner.GetBinContent(i,j))
                    hist_pre.SetBinContent(i, j, hist_pre_partner.GetBinContent(i,j))

            histFilePartner.Close()
        ##########################################

        hist_Correction = hist_tgt.Clone(f'BpMST_Correct{region}_{case}')
        hist_Correction.Add(hist_pre, -1.0)
        hist_Correction.Divide(hist_pre)


        for i in range(Nbins_BpM_actual+1):
            for j in range(Nbins_ST_actual+1):
                # set bin error to 0, so that the application correctly reflects the propogated change in bin error
                hist_Correction.SetBinError(i,j,0)
                # no correction on low stat bins
                if hist_tgt.GetBinContent(i,j)<=statCutoff:
                    hist_Correction.SetBinContent(i,j,-1) # TEMP: reduce ABCDnn when mnr overpredicts
                #if hist_pre.GetBinContent(i,j)<=statCutoff:
                #    hist_Correction.SetBinContent(i,j,0)

        histFileOut.cd()
        hist_Correction.Write(f'BpMST_Correct{region}')
        print(f'Saved BpMST_Correct{region} to {case}')

    
# apply correction
def applyCorrection(histFileIn, histFileOut, corrType, region, case):
    hist_tgt, hist_pre = getAlphaRatioTgtPreHists(histFileIn, region, case)

    # give clone names, so that ProfileX can distinguish them
    hist_preUp = hist_pre.Clone(f'CorrpreUp_{region}')
    hist_preDn = hist_pre.Clone(f'CorrpreDn_{region}')
    #hist_cor = histFileOut.Get(f'BpMST_Correct{region}').Clone('cor') # pre-approval response used D(V)_map to correct D(V)
    hist_cor = histFileOut.Get(f'BpMST_Correct{corrType}').Clone('cor')

    # pre = pre + pre*correction
    hist_cor.Multiply(hist_pre)
    hist_pre.Add(hist_cor)

    # count=0
    # if case=="case1" and region=="B":
    #     hist_tgt.Add(hist_pre, -1.0)
    #     for i in range(Nbins_BpM_actual):
    #         for j in range(Nbins_ST_actual):
    #             if hist_tgt.GetBinContent(i,j)!=0 and hist_cor_original.GetBinContent(i,j)!=-1:
    #                 count+=1
    #                 print("Difference:",hist_tgt.GetBinContent(i,j))
    #                 print("Target:",hist_tgt_original.GetBinContent(i,j))
    #                 print("Corrected:",hist_pre.GetBinContent(i,j))
    #                 print(f'CorrMap:{hist_cor_original.GetBinContent(i,j)}, Uncorrected:{hist_pre_original.GetBinContent(i,j)}')
    #                 print('')
    #     print(count)

    # preUp = pre + pre*2*correction
    hist_preUp.Add(hist_cor)
    hist_preUp.Add(hist_cor)

    # preDn = pre + pre*0, so do nothing

    # SMOOTH
    #hist_pre.Smooth()
    #hist_preUp.Smooth()
    #hist_preDn.Smooth()
    # Smoothing introduces negative bins. Set those bins to 0 before projection. Check with Julie
    # for i in range(Nbins_BpM_actual+1):
    #     for j in range(Nbins_ST_actual+1):
    #         if hist_pre.GetBinContent(i,j)<0:
    #             hist_pre.SetBinContent(i,j,0)
    #         if hist_preUp.GetBinContent(i,j)<0:
    #             hist_preUp.SetBinContent(i,j,0)
    #         if hist_preDn.GetBinContent(i,j)<0:
    #             hist_preDn.SetBinContent(i,j,0)

    # make sure that only lowST events are considered
    if ("V" in region) and (case=="case1" or case=="case2"):
        for i in range(Nbins_BpM_actual+1):
            for j in range(hist_preUp.GetYaxis().FindFixBin(validationCut),Nbins_ST_actual+1):
                hist_pre.SetBinContent(i,j,0)
                hist_preUp.SetBinContent(i,j,0)
                hist_preDn.SetBinContent(i,j,0)
                hist_pre.SetBinError(i,j,0)
                hist_preUp.SetBinError(i,j,0)
                hist_preDn.SetBinError(i,j,0)

    hist_pre_1DUp = hist_preUp.ProjectionX(f'Bprime_mass_pre_{region}_withCorrect{corrType}')
    hist_pre_1DDn = hist_preDn.ProjectionX(f'Bprime_mass_pre_{region}_withCorrect{corrType}Up')
    hist_pre_1D = hist_pre.ProjectionX(f'Bprime_mass_pre_{region}_withCorrect{corrType}Dn')

    # if region=="B" and case=="case1":
    #     hist_tgt_original1D = hist_tgt_original.ProjectionX()
    #     hist_tgt_original1D.Add(hist_pre_1D,-1.0)
    #     for i in range(Nbins_BpM_actual+1):
    #         if hist_tgt_original1D.GetBinContent(i)!=0:
    #             print(f'Difference:{hist_tgt_original.GetBinContent(i)}')
    #     exit()

    hist_cor1D = hist_pre_1D.Clone('cor1D')
    hist_cor1D.Add(hist_pre_1DDn, -1.0) # corrected - original
    hist_cor1D.Divide(hist_pre_1DDn)

    histFileOut.cd()

    hist_pre_1D.Write(f'Bprime_mass_pre_{region}_withCorrect{corrType}')
    hist_pre_1DUp.Write(f'Bprime_mass_pre_{region}_withCorrect{corrType}Up')
    hist_pre_1DDn.Write(f'Bprime_mass_pre_{region}_withCorrect{corrType}Dn')
    hist_cor1D.Write(f'Bprime_mass_pre_Correct{region}from{corrType}')


def applyTrainUncert(histFileIn, histFileOut, region, case):
    hist_pre = getAlphaRatioTgtPreHists(histFileIn, region, case, False)
    hist_preUp = hist_pre.Clone(f'TrainpreUp_{region}')
    hist_preDn = hist_pre.Clone(f'TrainpreDn_{region}')

    if ("V" in region) and (case=="case1" or case=="case2"):
        for i in range(Nbins_BpM_actual+1):
            for j in range(hist_preUp.GetYaxis().FindFixBin(validationCut),Nbins_ST_actual+1):
                hist_pre.SetBinContent(i,j,0)
                hist_preUp.SetBinContent(i,j,0)
                hist_preDn.SetBinContent(i,j,0)
                hist_pre.SetBinError(i,j,0)
                hist_preUp.SetBinError(i,j,0)
                hist_preDn.SetBinError(i,j,0)
    
    hist_pre = hist_pre.ProjectionX(f'Bprime_mass_pre_{region}_trainUncertfullST')
    hist_pre_1DUp = hist_preUp.ProjectionX(f'Bprime_mass_pre_{region}_trainUncertfullSTUp')
    hist_pre_1DDn = hist_preDn.ProjectionX(f'Bprime_mass_pre_{region}_trainUncertfullSTDn')

    hist_trainUncert = histFileOut.Get(f'BpM_trainUncertfullST').Clone('trainUncert')

    # pre = pre +/- pre*trainUncert
    hist_trainUncert.Multiply(hist_preUp)
    hist_preUp.Add(hist_trainUncert, 1.0)
    hist_preDn.Add(hist_trainUncert, -1.0)

    histFileOut.cd()
    hist_pre_1DUp.Write(f'Bprime_mass_pre_{region}_trainUncertfullSTUp')
    hist_pre_1DDn.Write(f'Bprime_mass_pre_{region}_trainUncertfullSTDn')


def applypNet(histFileIn, histFileOut, region, case):
    hist_pre   = histFileIn.Get(f'BpMST_pre_{region}').Clone('pNetpre')
    hist_preUp = histFileIn.Get(f'BpMST_pre_{region}_pNetUp').Clone('pNetUp') # TEMP: change to BpMST for newly run root files
    hist_preDn = histFileIn.Get(f'BpMST_pre_{region}_pNetDn').Clone('pNetDn')

    # ST binning doesn't matter. Will get integrated
    hist_pre.RebinX(rebinX)
    hist_preUp.RebinX(rebinX)
    hist_preDn.RebinX(rebinX)

    # Upshift = (Upshifted - original)/original
    # Dnshift = (Dnshifted - original)/original
    hist_preUp.Add(hist_pre,-1.0)
    hist_preDn.Add(hist_pre,-1.0)

    hist_preUp.Divide(hist_pre)
    hist_preDn.Divide(hist_pre)

    # set bin error to 0, so that it acts like a pure scale factor
    for i in range(Nbins_BpM_actual+1):
        for j in range(Nbins_ST_actual+1):
            hist_preUp.SetBinError(i,j,0)
            hist_preDn.SetBinError(i,j,0)

    modifyOverflow2D(hist_pre)
    hist_pre.Scale(alphaFactors[case][region]["prediction"]/hist_pre.Integral())

    # shifted = shift*original + original (allow both shape and yield to change)
    hist_preUp.Multiply(hist_pre)
    hist_preDn.Multiply(hist_pre)

    hist_preUp.Add(hist_pre)
    hist_preDn.Add(hist_pre)

    if region=="V" and (case=="case1" or case=="case2"):
        for i in range(Nbins_BpM_actual+1):
            for j in range(hist_preUp.GetYaxis().FindFixBin(validationCut),Nbins_ST_actual+1):
                hist_preUp.SetBinContent(i,j,0)
                hist_preDn.SetBinContent(i,j,0)
                hist_preUp.SetBinError(i,j,0)
                hist_preDn.SetBinError(i,j,0)

    hist_pre_1DUp = hist_preUp.ProjectionX(f'Bprime_mass_pre_{region}_pNetUp_1D')
    hist_pre_1DDn = hist_preDn.ProjectionX(f'Bprime_mass_pre_{region}_pNetDn_1D')

    histFileOut.cd()
    hist_pre_1DUp.Write(f'Bprime_mass_pre_{region}_pNetUp_1D')
    hist_pre_1DDn.Write(f'Bprime_mass_pre_{region}_pNetDn_1D')


for case in ['case1', 'case2', 'case3', 'case4']:
    
    if case=="case1" or case=="case4":
        rootDir = rootDir_case14
    else:
        rootDir = rootDir_case23
        
    histFileIn = ROOT.TFile.Open(f'{rootDir}/hists_ABCDnn_{case}_BpM{bin_lo_BpM}to{bin_hi_BpM}ST{bin_lo_ST}to{bin_hi_ST}_{Nbins_BpM}bins{Nbins_ST}bins_pNet{year}.root', 'READ')
    histFileOut = ROOT.TFile.Open(f'{rootDir}/hists_ABCDnn_{case}_BpM{bin_lo_BpM}to{bin_hi_BpM}ST{bin_lo_ST}to{bin_hi_ST}_{Nbins_BpM}bins{Nbins_ST}bins_pNet{year}_modified.root', 'RECREATE')
    print(f'{rootDir}/hists_ABCDnn_{case}_BpM{bin_lo_BpM}to{bin_hi_BpM}ST{bin_lo_ST}to{bin_hi_ST}_{Nbins_BpM}bins{Nbins_ST}bins_pNet{year}_modified.root')
    
    addHistograms(histFileIn, histFileOut, case)
    for region in ['D', 'V']: # general
    #for region in ['D']: # year-by-year
        applyCorrection(histFileIn, histFileOut, region, region, case) # Used until preapproval action 5
        #if case=='case1':
        #    if region=='D':
        #        applyCorrection(histFileIn, histFileOut, 'B', region, case)
        #    else:
        #        applyCorrection(histFileIn, histFileOut, 'BV', region, case)
        #    applyCorrection(histFileIn, histFileOut, region, region, case)
        #else:
        #    applyCorrection(histFileIn, histFileOut, region, region, case) # only correct case1 with B. The rest still corrected by V
        applyTrainUncert(histFileIn, histFileOut, region, case)
        if case=="case1" or case=="case2":
            applypNet(histFileIn, histFileOut, region, case)

    for region in ['B', 'BV']: # general
    #for region in ['B']: # year-by-year
        applyCorrection(histFileIn, histFileOut, region, region, case)
        applyCorrection(histFileIn, histFileOut, region, region, case)
    
    ## plot histograms
    #plotHists2D_Separate(histFileIn, histFileOut, case)

    histFileIn.Close()
    histFileOut.Close()

#plotHists2D_All() # this function needs some work. Complains about merging hists with diff bins
