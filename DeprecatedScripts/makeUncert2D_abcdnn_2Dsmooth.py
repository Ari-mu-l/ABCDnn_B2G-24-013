import ROOT
import os
import json
import numpy as np
import array

ROOT.TH2.SetDefaultSumw2(True) # Question: is this needed?
ROOT.gROOT.SetBatch(True) # suppress histogram display
ROOT.gStyle.SetOptStat(0) # no stat box

# model selection
model_case14 = '22' #'22'
model_case23 = '33'

rootDir_case14 = f'logBpMlogST_mmd1_case14_random{model_case14}'
rootDir_case23 = f'logBpMlogST_mmd1_case23_random{model_case23}'

# histogram settings
#bin_lo_BpM = 0
#bin_hi_BpM = 4000
#bin_lo_ST = 0
#bin_hi_ST = 5000
#Nbins_BpM = 800
#Nbins_ST  = 100

#bin_lo_BpM = 300
#bin_hi_BpM = 2600
#bin_lo_ST = 0
#bin_hi_ST = 1600
#Nbins_BpM = 460
#Nbins_ST  = 32

bin_lo_BpM = 300
bin_hi_BpM = 3000
bin_lo_ST = 0
bin_hi_ST = 2000
Nbins_BpM = 540
Nbins_ST  = 40

bin_lo_BpM_eff = 400
bin_hi_BpM_eff = 2500
bin_lo_ST_eff = 0
bin_hi_ST_eff = 1500
Nbins_BpM_eff = 420
Nbins_ST_eff = 30

validationCut = 850
statCutoff = 0 #10

unblind_BpM = 700
unblind_ST = 850

rebinX = 2 #4 for 2016 (105bins) and 2 for full run2 (210bins)
rebinY = 1 #1

Nbins_BpM_actual = int(Nbins_BpM/rebinX)
Nbins_ST_actual = int(Nbins_ST/rebinY)
Nbins_BpM_eff = int(Nbins_BpM_eff/rebinX)
Nbins_ST_eff = int(Nbins_ST_eff/rebinY)

year = '' # '', '_2016', '_2016APV'
varyBinSize = True

plotDir ='2D_plots_2Dsmooth/'
if not os.path.exists(plotDir):
    os.makedirs(plotDir)

yBinLowEdges = {}

def getNewBinning(hist, region):
    # get bin list
    Nx = hist.GetNbinsX()
    Ny = hist.GetNbinsY()
    Ncut = hist.GetYaxis().FindFixBin(validationCut)
    
    yBinLowEdge = [[0] for i in range(Nx+1)]
    yBinFewestLen = Ny+1
    yBinFewestNo = 0
    if region=="D":
        for i in range(1,Nx+1):
            beforeValCut = True
            j = 1
            while j<=Ny:
                #if hist.GetBinError(i,j)/hist.GetBinContent(i,j)<0.1:
                #if hist.GetBinContent(i,j)<0.0001:
                if hist.GetBinContent(i,j)<0.1:
                    nEvt = hist.GetBinContent(i,j)
                    nEvtErr = hist.GetBinError(i,j)**2
                    #while j+1<=Ny and nEvt<0.0001:
                    while j+1<=Ny and nEvt<0.1:
                        j+=1
                        nEvt+=hist.GetBinContent(i,j)
                        nEvtErr+=hist.GetBinError(i,j)**2
                if beforeValCut and hist.GetYaxis().GetBinLowEdge(j)>=validationCut:
                    yBinLowEdge[i-1].append(validationCut)
                    beforeValCut = False
                    j = hist.GetYaxis().FindFixBin(validationCut+1)+1
                else:
                    yBinLowEdge[i-1].append(hist.GetYaxis().GetBinLowEdge(j))
                    j+=1

            if len(yBinLowEdge[i-1])<yBinFewestLen and len(yBinLowEdge[i-1])>4:
                yBinFewestLen = len(yBinLowEdge[i-1])
                yBinFewestNo = i-1
        if yBinLowEdge[yBinFewestNo][-1]!=bin_hi_ST:
            yBinLowEdge[yBinFewestNo].append(bin_hi_ST)
            #print(hist.GetXaxis().GetBinLowEdge(i))
            #print(yBinLowEdge[i-1])
    else:
        for i in range(1,Nx+1):
            j = 1
            while j<=Ncut-1:
                #if hist.GetBinContent(i,j)<0.0001:
                if hist.GetBinContent(i,j)<0.1:
                    nEvt = hist.GetBinContent(i,j)
                    nEvtErr = hist.GetBinError(i,j)**2
                    #while j+1<=Ncut-1 and nEvt<0.0001:
                    while j+1<=Ncut-1 and nEvt<0.1:
                        j+=1
                        nEvt+=hist.GetBinContent(i,j)
                        nEvtErr+=hist.GetBinError(i,j)**2

                yBinLowEdge[i-1].append(hist.GetYaxis().GetBinLowEdge(j))
                j+=1
            if len(yBinLowEdge[i-1])<yBinFewestLen and len(yBinLowEdge[i-1])>4:
                yBinFewestLen = len(yBinLowEdge[i-1])
                yBinFewestNo = i-1
        if yBinLowEdge[yBinFewestNo][-1]!=validationCut:
            yBinLowEdge[yBinFewestNo].append(validationCut)
        
    print(yBinLowEdge[yBinFewestNo])
    return yBinLowEdge[yBinFewestNo]

    
def modifyBinning(hist, yBinLowEdge):
    # merge bins with the shortest binning
    hist_out = ROOT.TH2D('', '', Nbins_BpM_actual, bin_lo_BpM, bin_hi_BpM, len(yBinLowEdge)-1, array.array('d',yBinLowEdge))

    for i in range(1,Nbins_BpM_actual+1):
        for j in range(1,len(yBinLowEdge)):
            nEvt = 0
            nEvtErr = 0
            start_bin = hist.GetYaxis().FindFixBin(yBinLowEdge[j-1]+1)
            if j<len(yBinLowEdge)-1:
                end_bin = hist.GetYaxis().FindFixBin(yBinLowEdge[j]+1)
            else:
                end_bin = hist.GetYaxis().FindFixBin(bin_hi_ST+1)
            for k in range(start_bin, end_bin):
                nEvt+=hist.GetBinContent(i,k)
                nEvtErr+=hist.GetBinError(i,k)**2
            hist_out.SetBinContent(i,j,nEvt)
            hist_out.SetBinError(i,j,np.sqrt(nEvtErr))

    return hist_out
        
        
def modifyOverflow2D(hist):
    ## Julie comment: ROOT counts bins from 1, I think we need to be using 1 in place of 0 here
    ## Julie comment: I think we actually need to do the entire right EDGE and the entire top EDGE...
    # top edge should be [imass,Nbins_ST] as the bin number, where imass runs 1 through Nbins_BpM_actual
    Nbins_X = hist.GetNbinsX()
    Nbins_Y = hist.GetNbinsY()
    for imass in range(1,Nbins_X+1):
        newtotal = hist.GetBinContent(imass,Nbins_Y)+hist.GetBinContent(imass,Nbins_Y+1)
        newError = np.sqrt(hist.GetBinError(imass,Nbins_Y)**2+hist.GetBinError(imass,Nbins_Y+1)**2)
        hist.SetBinContent(imass,Nbins_Y,newtotal)
        hist.SetBinContent(imass,Nbins_Y+1,0)
        hist.SetBinError(imass,Nbins_Y,newError)
        hist.SetBinError(imass,Nbins_Y+1,0)
    # right edge should be [Nbins_BpM_actual,ist], where ist runs 1 through Nbins_ST
    for ist in range(1,Nbins_Y+1):
        newtotal = hist.GetBinContent(Nbins_X,ist)+hist.GetBinContent(Nbins_X+1,ist)
        newError = np.sqrt(hist.GetBinContent(Nbins_X,ist)**2+hist.GetBinContent(Nbins_X+1,ist)**2)
        hist.SetBinContent(Nbins_X,ist,newtotal)
        hist.SetBinContent(Nbins_X+1,ist,0)
        hist.SetBinError(Nbins_X,ist,newError)
        hist.SetBinError(Nbins_X+1,ist,0)
        
def getNormalizedTgtPreHists(histFile, histTag, getTgt=True):
    # outputs rebinned histograms. No need to call rebin if this function is called
    hist_pre = histFile.Get(f'BpMST_pre_{histTag}').Clone(f'BpMST_pre_{histTag}')
    hist_pre.RebinX(rebinX)
    hist_pre.RebinY(rebinY)
    modifyOverflow2D(hist_pre)
    hist_pre.Scale(1/hist_pre.Integral(1,999,1,999))
    
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
        hist_tgt.Scale(1/hist_tgt.Integral(1,999,1,999))

        return hist_tgt, hist_pre
    else:
        return hist_pre


alphaFactors = {}
if bin_lo_BpM == 0:
    with open(f'alphaRatio_factors{year}_2Dpadfull.json',"r") as alphaFile:
        alphaFactors = json.load(alphaFile)
else:
    with open(f'alphaRatio_factors{year}_2Dpad.json',"r") as alphaFile:
        alphaFactors = json.load(alphaFile)

counts = {}
with open(f'counts{year}_2Dpad.json',"r") as countsFile:
    counts = json.load(countsFile)
    
def getAlphaRatioTgtPreHists(histFile, histTag, case, getTgt=True):
    # outputs rebinned histograms. No need to call rebin if this function is called
    #TEMP: remove after the naming convention is changed in getAlphaRatio
    if histTag=="D2":
        region = "highST"
    else:
        region = histTag
        
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

    for region in ["D", "V"]: # , "D2"]:
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
    for region in ["D", "B"]:
        hist_tgt, hist_pre = getAlphaRatioTgtPreHists(histFileIn, f'{region}', case)

        if region=="D":
            hist_pre_modified = modifyBinning(hist_pre,yBinLowEdges[f'{case}_{region}'])
            hist_tgt_modified = modifyBinning(hist_tgt,yBinLowEdges[f'{case}_{region}'])

        # plot ABCDnn prediction
        c1 = ROOT.TCanvas(f'c1_{region}', f'ST_ABCDnn vs BpM_ABCDnn in region {region}', 900, 600)
        
        hist_pre.SetTitle(f'ST_ABCDnn vs BpM_ABCDnn in region {region}')
        hist_pre.GetXaxis().SetTitle('B mass (GeV)')
        hist_pre.GetYaxis().SetTitle('ST (GeV)')
        hist_pre.Draw("COLZ")
        c1.SaveAs(f'{plotDir}BpMST_ABCDnn_{region}_{case}.png')
        c1.Close()

        c1_mod = ROOT.TCanvas(f'c1_{region}_mod', f'ST_ABCDnn vs BpM_ABCDnn in region {region}', 900, 600)

        hist_pre_modified.SetTitle(f'ST_ABCDnn vs BpM_ABCDnn in region {region}')
        hist_pre_modified.GetXaxis().SetTitle('B mass (GeV)')
        hist_pre_modified.GetYaxis().SetTitle('ST (GeV)')
        hist_pre_modified.Draw("COLZ")
        c1_mod.SaveAs(f'{plotDir}BpMST_ABCDnn_modified_{region}_{case}.png')
        c1_mod.Close()

        
        # plot target hist for non-blinded cases in V,D,VhighST
        # blind for D and highST case1 and 2
        blind = False
        if year=='' and (region=='D' or region=='D2'):
            if case=='case1' or case=='case2':
                blind = True

        if not blind:
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
        hist_trainUncert = histFileOut.Get(f'BpMST_trainUncert{STrange}_D')
        #hist_trainUncert.SetTitle(f'Percentage training uncertainty from {STrange} ({case})')
        hist_trainUncert.SetTitle(f'')
        hist_trainUncert.GetXaxis().SetTitle('B mass (GeV)')
        hist_trainUncert.GetYaxis().SetTitle('ST (GeV)')
        hist_trainUncert.GetYaxis().SetRangeUser(400,1500)
        hist_trainUncert.GetZaxis().SetRangeUser(-1.0,2.0)
        hist_trainUncert.Draw("COLZ")
        c3.SaveAs(f'{plotDir}BpMST_trainUncertPercent{STrange}_{case}.png')
        c3.Close()

    #for region in ["B", "D", "V", "BV"]: # general
    #for region in ["B", "D", "V"]: # year-by-year gof
    for region in ["D","B"]:
        # plot 2D correction maps
        c4 = ROOT.TCanvas(f'c4_{case}_{region}', f'Percentage correction from {region} ({case})', 900, 600)
        hist_Correct = histFileOut.Get(f'BpMST_Correct{region}').Clone(f'BpMST_Correct{region}_Copy')
        #if (case=="case1" or case=="case2") and region=="D": # change in plotting made for ANv7
        #    unblind_BpM_bin = hist_Correct.GetXaxis().FindFixBin(unblind_BpM)
        #    unblind_ST_bin = hist_Correct.GetYaxis().FindFixBin(unblind_ST)
        #    for i in range(unblind_BpM_bin, Nbins_BpM_actual+1):
        #        for j in range(unblind_ST_bin, Nbins_ST_actual+1):
        #            hist_Correct.SetBinContent(i,j,0)
        hist_Correct.SetTitle(f'Percentage correction from {region} ({case})')
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
            hist_Corrected.Scale(alphaFactors[case][region]["prediction"]/hist_Corrected.Integral(1,999))
        else:
            hist_Corrected.Scale(counts[case][region]["major"]/hist_Corrected.Integral(1,999))
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
            #if region=="D":
            #    corrType="B"
            #else:
            #    corrType="BV"
            c7 = ROOT.TCanvas(f'c7_{case}_{region}', f'Bprime_mass_ABCDnn correction on {region} from {region} ({case})', 600, 600)
            hist1D = histFileOut.Get(f'Bprime_mass_pre_Correct{region}from{region}')
            hist1D.SetTitle(f'Bprime_mass_ABCDnn correction on {region} from {region} map ({case})')
            hist1D.GetXaxis().SetTitle('B mass (GeV)')
            hist1D.Draw("HIST")
            c7.SaveAs(f'{plotDir}Bprime_mass_ABCDnn_1Dcorrect{region}from2D{region}_{region}_{case}.png')
            c7.Close()

        # plot corrected region D BpM (2D)
        c8 = ROOT.TCanvas(f'c8_{case}_{region}', f'BpMST_ABCDnn corrected with 2D ({case})', 900, 600)
        hist_Corrected2D = histFileOut.Get(f'BpMST_pre_{region}_withCorrect{region}').Clone()
        hist_Corrected2D.SetTitle(f'BpMST_ABCDnn corrected with {region} 2D map ({case})') # fix lowST label
        hist_Corrected2D.GetXaxis().SetTitle('B mass (GeV)')
        hist_Corrected2D.GetYaxis().SetTitle('ST (GeV)')
        hist_Corrected2D.Draw("COLZ")
        c8.SaveAs(f'{plotDir}BpMST_ABCDnn_correctedfrom{region}_{case}.png')
        c8.Close()

def smoothAndTruncate(hist_pre_pad, uncertType, case, region, yBinLowEdge): # uncertType: nom, corrUp/Dn, trainUp/Dn, pNetUp/Dn
    # SMOOTH
    hist_pre2D_out = hist_pre_pad.Clone(f'BpMST_pre_{case}_{uncertType}_{region}_full')
    if region=="D":
        #if case=="case1" or case=="case2":
        if case=="case1" or case=="case2":
            hist_pre_pad.Smooth(ntimes=1,option="k3a")
            hist_pre_pad.Smooth(ntimes=1,option="k3a")
        else:
            hist_pre_pad.Smooth(ntimes=1,option="k5a") #k5b
            hist_pre_pad.Smooth(ntimes=1,option="k5a")
    else:
        if (case=="case1"):
            hist_pre_pad.Smooth(ntimes=1,option="k5a") # k7b
            hist_pre_pad.Smooth(ntimes=1,option="k5a")
            #hist_pre_pad.Smooth(ntimes=1,option="k5a")
            #hist_pre_pad.Smooth(ntimes=1,option="k5a") # k7b
        else:
            hist_pre_pad.Smooth(ntimes=1,option="k5a") # k7b
            hist_pre_pad.Smooth(ntimes=1,option="k5a")
            #hist_pre_pad.Smooth(ntimes=1,option="k5a") # k7b
        #hist_pre_pad.Smooth(ntimes=1,option="k5a")
        #hist_pre_pad.Smooth(ntimes=1,option="k5a")
        #hist_pre_pad.Smooth(ntimes=1,option="k5a")
        #hist_pre_pad.Smooth(ntimes=1,option="k5a")
        
    #hist_pre_pad.Smooth(ntimes=1,option="k5a")
    #if case=="case1" or case=="case2":
    #    hist_pre_pad.Smooth(ntimes=1,option="k3a")
    #hist_pre_pad.Smooth(ntimes=1,option="k5a")
    #hist_pre_pad.Smooth(ntimes=1,option="k5a")
    
    #if (case=="case1" or case=="case2"):
    #    hist_pre_pad.Smooth(ntimes=1,option="k5a")
    #    if case=="case1":
    #        hist_pre_pad.Smooth(ntimes=1,option="k5a")
       # hist_pre_pad.Smooth(ntimes=1,option="k5a"
    #if "corr" in uncertType or "train" in uncertType:
    #    hist_pre_pad.Smooth(ntimes=1,option="k5a")
    #    hist_pre_pad.Smooth(ntimes=1,option="k5a")
    #    hist_pre_pad.Smooth(ntimes=1,option="k5a")
    #    hist_pre_pad.Smooth(ntimes=1,option="k5a")
    #if uncertType!="nom":
    #    hist_pre_pad.Smooth(ntimes=1,option="k5a")

    # Take only the needed part
    #hist_pre = ROOT.TH2D(f'BpMST_pre_{case}_{uncertType}_{region}_truncate', "BpM_vs_ST", Nbins_BpM_eff, bin_lo_BpM_eff, bin_hi_BpM_eff, Nbins_ST_eff, bin_lo_ST_eff, bin_hi_ST_eff)
    hist_pre = ROOT.TH2D(f'BpMST_pre_{case}_{uncertType}_{region}_truncate', "BpM_vs_ST", Nbins_BpM_eff, bin_lo_BpM_eff, bin_hi_BpM_eff, len(yBinLowEdge)-1, array.array('d',yBinLowEdge))
    
    newLowBinBpM  = hist_pre_pad.GetXaxis().FindFixBin(bin_lo_BpM_eff+1)
    newHighBinBpM = hist_pre_pad.GetXaxis().FindFixBin(bin_hi_BpM_eff-1)
    #newHighBinST  = hist_pre_pad.GetYaxis().FindFixBin(bin_hi_ST_eff-1)

    # set negative bins to 0
    for i in range(1,Nbins_BpM_actual+1):
        for j in range(1,hist_pre_pad.GetNbinsY()+1):
            if hist_pre_pad.GetBinContent(i,j)<0:
                hist_pre_pad.SetBinContent(i,j,0)

    # add ST lower pads back in: No ST lower pads

    # add ST upper pads back in
    #for imass in range(1,Nbins_BpM_actual+1):
    #    newtotal = 0
    #    newError = 0
    #    for ist in range(newHighBinST,hist_pre_pad.GetNbinsY()+1):
    #        newtotal+=hist_pre_pad.GetBinContent(imass,ist)
    #        newError+=hist_pre_pad.GetBinError(imass,ist)**2
    #    hist_pre_pad.SetBinContent(imass,newHighBinST,newtotal)
    #    hist_pre_pad.SetBinError(imass,newHighBinST,np.sqrt(newError))

    # add BpM right pads back in
    for ist in range(1,hist_pre_pad.GetNbinsY()+1):
        newtotal = 0
        newError = 0
        for imass in range(newHighBinBpM,Nbins_BpM_actual+1):
            newtotal+=hist_pre_pad.GetBinContent(imass,ist)
            newError+=hist_pre_pad.GetBinError(imass,ist)**2
            
        hist_pre_pad.SetBinContent(newHighBinBpM,ist,newtotal)
        hist_pre_pad.SetBinError(newHighBinBpM,ist,np.sqrt(newError))

        
    # truncate BpM left pad and add the rest
    for i in range(1, Nbins_BpM_eff+1):
        for j in range(1,hist_pre_pad.GetNbinsY()+1):
            BpM_value = hist_pre.GetXaxis().GetBinCenter(i)
            ST_value =	hist_pre.GetYaxis().GetBinCenter(j)
            bin_i = hist_pre_pad.GetXaxis().FindFixBin(BpM_value)
            bin_j = hist_pre_pad.GetYaxis().FindFixBin(ST_value)
            hist_pre.SetBinContent(i,j,hist_pre_pad.GetBinContent(bin_i,bin_j))
            hist_pre.SetBinError(i,j,hist_pre_pad.GetBinError(bin_i,bin_j))
            
    # make sure that only lowST events are considered
    if ("V" in region) and (case=="case1" or case=="case2"):
        for i in range(1,Nbins_BpM_eff+1):
            for j in range(hist_pre.GetYaxis().FindFixBin(validationCut),hist_pre.GetNbinsY()+1):
                #print(hist_pre.GetYaxis().FindFixBin(validationCut))
                hist_pre.SetBinContent(i,j,0)
                hist_pre.SetBinError(i,j,0)
                #exit()
    
    hist_pre1D_out = hist_pre.ProjectionX(f'1D_output_{case}_{uncertType}_{region}')

    return hist_pre2D_out, hist_pre1D_out
        
def addHistograms(histFileIn, histFileOut, case):
    ##############
    # Correction #
    ##############
    for region in ["D","V","B"]:
    #for region in ["D", "V", "B","BV"]: #general
    #for region in ["D", "V", "B"]: # year-by-year gof
        hist_tgt, hist_pre = getAlphaRatioTgtPreHists(histFileIn, f'{region}', case)

        if case=="case1" or case=="case2":
            hist_pre_original = hist_pre.Clone(f'pre_{case}')
            
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
                #hist_tgt_partner, hist_pre_partner = getAlphaRatioTgtPreHists(histFilePartner, f'{region}', 'case4') # trying original
                hist_tgt_partner, hist_pre_partner = getAlphaRatioTgtPreHists(histFileIn, 'B', 'case1') # fill the hole with B
            else: # case2 partners with case3
                histFilePartner = ROOT.TFile.Open(f'{rootDir}/hists_ABCDnn_case3_BpM{bin_lo_BpM}to{bin_hi_BpM}ST{bin_lo_ST}to{bin_hi_ST}_{Nbins_BpM}bins{Nbins_ST}bins_pNet{year}.root', 'READ')
                #hist_tgt_partner, hist_pre_partner = getAlphaRatioTgtPreHists(histFilePartner, f'{region}', 'case3')
                hist_tgt_partner, hist_pre_partner = getAlphaRatioTgtPreHists(histFileIn, 'B', 'case2') # fill the hole with B

            unblind_BpM_bin = hist_tgt_partner.GetXaxis().FindFixBin(unblind_BpM)
            unblind_ST_bin = hist_tgt_partner.GetYaxis().FindFixBin(unblind_ST)

            # set case1/2 upper right corner to case3/4
            for i in range(unblind_BpM_bin, Nbins_BpM_actual+1):
                for j in range(unblind_ST_bin, Nbins_ST_actual+1):
                    hist_tgt.SetBinContent(i, j, hist_tgt_partner.GetBinContent(i,j))
                    hist_pre.SetBinContent(i, j, hist_pre_partner.GetBinContent(i,j))

            histFilePartner.Close()
        ##########################################

        if case=="case1" or case=="case2":
            # cand1 = getNewBinning(hist_pre_original,region)
            # cand2 = getNewBinning(hist_pre,region)

            # if len(cand1)>len(cand2):
            #     yBinLowEdges[f'{case}_{region}'] = cand2
            # else:
            #     yBinLowEdges[f'{case}_{region}'] = cand1OA
                
            yBinLowEdges[f'{case}_{region}'] = getNewBinning(hist_pre_original,region)
        else:
            yBinLowEdges[f'{case}_{region}'] = getNewBinning(hist_pre,region)
            
        #yBinLowEdges[f'{case}_{region}'] = getNewBinning(hist_pre,region)

        #yBinLowEdges[f'{case}_{region}'] = getNewBinning(hist_tgt,region)
            
        hist_pre_modified = modifyBinning(hist_pre,yBinLowEdges[f'{case}_{region}'])
        hist_tgt_modified = modifyBinning(hist_tgt,yBinLowEdges[f'{case}_{region}'])
        
        hist_Correction = hist_tgt_modified.Clone(f'BpMST_Correct{region}_{case}')
        hist_Correction.Add(hist_pre_modified, -1.0)
        hist_Correction.Divide(hist_pre_modified)

        for i in range(1,Nbins_BpM_actual+1):
            for j in range(1,hist_Correction.GetNbinsY()+1):
                # set bin error to 0, so that the application correctly reflects the propogated change in bin error
                hist_Correction.SetBinError(i,j,0)
                # no correction on low stat bins
                if hist_tgt_modified.GetBinContent(i,j)<statCutoff:
                    hist_Correction.SetBinContent(i,j,-1) # reduce ABCDnn when mnr overpredicts
                #elif hist_tgt_modified.GetBinError(i,j)!=0 and hist_tgt_modified.GetBinContent(i,j)/hist_tgt_modified.GetBinError(i,j)<0.2:
                    #print(hist_tgt_modified.GetBinContent(i,j), hist_Correction.GetBinContent(i,j))
                #if hist_pre_modified.GetBinContent(i,j)==0:
                #    hist_Correction.SetBinContent(i,j,0)
                #if hist_pre.GetBinContent(i,j)<=0:
                #    hist_Correction.SetBinContent(i,j,0)
                #if hist_tgt.GetBinContent(i,j)<=0.000000001 and hist_Correction.GetBinContent(i,j)>10:
                #    hist_Correction.SetBinContent(i,j,0)

        histFileOut.cd()
        hist_Correction.Write(f'BpMST_Correct{region}')
        print(f'Saved BpMST_Correct{region} to {case}')

    ###################
    # training uncert #
    ###################
    # use A,B,C to calculate train uncert with fullST, lowST,highST
    for STrange in ["fullST"]: #,"lowST","highST"]:
        for applicationRegion in ["D", "V"]:

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
            hist_tgtABC_raw = ROOT.TH2D(f'BpMST_tgt{STrange}_{case}_{applicationRegion}', "BpM_vs_ST", Nbins_BpM_actual, bin_lo_BpM, bin_hi_BpM, Nbins_ST_actual, bin_lo_ST, bin_hi_ST)
            hist_preABC_raw = ROOT.TH2D(f'BpMST_pre{STrange}_{case}_{applicationRegion}', "BpM_vs_ST", Nbins_BpM_actual, bin_lo_BpM, bin_hi_BpM, Nbins_ST_actual, bin_lo_ST, bin_hi_ST)

            for region in ["A","B","C"]: #["A", "B", "C"]:
            #for region in ["B"]:
                hist_tgt = histFileIn.Get(f'BpMST_dat_{region}').Clone(f'dat_{region}')
                hist_mnr = histFileIn.Get(f'BpMST_mnr_{region}').Clone(f'mnr_{region}')
                hist_pre = histFileIn.Get(f'BpMST_pre_{region}').Clone(f'pre_{region}')

                hist_tgt.Add(hist_mnr, -1.0)

                hist_tgt.RebinX(rebinX)
                hist_pre.RebinX(rebinX)

                hist_tgt.RebinY(rebinY)
                hist_pre.RebinY(rebinY)

                hist_tgtABC_raw.Add(hist_tgt)
                hist_preABC_raw.Add(hist_pre)

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

            modifyOverflow2D(hist_tgtABC_raw)
            modifyOverflow2D(hist_preABC_raw)

            hist_tgtABC = modifyBinning(hist_tgtABC_raw,yBinLowEdges[f'{case}_{applicationRegion}']) # train uncert from full ST
            hist_preABC = modifyBinning(hist_preABC_raw,yBinLowEdges[f'{case}_{applicationRegion}'])

            hist_devABC = hist_tgtABC.Clone(f'percentage deviation_{case}_{applicationRegion}')
            #hist_preOriginal = hist_preABC.Clone('pre_original')

            #hist_tgtABC.Scale(1/hist_tgtABC.Integral())
            hist_preABC.Scale(hist_tgtABC.Integral(0,999,0,999)/hist_preABC.Integral(0,999,0,999))

            # PrecentageDiff = (hist_tgt - hist_pre)/hist_pre
            hist_devABC.Add(hist_preABC, -1.0)
            hist_devABC.Divide(hist_preABC)

            for i in range(1,Nbins_BpM_actual+1):
                for j in range(1,hist_devABC.GetNbinsY()+1):
                    hist_devABC.SetBinError(i,j,0) # set bin error to 0, so that it acts as a pure scale factor

                    #if hist_preABC.GetBinContent(i,j)<=0.1 or hist_tgtABC.GetBinContent(i,j)<=0.1:
                    if hist_preABC.GetBinContent(i,j)<=0.00000001 or hist_tgtABC.GetBinContent(i,j)<=0.00000001:
                        hist_devABC.SetBinContent(i,j,0)

                    if hist_devABC.GetBinContent(i,j)>50:
                        print("Attention required: large uncertainty spotted. Might be due to low stats.")
                        print(f'The uncert value is: {hist_devABC.GetBinContent(i,j)}')

            histFileOut.cd()
            hist_devABC.Write(f'BpMST_trainUncert{STrange}_{applicationRegion}')
            print(f'Saved BpMST_trainUncert{STrange}_{applicationRegion} to {case}')


# add pre-correction histograms
def histBeforeCorrection(histFileIn, histFileOut, corrType, region, case):
    hist_tgt_pad_raw, hist_pre_pad_raw = getAlphaRatioTgtPreHists(histFileIn, region, case)

    hist_pre_pad = modifyBinning(hist_pre_pad_raw,yBinLowEdges[f'{case}_{region}'])
    #hist_tgt_pad = modifyBinning(hist_tgt_pad_raw,yBinLowEdges[f'{case}_{region}'])

    hist_pre, hist_pre_1D = smoothAndTruncate(hist_pre_pad, 'nom', case, region, yBinLowEdges[f'{case}_{region}'])

    histFileOut.cd()
    hist_pre_1D.Write(f'Bprime_mass_pre_{region}_noCorrect')

# apply correction
def applyCorrection(histFileIn, histFileOut, corrType, region, case):
    hist_tgt_pad_raw, hist_pre_pad_raw = getAlphaRatioTgtPreHists(histFileIn, region, case)

    #if case=="case4":
    #    print(hist_pre_pad_raw.Integral())

    hist_pre_pad = modifyBinning(hist_pre_pad_raw,yBinLowEdges[f'{case}_{region}'])
    hist_tgt_pad = modifyBinning(hist_tgt_pad_raw,yBinLowEdges[f'{case}_{region}'])

    # give clone names, so that ProfileX can distinguish them
    #hist_preUp_pad = hist_pre_pad.Clone(f'CorrpreUp_{region}')
    hist_preDn_pad = hist_pre_pad.Clone(f'CorrpreDn_{region}')
    hist_cor_pad = histFileOut.Get(f'BpMST_Correct{corrType}').Clone('cor')
    
    # pre = pre + pre*correction
    hist_cor_pad.Multiply(hist_pre_pad)
    hist_pre_pad.Add(hist_cor_pad)

    # preUp = pre + pre*2*correction
    #hist_preUp_pad.Add(hist_cor_pad)
    #hist_preUp_pad.Add(hist_cor_pad)
    
    # preDn = pre + pre*0, so do nothing

    # smooth and truncate the padded histogram into [400,2500]x[0,1500]
    hist_pre, hist_pre_1D = smoothAndTruncate(hist_pre_pad, 'nom', case, region, yBinLowEdges[f'{case}_{region}'])
    #hist_preUp, hist_pre_1DUp = smoothAndTruncate(hist_preUp_pad, 'corrUp', case, region, yBinLowEdges[f'{case}_{region}'])
    hist_preDn, hist_pre_1DDn = smoothAndTruncate(hist_preDn_pad, 'corrDn', case, region, yBinLowEdges[f'{case}_{region}'])

    hist_pre_1DUp = hist_pre_1D.Clone(f'hist_pre_corrUp_{region}_{case}')
    hist_del_1D = hist_pre_1D.Clone(f'hist_pre_del_{region}_{case}')
    hist_del_1D.Add(hist_pre_1DDn,-1.0)
    hist_pre_1DUp.Add(hist_del_1D)

    hist_cor1D = hist_pre_1D.Clone('cor1D')
    hist_cor1D.Add(hist_pre_1DDn, -1.0) # corrected - original
    hist_cor1D.Divide(hist_pre_1DDn)

    histFileOut.cd()

    # Save 1D histograms for template plots
    hist_pre_1D.Write(f'Bprime_mass_pre_{region}_withCorrect{corrType}')
    hist_pre_1DUp.Write(f'Bprime_mass_pre_{region}_withCorrect{corrType}Up')
    hist_pre_1DDn.Write(f'Bprime_mass_pre_{region}_withCorrect{corrType}Dn')
    hist_cor1D.Write(f'Bprime_mass_pre_Correct{region}from{corrType}')

    # Save 2D histograms for sanity check
    hist_pre.Write(f'BpMST_pre_{region}_withCorrect{corrType}')
    #hist_preUp.Write(f'BpMST_pre_{region}_withCorrect{corrType}Up')
    hist_preDn.Write(f'BpMST_pre_{region}_withCorrect{corrType}Dn')


def applyTrainUncert(histFileIn, histFileOut, region, case):
    hist_pre_pad_raw =  histFileOut.Get(f'BpMST_pre_{region}_withCorrect{region}').Clone(f'trainPre_{region}_{case}') # shift on the corrected hist
    
    hist_pre_pad = modifyBinning(hist_pre_pad_raw,yBinLowEdges[f'{case}_{region}'])
    
    #hist_preUp_pad = hist_pre_pad.Clone(f'TrainpreUp_{region}_{case}')
    hist_preDn_pad = hist_pre_pad.Clone(f'TrainpreDn_{region}_{case}')
    hist_pre_1D = histFileOut.Get(f'Bprime_mass_pre_{region}_withCorrect{region}').Clone(f'Nominal_{region}_{case}_1D')
    hist_pre_1DUp = hist_pre_1D.Clone(f'TrainpreUp_{region}_{case}_1D')
    
    hist_trainUncert_pad = histFileOut.Get(f'BpMST_trainUncertfullST_{region}').Clone('trainUncert')
    
    # pre = pre +/- pre*trainUncert
    hist_trainUncert_pad.Multiply(hist_pre_pad)
    #hist_preUp_pad.Add(hist_trainUncert_pad, 1.0)
    hist_preDn_pad.Add(hist_trainUncert_pad, -1.0)
    
    # smooth and truncate the padded histogram into [400,2500]x[0,1500]
    #hist_preUp, hist_pre_1DUp = smoothAndTruncate(hist_preUp_pad, 'trainUp', case, region, yBinLowEdges[f'{case}_{region}'])
    hist_preDn, hist_pre_1DDn = smoothAndTruncate(hist_preDn_pad, 'trainDn', case, region, yBinLowEdges[f'{case}_{region}'])

    # 2D smoothing might act funky when seeing a bunch of very large bins, so get trainUp from trainDn
    # del_shift = pre - dn
    # up = pre + del_shift
    hist_pre_1D.Add(hist_pre_1DDn,-1.0)
    hist_pre_1DUp.Add(hist_pre_1D,1.0)

    histFileOut.cd()
    hist_pre_1DUp.Write(f'Bprime_mass_pre_{region}_trainUncertfullSTUp')
    hist_pre_1DDn.Write(f'Bprime_mass_pre_{region}_trainUncertfullSTDn')


def applypNet(histFileIn, histFileOut, region, case):
    hist_pre_pad_raw   = histFileIn.Get(f'BpMST_pre_{region}').Clone('pNetpre')
    hist_preUp_pad_raw = histFileIn.Get(f'BpMST_pre_{region}_pNetUp').Clone('pNetUp')
    hist_preDn_pad_raw = histFileIn.Get(f'BpMST_pre_{region}_pNetDn').Clone('pNetDn')

    hist_pre_pad_raw.RebinX(rebinX)
    hist_preUp_pad_raw.RebinX(rebinX)
    hist_preDn_pad_raw.RebinX(rebinX)

    hist_pre_pad_raw.RebinY(rebinY)
    hist_preUp_pad_raw.RebinY(rebinY)
    hist_preDn_pad_raw.RebinY(rebinY)
    
    modifyOverflow2D(hist_pre_pad_raw)
    modifyOverflow2D(hist_preUp_pad_raw)
    modifyOverflow2D(hist_preDn_pad_raw)

    hist_pre_pad   = modifyBinning(hist_pre_pad_raw,yBinLowEdges[f'{case}_{region}'])
    hist_preUp_pad = modifyBinning(hist_preUp_pad_raw,yBinLowEdges[f'{case}_{region}'])
    hist_preDn_pad = modifyBinning(hist_preDn_pad_raw,yBinLowEdges[f'{case}_{region}'])

    # Upshift = (Upshifted - original)/original
    # Dnshift = (Dnshifted - original)/original
    hist_preUp_pad.Add(hist_pre_pad,-1.0)
    hist_preDn_pad.Add(hist_pre_pad,-1.0)

    hist_preUp_pad.Divide(hist_pre_pad)
    hist_preDn_pad.Divide(hist_pre_pad)

    # set bin error to 0, so that it acts like a pure scale factor
    for i in range(1,Nbins_BpM_actual+1):
        for j in range(1,hist_preUp_pad.GetNbinsY()+1):
            hist_preUp_pad.SetBinError(i,j,0)
            hist_preDn_pad.SetBinError(i,j,0)

    #modifyOverflow2D(hist_pre_pad)
    hist_pre_pad_corrected_raw = histFileOut.Get(f'BpMST_pre_{region}_withCorrect{region}').Clone('pNetpre') # shift on the corrected hist
    #hist_pre_pad_corrected_raw.Scale(alphaFactors[case][region]["prediction"]/hist_pre_pad_corrected_raw.Integral())

    hist_pre_pad_corrected = modifyBinning(hist_pre_pad_corrected_raw,yBinLowEdges[f'{case}_{region}'])

    # shifted = shift*original + original (allow both shape and yield to change)
    hist_preUp_pad.Multiply(hist_pre_pad_corrected)
    hist_preDn_pad.Multiply(hist_pre_pad_corrected)

    hist_preUp_pad.Add(hist_pre_pad_corrected)
    hist_preDn_pad.Add(hist_pre_pad_corrected)

    # smooth and truncate the padded histogram into [400,2500]x[0,1500]
    hist_preUp, hist_pre_1DUp = smoothAndTruncate(hist_preUp_pad, 'pNetUp', case, region, yBinLowEdges[f'{case}_{region}'])
    hist_preDn, hist_pre_1DDn = smoothAndTruncate(hist_preDn_pad, 'pNetDn', case, region, yBinLowEdges[f'{case}_{region}'])
    
    #hist_pre_1DUp = hist_preUp.ProjectionX(f'Bprime_mass_pre_{region}_pNetUp_1D')
    #hist_pre_1DDn = hist_preDn.ProjectionX(f'Bprime_mass_pre_{region}_pNetDn_1D')

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
        #applyCorrection(histFileIn, histFileOut, region, region, case) # Used until preapproval action 5
        #if case=='case1':
            #if region=='D':
            #    applyCorrection(histFileIn, histFileOut, 'B', region, case)
            #else:
            #    applyCorrection(histFileIn, histFileOut, 'BV', region, case)
        #else:
            #applyCorrection(histFileIn, histFileOut, region, region, case) # only correct case1 with B. The rest still corrected by V
        applyCorrection(histFileIn, histFileOut, region, region, case)
        applyTrainUncert(histFileIn, histFileOut, region, case)
        if case=="case1" or case=="case2":
            applypNet(histFileIn, histFileOut, region, case)

        histBeforeCorrection(histFileIn, histFileOut, region, region, case)

    applyCorrection(histFileIn, histFileOut, "B", "B", case)

    #for region in ['B', 'BV']: # general
    #for region in ['B']: # year-by-year
    #    applyCorrection(histFileIn, histFileOut, region, region, case)
    #    applyCorrection(histFileIn, histFileOut, region, region, case)
    
    ## plot histograms
    plotHists2D_Separate(histFileIn, histFileOut, case)

    histFileIn.Close()
    histFileOut.Close()

#plotHists2D_All() # this function needs some work. Complains about merging hists with diff bins
