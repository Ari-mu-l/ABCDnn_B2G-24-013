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
    if region=="D" or region=="B":
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
elif bin_lo_BpM == 400:
    with open(f'alphaRatio_factors{year}.json',"r") as alphaFile:
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


def smoothAndTruncate(hist_pre_pad, uncertType, case, region, yBinLowEdge, smooth): # uncertType: nom, corrUp/Dn, trainUp/Dn, pNetUp/Dn
    # SMOOTH
    hist_pre2D_out = hist_pre_pad.Clone(f'BpMST_pre_{case}_{uncertType}_{region}_full_{smooth}')
    #hist_pre2D_out.SetDirectory(0)

    if smooth:
        if region=="D":
            if case=="case1" or case=="case2":
                hist_pre2D_out.Smooth(ntimes=1,option="k5a")
            else:
                hist_pre2D_out.Smooth(ntimes=1,option="k7b") #k5b
                hist_pre2D_out.Smooth(ntimes=1,option="k7b")
        else:
            if (case=="case1") or (case=="case2"):
                hist_pre2D_out.Smooth(ntimes=1,option="k5a") # k7b
                hist_pre2D_out.Smooth(ntimes=1,option="k5a")
            else:
                hist_pre2D_out.Smooth(ntimes=1,option="k7b") # k7b
                hist_pre2D_out.Smooth(ntimes=1,option="k7b")

    # Take only the needed part
    #hist_pre = ROOT.TH2D(f'BpMST_pre_{case}_{uncertType}_{region}_truncate', "BpM_vs_ST", Nbins_BpM_eff, bin_lo_BpM_eff, bin_hi_BpM_eff, Nbins_ST_eff, bin_lo_ST_eff, bin_hi_ST_eff)
    hist_pre = ROOT.TH2D(f'BpMST_pre_{case}_{uncertType}_{region}_truncate', "BpM_vs_ST", Nbins_BpM_eff, bin_lo_BpM_eff, bin_hi_BpM_eff, len(yBinLowEdge)-1, array.array('d',yBinLowEdge))
    
    newLowBinBpM  = hist_pre2D_out.GetXaxis().FindFixBin(bin_lo_BpM_eff+1)
    newHighBinBpM = hist_pre2D_out.GetXaxis().FindFixBin(bin_hi_BpM_eff-1)
    #newHighBinST  = hist_pre_pad.GetYaxis().FindFixBin(bin_hi_ST_eff-1)

    # set negative bins to 0
    for i in range(1,Nbins_BpM_actual+1):
        for j in range(1,hist_pre2D_out.GetNbinsY()+1):
            if hist_pre2D_out.GetBinContent(i,j)<0:
                hist_pre2D_out.SetBinContent(i,j,0)

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
    for ist in range(1,hist_pre2D_out.GetNbinsY()+1):
        newtotal = 0
        newError = 0
        for imass in range(newHighBinBpM,Nbins_BpM_actual+1):
            newtotal+=hist_pre2D_out.GetBinContent(imass,ist)
            newError+=hist_pre2D_out.GetBinError(imass,ist)**2
            
        hist_pre2D_out.SetBinContent(newHighBinBpM,ist,newtotal)
        hist_pre2D_out.SetBinError(newHighBinBpM,ist,np.sqrt(newError))

        
    # truncate BpM left pad and add the rest
    for i in range(1, Nbins_BpM_eff+1):
        for j in range(1,hist_pre2D_out.GetNbinsY()+1):
            BpM_value = hist_pre.GetXaxis().GetBinCenter(i)
            ST_value =	hist_pre.GetYaxis().GetBinCenter(j)
            bin_i = hist_pre2D_out.GetXaxis().FindFixBin(BpM_value)
            bin_j = hist_pre2D_out.GetYaxis().FindFixBin(ST_value)
            hist_pre.SetBinContent(i,j,hist_pre2D_out.GetBinContent(bin_i,bin_j))
            hist_pre.SetBinError(i,j,hist_pre2D_out.GetBinError(bin_i,bin_j))
            
    # make sure that only lowST events are considered
    if ("V" in region) and (case=="case1" or case=="case2"):
        for i in range(1,Nbins_BpM_eff+1):
            for j in range(hist_pre.GetYaxis().FindFixBin(validationCut),hist_pre.GetNbinsY()+1):
                #print(hist_pre.GetYaxis().FindFixBin(validationCut))
                hist_pre.SetBinContent(i,j,0)
                hist_pre.SetBinError(i,j,0)
    
    hist_pre1D_out = hist_pre.ProjectionX(f'1D_output_{case}_{uncertType}_{region}_{smooth}')

    return hist_pre2D_out, hist_pre1D_out

def correctC1C2Map():
    histFileName = f'{rootDir_case14}/hists_ABCDnn_case1_BpM{bin_lo_BpM}to{bin_hi_BpM}ST{bin_lo_ST}to{bin_hi_ST}_{Nbins_BpM}bins{Nbins_ST}bins_pNet{year}_modified.root'
    histFileIn = ROOT.TFile.Open(histFileName, 'READ')
    histFilePartnerIn = ROOT.TFile.Open(histFileName.replace('_case1_','_case4_'), 'READ')
    histFileOut = ROOT.TFile.Open(histFileName.replace('.root','_corrected.root'), 'RECREATE')
    
    # save 2D training hists
    #allHists = [k.GetName() for k in histFileIn.GetListOfKeys() if ('Correct' not in k.GetName()) and ('BpMST' in k.GetName())]
    corrHists = ['BpMST_CorrectD','BpMST_CorrectV','BpMST_CorrectB']
    trainHists = ['BpMST_trainUncertfullST_D','BpMST_trainUncertfullST_V']

    # save 2D train uncert
    for histName in trainHists:
        hist = histFileIn.Get(histName)
        hist.SetDirectory(0)
        histFileOut.cd()
        hist.Write(histName)

    # compute difference between Case3(4) B and V
    histPart_B = histFilePartnerIn.Get('BpMST_CorrectB').Clone('BpMST_CorrectB')
    factor = histFilePartnerIn.Get('BpMST_CorrectD').Clone('BpMST_CorrectD')

    #factor = (histPart_V - histPart_B)/histPart_B
    factor.Add(histPart_B,-1.0)
    factor.Divide(histPart_B)

    print(factor.GetNbinsX(),factor.GetNbinsY())
    print(histPart_B.GetNbinsX(),histPart_B.GetNbinsY())
    unblind_BpM_bin = factor.GetXaxis().FindFixBin(unblind_BpM)
    unblind_ST_bin = factor.GetYaxis().FindFixBin(unblind_ST)
    for i in range(1,unblind_BpM_bin):
        for j in range(1,unblind_ST_bin):
            factor.SetBinContent(i,j,0)

    histFileOut.cd()
    factor.Write('BV_factor')

    c1 = ROOT.TCanvas('BV_factor_Case1','BV_factor_Case1',900,600)
    factor.GetXaxis().SetTitle('B mass (GeV)')
    factor.GetYaxis().SetTitle('ST (GeV)')
    factor.Draw("COLZ")
    c1.SaveAs(f'{plotDir}BV_factor_Case1.png')

    #for histName in corrHists:
    #    hist = histFileIn.Get(histName)
        

    histFileIn.Close()
    histFileOut.Close()

    print(f'{histFileName.replace(".root","_corrected.root")} created')
    exit()

correctC1C2Map()
