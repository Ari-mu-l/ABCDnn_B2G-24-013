import ROOT
import os

ROOT.TH1.SetDefaultSumw2(True)
ROOT.gROOT.SetBatch(True) # suppress histogram display

# histogram settings
binlo = 400
binhi = 2500
bins = 2100 # 420
validationCut = 850

plotDir ='1D_plots/'
if not os.path.exists(plotDir):
    os.makedirs(plotDir)

def modifyOverflow1D(hist):
    hist.SetBinContent(bins, hist.GetBinContent(bins)+hist.GetBinContent(bins+1))
    hist.SetBinContent(bins+1, 0)

def getNormalizedTgtPreHists(histFile, histTag):
    hist_tgt = histFile.Get(f'Bprime_mass_dat_{histTag}').Clone()
    hist_mnr = histFile.Get(f'Bprime_mass_mnr_{histTag}').Clone()
    print(f'Bprime_mass_pre_{histTag}')
    hist_pre = histFile.Get(f'Bprime_mass_pre_{histTag}').Clone()

    # tgt = dat - mnr
    hist_tgt.Add(hist_mnr, -1.0)

    modifyOverflow1D(hist_tgt)
    modifyOverflow1D(hist_pre)

    hist_tgt.Scale(1/hist_tgt.Integral())
    hist_pre.Scale(1/hist_pre.Integral())

    return hist_tgt, hist_pre

def plotHists1D(case):
    histFile = ROOT.TFile.Open(f'hists_ABCDnn_{case}_{binlo}to{binhi}_{bins}_pNet.root', 'READ')
    
    for region in ["D", "V", "highST"]:
        hist_tgt, hist_pre = getNormalizedTgtPreHists(histFile, f'{region}')

        # plot prediction hist for V,D,VhighST
        c1 = ROOT.TCanvas(f'c1_{region}', f'Bprime_mass_ABCDnn in region {region}', 500, 500)
        hist_pre.Draw("COLZ")
        c1.SaveAs(f'{plotDir}Bprime_mass_ABCDnn_{region}_{case}.png')
        c1.Close()
  
        # blind for D and highST case1 and 2
        blind = False
        if region=='D' or region=='highST':
            if case=='case1' or case=='case2':
                blind = True

        if not blind:
            # plot target hist for non-blinded cases in V,D,VhighST
            c2 = ROOT.TCanvas(f'c2_{region}', f'Bprime_mass_target in region {region}', 500, 500)
            hist_tgt.Draw("COLZ")
            c2.SaveAs(f'{plotDir}Bprime_mass_target_{region}_{case}.png')
            c2.Close()

    # plot training uncertainty derived from full ST
    c3 = ROOT.TCanvas(f'c3_{case}', f'Percentage training uncertainty from full ST ({case})', 500, 500)
    hist_trainUncertFullST = histFile.Get('Bprime_mass_trainUncert')
    hist_trainUncertFullST.Draw("COLZ")
    c3.SaveAs(f'{plotDir}Bprime_mass_trainUncertPercentFullST_{case}.png')
    c3.Close()

    # plot training uncertainty derived from low ST
    c4 = ROOT.TCanvas(f'c4_{case}', f'Percentage training uncertainty from low ST ({case})', 500, 500)
    hist_trainUncertLowST = histFile.Get('Bprime_mass_trainUncertlowST')
    hist_trainUncertLowST.Draw("COLZ")
    c4.SaveAs(f'{plotDir}Bprime_mass_trainUncertPercentLowST_{case}.png')
    c4.Close()

    # plot training uncertainty derived from high ST
    c5 = ROOT.TCanvas(f'c5_{case}', f'Percentage training uncertainty from high ST ({case})', 500, 500)
    hist_trainUncertHighST = histFile.Get('Bprime_mass_trainUncerthighST')
    hist_trainUncertHighST.Draw("COLZ")
    c5.SaveAs(f'{plotDir}Bprime_mass_trainUncertPercentHighST_{case}.png')
    c5.Close()

    # plot VR correction
    c6 = ROOT.TCanvas(f'c6_{case}', f'Percentage correction from V ({case})', 500, 500)
    hist_valCorrect = histFile.Get('Bprime_mass_CorrectV')
    hist_valCorrect.Draw("COLZ")
    c6.SaveAs(f'{plotDir}Bprime_mass_correctionPercentV_{case}.png')
    c6.Close()
    
    # plot highST correction
    c7 = ROOT.TCanvas(f'c7_{case}', 'Percentage correction from highST ({case})', 500, 500)
    hist_highSTCorrect= histFile.Get('Bprime_mass_CorrecthighST')
    hist_highSTCorrect.Draw("COLZ")
    c7.SaveAs(f'{plotDir}Bprime_mass_correctionPercentHighST_{case}.png')
    c7.Close()

    histFile.Close()


STTag = {"fullST": "",
         "lowST": "V",
         "highST": "2"}
    
for case in ["case1", "case2", "case3", "case4"]:
    histFile = ROOT.TFile.Open(f'hists_ABCDnn_{case}_{binlo}to{binhi}_{bins}_pNet.root', 'UPDATE')
    ###################
    # training uncert #
    ###################
    # use A,B,C to calculate train uncert with fullST, lowST,highST
    for STrange in ["fullST","lowST","highST"]:
        hist_trainUncert = ROOT.TH1D(f'Bprime_mass_trainUncert{STrange}_{case}', "Bprime_mass", bins, binlo, binhi)
        for region in ["A", "B", "C"]:
            hist_tgt, hist_pre = getNormalizedTgtPreHists(histFile, f'{region}{STTag[STrange]}')
            
            # PrecentageDiff = (hist_tgt - hist_pre)/hist_pre
            hist_tgt.Add(hist_pre, -1.0)
            hist_tgt.Divide(hist_pre)

            # take absolute value abs(PercentageDiff) for training uncert
            # allow negative values for V and highST corrections
            for i in range(bins+1):
                if hist_tgt.GetBinContent(i)<0:
                    hist_tgt.SetBinContent(i,-hist_tgt.GetBinContent(i))
            # add contribution to training uncert
            hist_trainUncert.Add(hist_tgt, 1.0)
        # average over A,B,C
        hist_trainUncert.Scale(1/3)
        hist_trainUncert.Write(f'Bprime_mass_trainUncert{STrange}')
        print(f'Saved Bprime_mass_trainUncert{STrange} to {case}')
            
    for region in ["V", "D2"]:
        hist_Correction = ROOT.TH1D(f'Bprime_mass_Correct{region}_{case}', "Bprime_mass", bins, binlo, binhi)
        hist_tgt, hist_pre = getNormalizedTgtPreHists(histFile, f'{region}')
        
        # PrecentageDiff = (hist_tgt - hist_pre)/hist_pre
        hist_tgt.Add(hist_pre, -1.0)
        hist_tgt.Divide(hist_pre)
        
        hist_Correction.Add(hist_tgt, 1.0)
        
        if region=="D2": # only keep derivation from case3,4 in region V
            # assgin the same correction for case2 and 3
            if case=="case3":
                hist_Correction.Write(f'Bprime_mass_Correct{region}')
                print(f'Saved Bprime_mass_Correct{region} to {case}')
                histFilePartner = ROOT.TFile.Open(f'hists_ABCDnn_case2_{binlo}to{binhi}_{bins}_pNet.root', 'UPDATE')
                hist_Correction.Write(f'Bprime_mass_Correct{region}')
                print(f'Saved Bprime_mass_Correct{region} to case2')
                histFilePartner.Close()
            # assign the same correction for case1 and 4
            if case=="case4":
                hist_Correction.Write(f'Bprime_mass_Correct{region}')
                print(f'Saved Bprime_mass_Correct{region} to {case}')
                histFilePartner = ROOT.TFile.Open(f'hists_ABCDnn_case1_{binlo}to{binhi}_{bins}_pNet.root', 'UPDATE')
                hist_Correction.Write(f'Bprime_mass_Correct{region}')
                print(f'Saved Bprime_mass_Correct{region} to case1')
                histFilePartner.Close()
        else: # write correction for case1,2,3,4 derived from region V
            hist_Correction.Write(f'Bprime_mass_Correct{region}')
            print(f'Saved Bprime_mass_Correct{region} to {case}')

    histFile.Close()

# plot histograms
#for case in ["case1", "case2", "case3", "case4"]:
#    plotHists1D(case)
