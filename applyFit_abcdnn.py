import os
import numpy as np
from ROOT import *
import json

gStyle.SetOptFit(1)
gStyle.SetOptStat(0)
gROOT.SetBatch(True) # suppress histogram display
TH1.SetDefaultSumw2(True)

# define the fit function
# f(x) = 2 pdf(x) * cdf(a x)
# f(x) = 2/w pdf((x-xi)/2) * cdf(a*(x-xi)/2)
# f(x) = f(x) + ax^2 + bx

binlo = 400
binhi = 2500
bins = 2100 # will be changed to 420 later in the code for the creation of histograms

doV2 = False
withFit = False
separateUncertCases = True

# store parameters in dictionaries
# region: A, B, C, D, X, Y
# param: param0, param1, ..., lastbin,
params = {#"case14":{"tgt":{},"pre":{}},
          #"case23":{"tgt":{},"pre":{}},
          "case1":{"tgt":{},"pre":{}},
          "case2":{"tgt":{},"pre":{}},
          "case3":{"tgt":{},"pre":{}},
          "case4":{"tgt":{},"pre":{}},
          } 
pred_uncert = {"Description":"Prediction uncertainty",
               #"case14":{},
               #"case23":{},
               "case1":{},
               "case2":{},
               "case3":{},
               "case4":{}
               }

# region: "D", "V"
normalization = {#"case14":{},
                 #"case23":{},
                 "case1":{},
                 "case2":{},
                 "case3":{},
                 "case4":{}
                 }

tag = {#"case14": "allWlep",
       #"case23": "allTlep",
       "case1" : "tagTjet",
       "case2" : "tagWjet",
       "case3" : "untagTlep",
       "case4" : "untagWlep",
       }

uncerRegion = {"case1": ["A", "C"],
               "case2": ["A", "B", "C"],
               "case3": ["A","B", "C"],
               "case4": ["A", "C"]}

outDir = '/uscms/home/xshen/nobackup/alma9/CMSSW_13_3_3/src/vlq-BtoTW-SLA/makeTemplates'
#######
# Fit #
#######

# skewNorm_6
#fitFunc = "[3] * (2/[1]) * ( TMath::Exp(-((x-[2])/[1])*((x-[2])/[1])/2) / TMath::Sqrt(2*TMath::Pi()) ) * ( (1 + TMath::Erf([0]*((x-[2])/[1])/TMath::Sqrt(2)) ) / 2 ) + [4] * (x/2500) + [5] * TMath::Power(x/2500, 2) + [6] * TMath::Power(x/2500, 3) + [7] * TMath::Power(x/2500, 4) + [8] * TMath::Power(x/2500, 5) + [9] * TMath::Power(x/2500, 6)"
#nparams = 10

# skewNorm_4 with initialization 1 worked the best for both case14 and case 23
fitFunc = "[3] * (2/[1]) * ( TMath::Exp(-((x-[2])/[1])*((x-[2])/[1])/2) / TMath::Sqrt(2*TMath::Pi()) ) * ( (1 + TMath::Erf([0]*((x-[2])/[1])/TMath::Sqrt(2)) ) / 2 ) + [4] * x + [5] * x * x + [6] * x * x * x + [7] * x * x * x * x"
nparams=8

# skewNorm_cubic_2
#fitFunc = "[3] * (2/[1]) * ( TMath::Exp(-((x-[2])/[1])*((x-[2])/[1])/2) / TMath::Sqrt(2*TMath::Pi()) ) * ( (1 + TMath::Erf([0]*((x-[2])/[1])/TMath::Sqrt(2)) ) / 2 ) + [4] * x + [5] * x * x + [6] * x * x * x"
#nparams = 7

# crysalBall
#fitFunc="crystalball"
#nparams=5

def fit_and_plot(hist, plotname, case, doPlot):
    fit = TF1(f'fitFunc', fitFunc, binlo, binhi, nparams)
    #fit.SetParameters(5, 400, 500, 50)
    #fit.SetParameters(5, 400, 500, 50, 0.000001, 0.00000001) #skewNorm_cubic
    fit.SetParameters(5, 400, 500, 50, 0.0000001, 0.00000001, 0.000000000001) # skewNorm_4. AN v3
    #if case=="case4":
    #    fit.SetParameters(5, 400, 500, 50, 0.0000001, 0.000000001, 0.0000000001)
    #    fit.SetParLimits(1, 0, 600)
    #    fit.SetParLimits(2, 0, 600)
    #else:
    #    fit.SetParameters(5, 400, 500, 50, 0.0000001, 0.000000001, 0.000000000001)
    #fit.SetParameters(0.1, 500, 200, -2.5, 100000000) #crystalball
    
    hist.Scale(1/hist.Integral())
    hist_target = hist.Clone()
    hist.Fit(fit, "E")

    if doPlot:
        c = TCanvas("c", "c", 800, 800)
        pad1 = TPad("hist_plot", "hist_plot", 0.05, 0.3, 1, 1)
        pad1.SetBottomMargin(0.01) #join upper and lower plot
        pad1.SetLeftMargin(0.1)
        pad1.Draw()
        pad1.cd()

        latex = TLatex()
        latex.SetNDC()

        hist.Draw("E")
        latex.DrawText(0.6, 0.4, f'{case}')

        c.cd()
        pad2 = TPad("ratio_plot", "ratio_plot", 0.05, 0.05, 1, 0.3)
        pad2.SetTopMargin(0.01)
        pad2.SetBottomMargin(0.2)
        pad2.SetLeftMargin(0.1)
        pad2.SetGrid()
        pad2.Draw()
        pad2.cd()

        line = TF1("line", "0", binlo, binhi, 0)

        #hist_ratio = TRatioPlot(hist) # outputs 'Pull'. Set y-axis title 'Pull' and range[-3,3]
        hist_fill = hist_target.Clone()
        hist_fill.Reset()
        for i in range(bins):
            x = hist_fill.GetBinCenter(i+1)
            hist_fill.SetBinContent(i+1, fit.Eval(x))
        #hist_ratio = hist_fill/hist_target
        hist_ratio = (hist_fill - hist_target)/hist_target

        # get uncertainty band
        hist_Up = hist_target.Clone()
        hist_Dn = hist_target.Clone()
        hist_Up.Reset()
        hist_Dn.Reset()

        params_arr = np.zeros((nparams,3))
        for i in range(nparams):
            params_arr[i][0] = fit.GetParameter(i)
            params_arr[i][1] = fit.GetParameter(i)+fit.GetParError(i)
            params_arr[i][2] = fit.GetParameter(i)-fit.GetParError(i)

        # get possible combinations of params
        grids = np.array(np.meshgrid(params_arr[0],params_arr[1],params_arr[2],params_arr[3],params_arr[4],params_arr[5],params_arr[6],params_arr[7])).T.reshape(-1,8)[1:] # grids[0] is the nominal
        Ngrids = len(grids)

        for i in range(bins):
            x   = hist_fill.GetBinCenter(i+1)
            nom = fit.Eval(x)
            shifted_arr = np.zeros(Ngrids)
            for j in range(Ngrids):
                fitshift = fit.Clone()
                fitshift.SetParameters(grids[j])
                shifted_arr[j] = fitshift.Eval(x)
            up_arr = shifted_arr[shifted_arr>nom]
            dn_arr = shifted_arr[shifted_arr<nom]
            hist_Up.SetBinContent(i+1,up_arr.max())
            hist_Dn.SetBinContent(i+1,dn_arr.min())

        hist_ratioUp = (hist_Up-hist_target)/hist_target
        hist_ratioDn = (hist_Dn-hist_target)/hist_target

        hist_ratioUp.SetTitle("")
        hist_ratioUp.GetYaxis().SetRangeUser(-0.5,0.5)
        hist_ratioUp.GetYaxis().SetTitle("ratio")

        hist_ratio.SetMarkerStyle(20)
        hist_ratio.SetLineColor(kBlack)

        hist_ratioUp.SetFillColor(18)
        hist_ratioUp.SetLineColor(18)
        hist_ratioDn.SetFillColor(18)
        hist_ratioDn.SetLineColor(18)

        hist_ratioUp.Draw("HIST")
        hist_ratioDn.Draw("HIST SAME")
        hist_ratio.Draw("pex0 SAME")

        line.SetLineColor(kBlack)
        line.Draw("SAME")

        hist.SetTitle("")
        #hist.GetXaxis().SetTitle("Mass reco [GeV]")
        #hist.GetXaxis().SetTitleSize(20)
        #hist.GetXaxis().SetLabelSize(0.07)
        hist.GetYaxis().SetTitle("Frequency")
        hist.GetYaxis().SetTitleSize(20)
        hist.GetYaxis().SetTitleFont(43)

        #hist_ratioUp.GetYaxis().SetLabelSize(0.07)
        hist_ratioUp.GetYaxis().SetTitleSize(20)
        hist_ratioUp.GetYaxis().SetTitleFont(43)
        hist_ratioUp.GetYaxis().SetLabelFont(43)
        hist_ratioUp.GetYaxis().SetLabelSize(15)
        hist_ratioUp.GetXaxis().SetTitle("Mass reco [GeV]")
        hist_ratioUp.GetXaxis().SetTitleSize(20)
        hist_ratioUp.GetXaxis().SetTitleOffset(4)
        #hist_ratioUp.GetXaxis().SetLabelSize(0.1)
        hist_ratioUp.GetXaxis().SetLabelFont(43)
        hist_ratioUp.GetXaxis().SetLabelSize(15)

        c.Update()
        c.SaveAs(plotname)
        print(f'Saved plot to {plotname}.')
     
    return fit

def fitHist(case, doPlot):
    print(f'Fitting hists_ABCDnn_{case}_{binlo}to{binhi}_{bins}_pNet.root...')

    plotDir = f'fit_plots/{case}_{binlo}to{binhi}_{bins}'
    if not os.path.exists(plotDir):
        os.makedirs(plotDir)

    histFile = TFile.Open(f'hists_ABCDnn_{case}_{binlo}to{binhi}_{bins}_pNet.root', "READ")
    #histFile_lastbin = TFile.Open(f'hists_ABCDnn_{case}_{binlo}to{binhi}_420.root', "READ") # 42fit420bins

    for region in ["A", "B", "C", "D", "X", "Y", "V"]:
        params[case]["tgt"][region] = {}
        params[case]["pre"][region] = {}

    # fit
    # hist range: [400, 2500]
    if doPlot:
        regionList = ["D","V"]
    else:
        regionList = ["A", "B", "C", "D", "V"]
    for region in regionList:
        for htype in ["tgt", "pre"]:
            if htype == "tgt":
                hist = histFile.Get(f'Bprime_mass_dat_{region}') - histFile.Get(f'Bprime_mass_mnr_{region}')
                #hist_lastbin = histFile_lastbin.Get(f'Bprime_mass_dat_{region}') - histFile_lastbin.Get(f'Bprime_mass_mnr_{region}') # 42fit420bins
            else:
                hist = histFile.Get(f'Bprime_mass_pre_{region}')
                #hist_lastbin = histFile_lastbin.Get(f'Bprime_mass_pre_{region}') # 42fit420bins

            fit = fit_and_plot(hist, f'{plotDir}/fit_{htype}_{region}.png', case, doPlot) # normalizes hist

            # last bin is not fitted. get bin content. after scaling. capture shape only
            #hist_lastbin.Scale(1/hist_lastbin.Integral()) # 42fit420bins
            #params[case][htype][region]["lastbin"] = hist_lastbin.GetBinContent(420+1) # 42fit420bins
            params[case][htype][region]["lastbin"] = hist.GetBinContent(bins+1)
            for i in range(nparams):
                params[case][htype][region][f'param{str(i)}'] = [fit.GetParameter(i), fit.GetParError(i)]

    histFile.Close()
    #histFile_lastbin.Close() #TEMP # 42fit420bins
    
    #if doPlot:
    #    return 0


    #### FIT UNCERT ONLY ####
    # not needed for training+fit uncert method
    for i in range(nparams):
        pred_uncert[case][f'param{i}'] = params[case]["pre"]["D"][f'param{i}'][1]
    
    #### TRAINING UNCERTAINTY ON PARAMETERS ####
    # for i in range(nparams):
    #     train_uncert = 0
    #     nRegions = 3
    #     ## Method 1 ##
    #     # for region in ["A", "B", "C"]:
    #     #     uncert = abs((params[case]["pre"][region][f'param{i}'][0]-params[case]["tgt"][region][f'param{i}'][0])/params[case]["tgt"][region][f'param{i}'][0])
    #     #     if uncert<0.5:
    #     #         train_uncert += uncert
    #     #     else:
    #     #         nRegions = nRegions-1
    #     # if nRegions==0:
    #     #     train_uncert = 0
    #     # else:
    #     #     train_uncert = (train_uncert/nRegions)*params[case]["pre"]["D"][f'param{i}'][0] #absolute shift for D
    #     # fit_uncer = params[case]["pre"]["D"][f'param{i}'][1]
    #     # #fit_uncer = 0 # TEMP. debug only
    #     # pred_uncert[case][f'param{i}'] = abs(np.sqrt(train_uncert**2+fit_uncer**2)/params[case]["pre"]["D"][f'param{i}'][0])

    #     ## Method 2 ##
    #     if i<4:
    #         uncert = 0
    #         for region in ["A", "B", "C"]:
    #             uncert += abs((params[case]["pre"][region][f'param{i}'][0]-params[case]["tgt"][region][f'param{i}'][0])/params[case]["tgt"][region][f'param{i}'][0]) # params[case]["pre"][region][i] is a list of [param, err]
    #         train_uncert = (uncert/3)*params[case]["pre"]["D"][f'param{i}'][0] # absolute shift for D
    #         fit_uncer = params[case]["pre"]["D"][f'param{i}'][1]
    #         #fit_uncer = 0 # TEMP. for train_uncert fit_uncert comparison
    #         pred_uncert[case][f'param{i}'] = abs(np.sqrt(train_uncert**2+fit_uncer**2)/params[case]["pre"]["D"][f'param{i}'][0])
    #     else:
    #         pred_uncert[case][f'param{i}'] = abs(params[case]["pre"]["D"][f'param{i}'][1]/params[case]["pre"]["D"][f'param{i}'][0])
    #     print(f'Uncertainty in param{i}: ',pred_uncert[case][f'param{i}'])
            
    # lastbin_uncert=0
    # for region in ["A", "B", "C"]:
    #     #if params[case]["tgt"][region]["lastbin"]!=0: # TEMP. simplified for test only.
    #     #lastbin_uncert += abs(params[case]["pre"][region]["lastbin"]-params[case]["tgt"][region]["lastbin"])/params[case]["tgt"][region]["lastbin"]
    #     #    print("LASTBIN NOT 0")
    #     #elif params[case]["pre"][region]["lastbin"]!=0:
    #     #    print("LASTBIN NOT 0")
    #     lastbin_uncert += abs(params[case]["pre"][region]["lastbin"]-params[case]["tgt"][region]["lastbin"])/params[case]["pre"][region]["lastbin"]
    # pred_uncert[case]["lastbin"] = lastbin_uncert/3

def modifyOverflow(hist, bins):
    hist.SetBinContent(bins, hist.GetBinContent(bins)+hist.GetBinContent(bins+1))
    hist.SetBinContent(bins+1, 0)
    
#### TRAINING UNCERTAINTY BY BINS ####
def getPredUncert(case):
    # if case=="case1" or case=="case4": # TMEP: test other case14 models
    #    histFile = TFile.Open(f'logBpMlogST_mmd1_case14_random113/hists_ABCDnn_{case}_{binlo}to{binhi}_{bins}.root', "READ")
    # else:
    #    histFile = TFile.Open(f'hists_ABCDnn_{case}_{binlo}to{binhi}_{bins}.root', "READ")
    histFile = TFile.Open(f'hists_ABCDnn_{case}_{binlo}to{binhi}_{bins}_pNet.root', "READ")
    
    hist_dev = TH1D(f'train_uncert_{case}', f'Training Uncertainty ({case})', bins, binlo, binhi)
    for region in ["A", "B", "C"]:
        hist_pre = histFile.Get(f'Bprime_mass_pre_{region}')
        hist_tgt = histFile.Get(f'Bprime_mass_dat_{region}') - histFile.Get(f'Bprime_mass_mnr_{region}')
        hist_pre.Scale(1/hist_pre.Integral())
        hist_tgt.Scale(1/hist_tgt.Integral())
        modifyOverflow(hist_pre,bins)
        modifyOverflow(hist_tgt,bins)
        hist_dev += (hist_tgt - hist_pre)/hist_pre
    hist_dev.Scale(1/3) # average over 3 regions
    
    uncertFile.cd()
    hist_dev.Write()
    histFile.Close()

#fitHist("case14")
#fitHist("case23")
if withFit:
    fitHist("case1", doPlot=False) #plotting uncertainty band in ratio panel takes very long. skip plotting unless we need it
    fitHist("case2", doPlot=False)
    fitHist("case3", doPlot=False)
    fitHist("case4", doPlot=False)

uncertFile = TFile.Open(f'hists_trainUncert_{binlo}to{binhi}_{bins}_pNet.root', "RECREATE")
#getPredUncert("case14") #TEMP
#getPredUncert("case23") #TEMP 
getPredUncert("case1")
getPredUncert("case2")
getPredUncert("case3")
getPredUncert("case4")
uncertFile.Close()

# save parameters and last bin to a json file
json_obj = json.dumps(params, indent=4)
with open("fit_parameters.json", "w") as outjsonfile:
    outjsonfile.write(json_obj)
print("Saved parameter and last bin info to fit_parameters.json")

json_obj = json.dumps(pred_uncert, indent=4)
with open("pred_uncert.json", "w") as outjsonfile:
    outjsonfile.write(json_obj)
print("Saved parameter and last bin info to pred_uncert.json")

#exit() # TEMP

#############################
# create histogram from fit #
#############################
bins = 2100

alphaFactors = {}
with open("alphaRatio_factors_AN3.json","r") as alphaFile: #TEMP
    alphaFactors = json.load(alphaFile)
counts = {}
with open("counts_AN3.json","r") as countsFile: #TEMP
    counts = json.load(countsFile)
    
if not os.path.exists('application_plots'):
    os.makedirs('application_plots')

def shapePlot(region, case, hist_gen, hist_ABCDnn, step):
    c1 = TCanvas("c1", "c1")
    legend = TLegend(0.5,0.2,0.9,0.3)
    
    hist_gen.SetTitle(f'Shape verification {case}')
    hist_gen.SetLineColor(kBlue)
    #hist_gen.Scale(1/hist_gen.Integral())
    hist_gen.Draw("HIST")

    hist_ABCDnn.SetLineColor(kBlack)
    hist_ABCDnn.Scale(1/hist_ABCDnn.Integral())
    hist_ABCDnn.Draw("SAME")
    
    legend.AddEntry(hist_gen, "Histogram from the fit", "l")
    legend.AddEntry(hist_ABCDnn, "Histogram directly from ABCDnn", "l")
    #legend.AddEntry(fit, "Fit function from ABCDnn", "l")
    legend.Draw()

    #hist_ratio = hist_ABCDnn/hist_gen
    #hist_ratio.Print("all")

    c1.SaveAs(f'application_plots/GeneratedHist_with_fit_{case}_{region}_{step}.png')
    c1.Close()
    
def targetAgreementPlot(region, case, fit, hist_gen, hist_target, hist_abcdnn):
    c2 = TCanvas("c2", "c2", 800, 800)
    pad1 = TPad("hist_plot", "hist_plot", 0.05, 0.3, 1, 1)
    pad1.SetBottomMargin(0.01) #join upper and lower plot
    pad1.SetLeftMargin(0.1)
    pad1.Draw()
    pad1.cd()
    
    legend = TLegend(0.6,0.6,0.9,0.9)

    hist_gen.SetLineColor(kRed)
    hist_gen.Draw("HIST")
    hist_target.SetLineColor(kBlack)
    hist_target.Draw("SAME")
    hist_abcdnn.SetLineColor(kBlue)
    hist_abcdnn.Draw("HIST SAME")
    
    legend.AddEntry(hist_gen, "Histogram from fit", "l")
    legend.AddEntry(hist_abcdnn, "Directly from ABCDnn", "l")
    legend.AddEntry(hist_target, "Data-minor", "l")
    legend.Draw()    
    
    c2.cd()
    pad2 = TPad("ratio_plot", "ratio_plot", 0.05, 0.05, 1, 0.3)
    pad2.SetTopMargin(0)
    pad2.SetBottomMargin(0.2)
    pad2.SetLeftMargin(0.1)
    pad2.SetGrid()
    pad2.Draw()
    pad2.cd()

    line = TF1("line", "1", binlo, binhi, 0)
    
    hist_ratio = hist_target / hist_gen
    hist_ratio.SetTitle("")
    #hist_ratio.GetYaxis().SetRangeUser(0.5,1.5) #TEMP
    hist_ratio.GetYaxis().SetRangeUser(0,2)
    hist_ratio.GetYaxis().SetTitle("fit/target")

    hist_ratio.SetMarkerStyle(20)
    hist_ratio.SetLineColor(kBlack)
    hist_ratio.Draw("pex0")
    line.SetLineColor(kBlack)
    line.Draw("SAME")

    hist_gen.SetTitle(f'Major background {case} in region {region}')
    hist_gen.GetYaxis().SetTitle("Events/50GeV")
    hist_gen.GetYaxis().SetTitleSize(20)
    hist_gen.GetYaxis().SetTitleFont(43)

    hist_ratio.GetYaxis().SetTitleSize(20)
    hist_ratio.GetYaxis().SetTitleFont(43)

    #text = TText(0.4, 0.4,"Statistical uncertainty only")
    #text.Draw()

    c2.SaveAs(f'application_plots/GeneratedHist_with_target_{case}_{region}.png')
    c2.Close()

def fillHistogram(hist):
    # TEMP: turned of negative bin adjustment
    #for i in range(bins+1): # deals with negative bins
    #    if hist.GetBinContent(i)<0:
    #        hist.SetBinContent(i,0)
    return hist
    
def createHist(case, region):
    histFile = TFile.Open(f'hists_ABCDnn_{case}_{binlo}to{binhi}_{bins}_pNet.root', "READ")
    # if case=="case1" or case=="case4":
    #     histFile = TFile.Open(f'logBpMlogST_mmd1_case14_random113/hists_ABCDnn_{case}_{binlo}to{binhi}_{bins}.root', "READ") #TEMP: test other case14 models
    # else:
    #     histFile = TFile.Open(f'hists_ABCDnn_{case}_{binlo}to{binhi}_{bins}.root', "READ")
    
    if withFit:
        fit = TF1(f'fitFunc', fitFunc,binlo,binhi, nparams)
        for i in range(nparams):
            fit.SetParameter(i, params[case]["pre"][region][f'param{i}'][0])
        
        #fit.SetRange(0,2500)
        #fit.SetNpx(50)
        fit.SetNpx(bins)
        hist_gen = fit.CreateHistogram()
        #for j in range(10):
        #    if hist_gen.GetXaxis().GetBinCenter(j) < 400:
        #        hist_gen.SetBinContent(j, 0)
    
        hist_target = histFile.Get(f'Bprime_mass_dat_{region}').Clone(f'Bprime_mass_tgt_{region}') - histFile.Get(f'Bprime_mass_mnr_{region}').Clone()
        hist_abcdnn = histFile.Get(f'Bprime_mass_pre_{region}').Clone()
        hist_abcdnn_shape = hist_abcdnn.Clone()
    
        shapePlot(region, case, hist_gen, hist_abcdnn_shape, "step1")

        hist_gen.SetBinContent(bins, hist_gen.GetBinContent(bins)+params[case]["pre"][region]["lastbin"])
        hist_gen.SetBinContent(bins+1, 0) # overflow added to the previous bin
        hist_abcdnn_shape.SetBinContent(bins, hist_abcdnn_shape.GetBinContent(bins)+hist_abcdnn_shape.GetBinContent(bins+1))
        hist_abcdnn_shape.SetBinContent(bins+1, 0)
        hist_abcdnn.SetBinContent(bins, hist_abcdnn.GetBinContent(bins)+hist_abcdnn.GetBinContent(bins+1))
        hist_abcdnn.SetBinContent(bins+1, 0)
    
        shapePlot(region, case, hist_gen, hist_abcdnn_shape, "step2")

        normalization[case][region] = alphaFactors[case][region]["prediction"]/(hist_gen.Integral()) 
        hist_gen.Scale(normalization[case][region])
        hist_abcdnn_shape.Scale(normalization[case][region]) 

        # scale and plot with target and ABCDnn histograms
        shapePlot(region, case, hist_gen, hist_abcdnn_shape, "step3")
        
        hist_target.SetBinContent(bins, hist_target.GetBinContent(bins)+hist_target.GetBinContent(bins+1))
        hist_target.SetBinContent(bins+1, 0)
        hist_target.Scale(counts[case][region]["data"]/hist_target.Integral())

        if (case!="case3") and (case!="case4") and (region == "D"):
            for i in range(bins+1):
                if hist_target.GetBinCenter(i) > 1000:
                    hist_target.SetBinContent(i,0)
                
        targetAgreementPlot(region, case, fit, hist_gen, hist_target, hist_abcdnn_shape)

        hist_out = fillHistogram(hist_gen)
    else:
        hist_original = histFile.Get(f'Bprime_mass_pre_{region}').Clone() #with pNetSF

        normalization[case][region] = alphaFactors[case][region]["prediction"]/hist_original.Integral()
        hist_original.Scale(normalization[case][region])
        modifyOverflow(hist_original,bins)
        
        hist_out = fillHistogram(hist_original)

        if case=="case1" or case=="case2":
            hist_pNetUp = histFile.Get(f'Bprime_mass_pre_{region}_pNetUp').Clone()
            hist_pNetDn = histFile.Get(f'Bprime_mass_pre_{region}_pNetDn').Clone()
            hist_pNetUp.Scale(normalization[case][region])
            hist_pNetDn.Scale(normalization[case][region])
            modifyOverflow(hist_pNetUp,bins)
            modifyOverflow(hist_pNetDn,bins)
            hist_outUp = fillHistogram(hist_pNetUp)
            hist_outDn = fillHistogram(hist_pNetDn)
    
    if binlo==400:
        if doV2:
            if (region=="V" and (case=="case1" or case=="case2")) or (region=="D" and (case=="case3" or case=="case4")):
                outFile = TFile.Open(f'{outDir}/templatesV2_Jan2025_{bins}bins/templates_BpMass_ABCDnn_138fbfb_noNegBinAdjust.root', "UPDATE")
                hist_out.Write(f'BpMass_ABCDnn_138fbfb_isL_{tag[case]}_V2__major')
                if case=="case1":
                    hist_outUp.Write(f'BpMass_ABCDnn_138fbfb_isL_{tag[case]}_V2__major__pNetTtagUp')
                    hist_outDn.Write(f'BpMass_ABCDnn_138fbfb_isL_{tag[case]}_V2__major__pNetTtagDown')
                elif case=="case2":
                    hist_outUp.Write(f'BpMass_ABCDnn_138fbfb_isL_{tag[case]}_V2__major__pNetWtagUp')
                    hist_outDn.Write(f'BpMass_ABCDnn_138fbfb_isL_{tag[case]}_V2__major__pNetWtagDown')
                outFile.Close()
        else:
            outFile = TFile.Open(f'{outDir}/templates{region}_Jan2025_{bins}bins/templates_BpMass_ABCDnn_138fbfb_noNegBinAdjust.root', "UPDATE")
            hist_out.Write(f'BpMass_ABCDnn_138fbfb_isL_{tag[case]}_{region}__major')
            if case=="case1":
                hist_outUp.Write(f'BpMass_ABCDnn_138fbfb_isL_{tag[case]}_{region}__major__pNetTtagUp')
                hist_outDn.Write(f'BpMass_ABCDnn_138fbfb_isL_{tag[case]}_{region}__major__pNetTtagDn')
            elif case=="case2":
                hist_outUp.Write(f'BpMass_ABCDnn_138fbfb_isL_{tag[case]}_{region}__major__pNetWtagUp')
                hist_outDn.Write(f'BpMass_ABCDnn_138fbfb_isL_{tag[case]}_{region}__major__pNetWtagDn')
            outFile.Close()
    elif binlo==0:
        outFile = TFile.Open(f'{outDir}/templates{region}_Jan2025_Julie/templates_BpMass_138fbfb_noNegBinAdjust.root', "UPDATE")
        hist_out.Write(f'BpMass_138fbfb_isL_{tag[case]}_{region}__major')
        outFile.Close()
    else:
        exit()
        print("New binning encounterd. Make up a new name and folder.")
    
for case in ["case1", "case2", "case3", "case4"]:
    createHist(case, "D") #TEMP
    #createHist(case, "D2") #TEMP
    createHist(case, "V") #TEMP
    # if doV2:
    #     createHist(case, "D")
    #     createHist(case, "V")
    # else:
    #     createHist(case, "D")

# shift
def shiftTrainingUncert(case, region, shift):
    if withFit:
        fit = TF1(f'fitFunc', fitFunc, binlo,binhi, nparams)
        for i in range(nparams):
            fit.SetParameter(i, params[case]["pre"][region][f'param{i}'][0])
        fit.SetNpx(bins)
        hist_bin = fit.CreateHistogram() # fit method
        hist_bin.SetBinContent(bins+1, params[case]["pre"][region]["lastbin"]) #fit method
    else:
        histFile = TFile.Open(f'hists_ABCDnn_{case}_{binlo}to{binhi}_{bins}_pNet.root', "READ")
        # if case=='case1' or case=="case4":
        #     histFile = TFile.Open(f'logBpMlogST_mmd1_case14_random113/hists_ABCDnn_{case}_{binlo}to{binhi}_{bins}.root', "READ") #TEMP: test other models
        # else:
        #     histFile = TFile.Open(f'hists_ABCDnn_{case}_{binlo}to{binhi}_{bins}.root', "READ")
        
        hist_bin = histFile.Get(f'Bprime_mass_pre_{region}').Clone()
    
    modifyOverflow(hist_bin,bins) # take overflow bin and add to last bin. set over flow bin to 0

    uncertFile = TFile.Open(f'hists_trainUncert_{binlo}to{binhi}_{bins}_pNet.root', 'READ')
    if separateUncertCases:
        uncertHist = uncertFile.Get(f'train_uncert_{case}') # apply separete uncerts
    else:
        if case=="case1" or case=="case4": # apply combined uncerts
            uncertHist = uncertFile.Get("train_uncert_case14")
        else: # assumes not applying to combined cases (case14 and 23)
            uncertHist = uncertFile.Get("train_uncert_case23")

    if shift=="Up":
        for i in range(bins+1):
            hist_bin.SetBinContent(i, hist_bin.GetBinContent(i) * (1 + uncertHist.GetBinContent(i)))
    else:
        for i in range(bins+1):
    	    hist_bin.SetBinContent(i, hist_bin.GetBinContent(i) * (1 - uncertHist.GetBinContent(i)))

    hist_bin.Scale(normalization[case][region])
    
    hist_out = fillHistogram(hist_bin)

    if binlo==400:
        if doV2:
            if (region=="V" and (case=="case1" or case=="case2")) or (region=="D" and (case=="case3" or case=="case4")):
                outFile = TFile.Open(f'{outDir}/templatesV2_Jan2025_{bins}bins/templates_BpMass_ABCDnn_138fbfb_noNegBinAdjust.root', "UPDATE")
                hist_out.Write(f'BpMass_ABCDnn_138fbfb_isL_{tag[case]}_V2__major__train{shift}')
                outFile.Close()
        else:
            outFile = TFile.Open(f'{outDir}/templates{region}_Jan2025_{bins}bins/templates_BpMass_ABCDnn_138fbfb_noNegBinAdjust.root', "UPDATE")
            hist_out.Write(f'BpMass_ABCDnn_138fbfb_isL_{tag[case]}_{region}__major__train{shift}')
            outFile.Close()
    elif binlo==0:
        outFile = TFile.Open(f'{outDir}/templates{region}_Jan2025_Julie/templates_BpMass_138fbfb_noNegBinAdjust.root', "UPDATE")
        hist_out.Write(f'BpMass_138fbfb_isL_{tag[case]}_{region}__major__train{shift}')
        outFile.Close()
    else:
        exit()
        print("New binning encounterd. Make up a new name and folder.")
    
def shiftLastBin(case, region, shift):
    fit = TF1(f'fitFunc', fitFunc, binlo,binhi, nparams)
    for i in range(nparams):
        fit.SetParameter(i, params[case]["pre"][region][f'param{i}'][0])

    #fit.SetRange(0,2500)
    #fit.SetNpx(50)
    fit.SetNpx(bins)

    hist_bin = fit.CreateHistogram()
    #for j in range(10):
    #    if hist_bin.GetXaxis().GetBinCenter(j) < 400:
    #        hist_bin.SetBinContent(j, 0)
            
    if shift=="Up":
        hist_bin.SetBinContent(bins+1, params[case]["pre"][region]["lastbin"]*(1+pred_uncert[case]["lastbin"]))
    else:
        hist_bin.SetBinContent(bins+1, params[case]["pre"][region]["lastbin"]*(1-pred_uncert[case]["lastbin"]))

    hist_bin.SetBinContent(bins, hist_bin.GetBinContent(bins)+hist_bin.GetBinContent(bins+1))
    hist_bin.SetBinContent(bins+1, 0)
    #hist_bin.Scale(1/hist_bin.Integral())

    hist_bin.Scale(normalization[case][region])

    hist_out = fillHistogram(hist_bin)
    
    if binlo==400:
        if doV2:
            if (region=="V" and (case=="case1" or case=="case2")) or (region=="D" and (case=="case3" or case=="case4")):
                outFile = TFile.Open(f'{outDir}/templatesV2_Jan2025_{bins}bins/templates_BpMass_ABCDnn_138fbfb_noNegBinAdjust.root', "UPDATE")
                hist_out.Write(f'BpMass_ABCDnn_138fbfb_isL_{tag[case]}_V2__major__lastbin{shift}')
                outFile.Close()
        else:
            outFile = TFile.Open(f'{outDir}/templates{region}_Jan2025_{bins}bins/templates_BpMass_ABCDnn_138fbfb_noNegBinAdjust.root', "UPDATE")
            hist_out.Write(f'BpMass_ABCDnn_138fbfb_isL_{tag[case]}_{region}__major__lastbin{shift}')
            outFile.Close()
    elif binlo==0:
        outFile = TFile.Open(f'{outDir}/templates{region}_Jan2025_Julie/templates_BpMass_138fbfb_noNegBinAdjust.root', "UPDATE")
        hist_out.Write(f'BpMass_138fbfb_isL_{tag[case]}_{region}__major__lastbin{shift}')
        outFile.Close()
    else:
        exit()
        print("New binning encounterd. Make up a new name and folder.")
        
def shiftParam(case, region, i, shift):
    fit = TF1(f'fitFunc', fitFunc,binlo,binhi, nparams)
    for j in range(nparams):
        fit.SetParameter(j, params[case]["pre"][region][f'param{j}'][0])

    #fit.SetRange(0,2500)
    #fit.SetNpx(50)
    fit.SetNpx(bins)

    fit_original = fit.Clone()
    
    if shift=="Up":
        #fit.SetParameter(i, params[case]["pre"][region][f'param{i}'][0]*(1+pred_uncert[case][f'param{i}'])) # param uncert method
        fit.SetParameter(i, params[case]["pre"][region][f'param{i}'][0]+params[case]["pre"][region][f'param{i}'][1]) #bin uncert method
    else:
        #fit.SetParameter(i, params[case]["pre"][region][f'param{i}'][0]*(1-pred_uncert[case][f'param{i}'])) # param uncert method
        fit.SetParameter(i, params[case]["pre"][region][f'param{i}'][0]-params[case]["pre"][region][f'param{i}'][1]) #bin uncert method 
        
    hist_param = fit.CreateHistogram()

    hist_param.SetBinContent(bins+1, params[case]["pre"][region]["lastbin"])
    hist_param.SetBinContent(bins, hist_param.GetBinContent(bins)+hist_param.GetBinContent(bins+1))
    hist_param.SetBinContent(bins+1, 0)
    #hist_param.Scale(1/hist_param.Integral())

    hist_param.Scale(normalization[case][region])

    hist_out = fillHistogram(hist_param)
    if binlo==400:
        if doV2:
            if (region=="V" and (case=="case1" or case=="case2")) or (region=="D" and (case=="case3" or case=="case4")):
                outFile = TFile.Open(f'{outDir}/templatesV2_Jan2025_{bins}bins/templates_BpMass_ABCDnn_138fbfb_noNegBinAdjust.root', "UPDATE")
                hist_out.Write(f'BpMass_ABCDnn_138fbfb_isL_{tag[case]}_V2__major__param{i}{shift}')
                outFile.Close()
        else:
            outFile = TFile.Open(f'{outDir}/templates{region}_Jan2025_{bins}bins/templates_BpMass_ABCDnn_138fbfb_noNegBinAdjust.root', "UPDATE")
            hist_out.Write(f'BpMass_ABCDnn_138fbfb_isL_{tag[case]}_{region}__major__param{i}{shift}')
            outFile.Close()
    elif binlo==0:
        outFile = TFile.Open(f'{outDir}/templates{region}_Jan2025_Julie/templates_BpMass_138fbfb_noNegBinAdjust.root', "UPDATE")
        hist_out.Write(f'BpMass_138fbfb_isL_{tag[case]}_{region}__major__param{i}{shift}')
        outFile.Close()
    else:
        exit()
        print("New binning encounterd. Make up a new name and folder.")

# Factor uncertainty
def shiftFactor(case, region, shift):
    fit = TF1(f'fitFunc', fitFunc, binlo, binhi, nparams)
    
    for i in range(nparams):
        fit.SetParameter(i, params[case]["pre"][region][f'param{i}'][0])

    fit.SetNpx(bins)
    hist = fit.CreateHistogram()

    if shift=="Up":
        hist.Scale(normalization[case][region]*(1+alphaFactors[case][region]["uncertainty"]))
    else:
        hist.Scale(normalization[case][region]*(1-alphaFactors[case][region]["uncertainty"]))

    hist.Scale(normalization[case][region])

    hist_out = fillHistogram(hist)
    #hist_out.Rebin(10)
    
    if binlo==400:
        if doV2:
            if (region=="V" and (case=="case1" or case=="case2")) or (region=="D" and (case=="case3" or case=="case4")):
                outFile = TFile.Open(f'{outDir}/templatesV2_Jan2025_{bins}bins/templates_BpMass_ABCDnn_138fbfb_noNegBinAdjust.root', "UPDATE")
                hist_out.Write(f'BpMass_ABCDnn_138fbfb_isL_{tag[case]}_V2__major__factor{shift}')
                outFile.Close()
        else:
            outFile = TFile.Open(f'{outDir}/templates{region}_Jan2025_{bins}bins/templates_BpMass_ABCDnn_138fbfb_noNegBinAdjust.root', "UPDATE")
            hist_out.Write(f'BpMass_ABCDnn_138fbfb_isL_{tag[case]}_{region}__major__factor{shift}')
            outFile.Close()
    elif binlo==0:
        outFile = TFile.Open(f'{outDir}/templates{region}_Jan2025_Julie/templates_BpMass_138fbfb_noNegBinAdjust.root', "UPDATE")
        hist_out.Write(f'BpMass_138fbfb_isL_{tag[case]}_{region}__major__factor{shift}')
        outFile.Close()
    else:
        exit()
        print("New binning encounterd. Make up a new name and folder.")
    
    
for case in ["case1", "case2", "case3", "case4"]:
    for shift in ["Up", "Down"]:
        shiftTrainingUncert(case, "D", shift) #TEMP
        #shiftTrainingUncert(case, "D2", shift) #TEMP
        ##shiftFactor(case, "D", shift)
        ##shiftLastBin(case, "D", shift) # not needed for bin train uncert method
        
        shiftTrainingUncert(case, "V", shift) #TEMP
        ##shiftFactor(case, "V", shift)
        ##shiftLastBin(case, "V", shift)
        # if doV2:
        #     shiftLastBin(case, "V", shift)
        #     shiftLastBin(case, "D", shift)
        #     shiftFactor(case, "V", shift)
        #     shiftFactor(case, "D", shift)
        # else:
        #     shiftLastBin(case, "D", shift)
        #     shiftFactor(case, "D", shift)

        if withFit:
            for i in range(nparams):
                shiftParam(case, "V", i, shift)
                shiftParam(case, "D", i, shift)
            # if doV2:
            #     shiftParam(case, "V", i, shift)
            #     shiftParam(case, "D", i, shift)
            # else:
            #     shiftParam(case, "D", i, shift)

#print(pred_uncert)
