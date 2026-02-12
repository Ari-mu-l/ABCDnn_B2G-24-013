# python3 optimizeFit_abcdnn.py -f skewNorm_cubic -c case14
import os
import numpy as np
from ROOT import *
from argparse import ArgumentParser

gStyle.SetOptFit(1)
gROOT.SetBatch(True) # suppress histogram display
TH1.SetDefaultSumw2(True)

# define the fit function
# f(x) = 2 pdf(x) * cdf(a x)
# f(x) = 2/w pdf((x-xi)/2) * cdf(a*(x-xi)/2)
# f(x) = f(x) + ax^2 + bx + c

parser = ArgumentParser()
parser.add_argument("-f", "--fitType" , default="skewNorm_cubic")
parser.add_argument("-c", "--case"    , default="case14")

fitType = parser.parse_args().fitType
case = parser.parse_args().case

binlo = 400 #400
binhi = 2500
bins = 420 #42

log = False # set True if want to use log(BpM)

if log:
    histFileName = f'hists_ABCDnn_{case}_{binlo}to{binhi}_{bins}_log.root'
    plotDir = f'fit_plots_{case}/{fitType}_{binlo}to{binhi}_{bins}_log'
else:
    histFileName = f'hists_ABCDnn_{case}_{binlo}to{binhi}_{bins}.root'
    plotDir = f'fit_plots_{case}/{fitType}_{binlo}to{binhi}_{bins}'
    
if not os.path.exists(plotDir):
    os.makedirs(plotDir)

if fitType=="skewNorm":
    fitFunc = "[3] * (2/[1]) * ( TMath::Exp(-((x-[2])/[1])*((x-[2])/[1])/2) / TMath::Sqrt(2*TMath::Pi()) ) * ( (1 + TMath::Erf([0]*((x-[2])/[1])/TMath::Sqrt(2)) ) / 2 )"
    nparams = 4
    fit     = TF1("fitFunc", fitFunc, binlo, binhi, nparams)
    fit.SetParameters(5, 400, 500, 50)
elif fitType=="skewNorm_linear":
    fitFunc = "[3] * (2/[1]) * ( TMath::Exp(-((x-[2])/[1])*((x-[2])/[1])/2) / TMath::Sqrt(2*TMath::Pi()) ) * ( (1 + TMath::Erf([0]*((x-[2])/[1])/TMath::Sqrt(2)) ) / 2 ) + [4] + [5] * x"
    nparams = 6
    fit     = TF1("fitFunc", fitFunc, 400, 2500, nparams)
    #fit.SetParameters(5, 400, 500, 50, 0.00001, 0.00001) # good for case1
    #fit.SetParameters(5, 400, 500, 50, 0.001, 0.0001)
    fit.SetParameters(5, 400, 500, 50)
elif fitType=="skewNorm_quadratic":
    fitFunc = "[3] * (2/[1]) * ( TMath::Exp(-((x-[2])/[1])*((x-[2])/[1])/2) / TMath::Sqrt(2*TMath::Pi()) ) * ( (1 + TMath::Erf([0]*((x-[2])/[1])/TMath::Sqrt(2)) ) / 2 ) + [4] + [5] * x + [6] * x * x"
    nparams = 7
    fit     = TF1("fitFunc", fitFunc, 400, 2500, nparams)
    #fit.SetParameters(5, 400, 500, 50, 0.00001, 0.0000001, 0.00000001)
    #fit.SetParameters(5, 400, 500, 50)
elif fitType=="skewNorm_cubic":
    fitFunc = "[3] * (2/[1]) * ( TMath::Exp(-((x-[2])/[1])*((x-[2])/[1])/2) / TMath::Sqrt(2*TMath::Pi()) ) * ( (1 + TMath::Erf([0]*((x-[2])/[1])/TMath::Sqrt(2)) ) / 2 ) + [4] + [5] * x + [6] * x * x + [7] * x * x * x"
    nparams = 8
    fit = TF1("fitFunc", fitFunc, binlo, binhi, nparams)
    fit.SetParameters(5, 400, 500, 50, 0.00001, 0.000001, 0.00000001) #case2
    #fit.SetParLimits(1, 100, 400)
    #fit.SetParLimits(2, 200, 800)
    #fit.SetParameters(5, 400, 500, 50, 0.0000001, 0.0000000001)
elif fitType=="skewNorm_cubic_2":
    fitFunc = "[3] * (2/[1]) * ( TMath::Exp(-((x-[2])/[1])*((x-[2])/[1])/2) / TMath::Sqrt(2*TMath::Pi()) ) * ( (1 + TMath::Erf([0]*((x-[2])/[1])/TMath::Sqrt(2)) ) / 2 ) + [4] * (x/2500) + [5] * (x/2500) * (x/2500) + [6] * (x/2500) * (x/2500) * (x/2500)"
    nparams = 7
    fit = TF1("fitFunc", fitFunc, 400, 2500, nparams)
    fit.SetParameters(5, 400, 500, 50)
    #fit.SetParameters(5, 400, 500, 50, 0.00001, 0.000001, 0.00000001)
    #fit.SetParameters(5, 400, 500, 50, 0.0000001, 0.0000000001, 0.00000000001)
    #fit.SetParameters(5, 400, 500, 50, 0.0000001, 0.00000001, 0.000000000001)
    #fit.SetParLimits(1, 100, 400)
    #fit.SetParLimits(2, 200, 800)
elif fitType=="skewNorm_4":
    fitFunc = "[3] * (2/[1]) * ( TMath::Exp(-((x-[2])/[1])*((x-[2])/[1])/2) / TMath::Sqrt(2*TMath::Pi()) ) * ( (1 + TMath::Erf([0]*((x-[2])/[1])/TMath::Sqrt(2)) ) / 2 ) + [4] + [5] * x + [6] * x * x + [7] * x * x * x + [8] * x * x * x * x"
    nparams = 9
    fit = TF1("fitFunc", fitFunc, 400, 2500, nparams)
    #fit.SetParameters(5, 400, 500, 50, 0.0000001, 0.0000000001)
    fit.SetParameters(5, 400, 500, 50, 0.0000001, 0.000000001, 0.00000000001)
elif fitType=="skewNorm_4_2":
    fitFunc = "[3] * (2/[1]) * ( TMath::Exp(-((x-[2])/[1])*((x-[2])/[1])/2) / TMath::Sqrt(2*TMath::Pi()) ) * ( (1 + TMath::Erf([0]*((x-[2])/[1])/TMath::Sqrt(2)) ) / 2 ) + [4] * x + [5] * x * x + [6] * x * x * x + [7] * x * x * x * x"
    nparams = 8
    fit = TF1("fitFunc", fitFunc, 400, 2500, nparams)
    fit.SetParameters(5, 400, 500, 50, 0.0000001, 0.000000001, 0.00000000001) # works for case1 (not V)
elif fitType=="skewNorm_5":
    fitFunc = "[3] * (2/[1]) * ( TMath::Exp(-((x-[2])/[1])*((x-[2])/[1])/2) / TMath::Sqrt(2*TMath::Pi()) ) * ( (1 + TMath::Erf([0]*((x-[2])/[1])/TMath::Sqrt(2)) ) / 2 ) + [4] + [5] * x + [6] * x * x + [7] * x * x * x + [8] * x * x * x * x + [9] * x * x * x * x * x"
    nparams = 10
    fit = TF1("fitFunc", fitFunc, 400, 2500, nparams)
    fit.SetParameters(5, 400, 500, 50, 0.0000001, 0.000000001, 0.000000000001)
elif fitType=="skewNorm_5_2":
    fitFunc = "[3] * (2/[1]) * ( TMath::Exp(-((x-[2])/[1])*((x-[2])/[1])/2) / TMath::Sqrt(2*TMath::Pi()) ) * ( (1 + TMath::Erf([0]*((x-[2])/[1])/TMath::Sqrt(2)) ) / 2 ) + [4] * x + [5] * x * x + [6] * x * x * x + [7] * x * x * x * x + [8] * x * x * x * x * x"
    nparams = 9
    fit = TF1("fitFunc", fitFunc, 400, 2500, nparams) 
    fit.SetParameters(5, 400, 500, 50, 0.0000001, 0.000000001, 0.000000000001)
    #fit.SetParLimits(1, 0, 400)
    #fit.SetParLimits(2, 200, 1000)
elif fitType=="skewNorm_6":
    fitFunc = "[3] * (2/[1]) * ( TMath::Exp(-((x-[2])/[1])*((x-[2])/[1])/2) / TMath::Sqrt(2*TMath::Pi()) ) * ( (1 + TMath::Erf([0]*((x-[2])/[1])/TMath::Sqrt(2)) ) / 2 ) + [4] + [5] * x + [6] * x * x + [7] * x * x * x + [8] * x * x * x * x + [9] * x * x * x * x * x + [10] * x * x * x * x * x * x"
    nparams = 11
    fit = TF1("fitFunc", fitFunc, 400, 2500, nparams)
    fit.SetParameters(5, 400, 500, 50, 0.0000001, 0.000000001, 0.000000000001) # case 23
    #fit.SetParameters(5, 400, 500, 50, 0.0000001, 0.000000001, 0.0000000001)#case4
    #fit.SetParLimits(1, 0, 600) #case4
    #fit.SetParLimits(2, 0, 600) #case4
elif fitType=="landau":
    fitFunc = "TMath::Landau(x, [0], [1])"
    nparams = 2
    fit = TF1("fitFunc", fitFunc, 400, 2500, nparams)
    fit.SetParameters(100, 200)
elif fitType=="landau_linear":
    fitFunc = "TMath::Landau(x, [0], [1]) + [2] + [3] * x"
    nparams = 4
    fit = TF1("fitFunc", fitFunc, 400, 2500, nparams)
    fit.SetParameters(600, 500)
elif fitType=="landau_quadratic":
    fitFunc = "TMath::Landau(x, [0], [1]) + [2] + [3] * x + [4] * x * x"
    nparams = 5
    fit = TF1("fitFunc", fitFunc, 400, 2500, nparams)
    fit.SetParameters(600, 500)
elif fitType=="landau_cubic":
    fitFunc = "TMath::Landau(x, [0], [1]) + [2] + [3] * x + [4] * x * x + [5] * x * x * x"
    nparams = 6
    fit = TF1("fitFunc", fitFunc, 400, 2500, nparams)
    fit.SetParameters(600, 500)
elif fitType=="crystalball":
    nparams = 5
    fit = TF1("crystalball", "crystalball", 400, 2500, nparams)
    #fit.SetParameters(0.1, 500, 200, -2.5, 100000000) # case14, regionABCD
    #fit.SetParameters(0.1, 500, 200, -3, 100000000) # case14, regionV
    #fit.SetParameters(0.001, 500, 250, -2.5, 100000000) # case23, regionABCDV
    fit.SetParameters(0.01, 600, 150, -2, 1000)
elif fitType=="quadratic":
    fitFunc = "[0] + [1]*(x/2500) + [2]*(x/2500)*(x/2500)"
    nparams = 3
    fit = TF1("fitFunc", fitFunc, 400, 2500, nparams)
else:
    print("fitFunc not defined. Please specify.")
    exit()

#skewFit.SetParameters(5, 400, 500, 50, 0.00001, 0.000001, 0.00000001)
#skewFit.SetParameters(5, 400, 500, 50, 0.00001, 0.000001, 0.00000001, 0.0000000001)
#skewFit.SetParameters(5, 500, 500, 50, 0.00001, 0.000001, 0.000000001)

def fit_and_plot(hist, plotname):
    c = TCanvas("")
    latex = TLatex()
    latex.SetNDC()

    hist.SetBinContent(bins+1, 0)
    hist.Scale(1/hist.Integral())
    hist.Fit(fit, "E")
    hist.Draw()
        
    chi2ndof = fit.GetChisquare() / fit.GetNDF()
    latex.DrawText(0.6, 0.2, f'chi2/ndof = {round(chi2ndof,2)}')
    
    c.SaveAs(plotname)
    print(f'Saved plot to {plotname}.')


print(f'Fitting {histFileName}...')
histFile = TFile.Open(histFileName, "READ")

# store parameters in dictionaries
params = {"tgt":{},
          "pre":{}
          }

#for region in ["A", "B", "C", "D", "X", "Y", "V"]:
for region in ["A", "B", "C", "D", "V"]:
    params["tgt"][region] = {}
    params["pre"][region] = {}

# fit
# hist range: [400, 2500]
#for region in ["A", "B", "C", "D", "X", "Y", "V"]:
for region in ["A", "B", "C", "D", "V"]:
    for htype in ["tgt", "pre"]:
        if htype == "tgt":
            hist = histFile.Get(f'Bprime_mass_dat_{region}') - histFile.Get(f'Bprime_mass_mnr_{region}')
        elif htype == "pre":
            hist = histFile.Get(f'Bprime_mass_pre_{region}')
        else:
            print(f'Undefined histogram type {htype}. Check for typo.')

        fit_and_plot(hist, f'{plotDir}/fit_{htype}_{region}.png')

        for i in range(nparams):
            params[htype][region][str(i)] = fit.GetParameter(i)

histFile.Close()

# compare
params_uncert = {}
for i in range(nparams):
    uncert = 0
    for region in ["A", "B", "C"]: # excluded X, Y. Y p6 oddly different.
    #for region in ["B", "C"]:
        uncert += abs((params["pre"][region][str(i)]-params["tgt"][region][str(i)])/params["tgt"][region][str(i)])
        print(f'Region {region} deviation in param{i}:', abs((params["pre"][region][str(i)]-params["tgt"][region][str(i)])/params["tgt"][region][str(i)]))
    params_uncert[i] = uncert/3
    print(f'Avg deviation in param{i}: {100*uncert/3}%')
