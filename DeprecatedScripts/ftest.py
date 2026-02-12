# python3 ftest.py -f skewNorm -c case1
import os
import numpy as np
import scipy.stats
from ROOT import *
from argparse import ArgumentParser

gStyle.SetOptFit(1)
gROOT.SetBatch(True) # suppress histogram display

parser = ArgumentParser()
parser.add_argument("-f", "--fitType" , default="skewNorm")
parser.add_argument("-c", "--case"    , default="case14")

fitType = parser.parse_args().fitType
case = parser.parse_args().case

binlo = 400
binhi = 2500
bins = 420
npoly = 8

plotDir = f'ftest_{case}/{fitType}_{binlo}to{binhi}_{bins}'
if not os.path.exists(plotDir):
    os.makedirs(plotDir)
    
baseFunc = "[3] * (2/[1]) * ( TMath::Exp(-((x-[2])/[1])*((x-[2])/[1])/2) / TMath::Sqrt(2*TMath::Pi()) ) * ( (1 + TMath::Erf([0]*((x-[2])/[1])/TMath::Sqrt(2)) ) / 2 )"

def fit_and_create(hist, fitFunc, nparams, plotname):
    c = TCanvas("")
    latex = TLatex()
    latex.SetNDC()
    
    fit = TF1("fitFunc", fitFunc, binlo, binhi, nparams)
    fit.SetParameters(5, 400, 500, 50)

    hist.SetBinContent(bins+1, 0)
    hist.Scale(1/hist.Integral())
    hist.Fit(fit, "E")
    hist.Draw()

    fit.SetNpx(bins)
    hist_gen = fit.CreateHistogram()
    
    c.SaveAs(plotname)
    print(f'Saved plot to {plotname}.')

    return hist_gen

def get_RSS(hist_in, htype, region):
    for npower in range(npoly+1):
        nparams = npower+4
        polyfunc = '' # w/o constant term
        #polyfunc = '+ [4]' # with constant term
        for i in range(npower):
            polyfunc += f'+ [{i+4}]*(x/2500)' # w/o constant term
            #polyfunc += f'+ [{i+5}]*(x/2500)' # with constant term
            for power in range(i):
                polyfunc += '*(x/2500)'
        fitFunc = baseFunc + polyfunc

        hist_out = fit_and_create(hist_in, fitFunc, nparams, f'{plotDir}/fit_{htype}_{region}_{npower}.png')

        hist_diff = hist_out - hist_in
        diff = 0
        for i in range(bins):
            diff += (hist_diff.GetBinContent(i))**2
        RSS_dict[htype][region][f'poly{npower}'] = diff

def get_f(p1,p2,n):
    if separateRegions:
        RSS1 = RSS_dict[htype][region][f'poly{p1}']
        RSS2 = RSS_dict[htype][region][f'poly{p2}']
    else:
        RSS1 = RSS_all[f'poly{p1}']
        RSS2 = RSS_all[f'poly{p2}']
    return (RSS1-RSS2)*(n-p2-4)/(RSS2*(p2-p1))

histFile = TFile.Open(f'hists_ABCDnn_{case}_{binlo}to{binhi}_{bins}.root', "READ")
RSS_dict = {"tgt":{}, "pre":{}}
f_dict = {"tgt":{}, "pre":{}}
separateRegions = False

for region in ["A", "B", "C", "D", "V"]:
    for htype in ["tgt",  "pre"]:
        if htype=="tgt":
            hist_in = histFile.Get(f'Bprime_mass_dat_{region}') - histFile.Get(f'Bprime_mass_mnr_{region}')
        else:
            hist_in = histFile.Get(f'Bprime_mass_pre_{region}')

        RSS_dict[htype][region] = {}
        get_RSS(hist_in, htype, region)

        if separateRegions:
            f_dict[htype][region] = {}
            for i in range(npoly):
                f_dict[htype][region][f'f{i}{i+1}'] = get_f(i,i+1,bins)
                #print(scipy.stats.f.ppf(0.05, 1, bins-(i+1+4)))

if not separateRegions:
    #combinedRegions = ["A", "B", "C", "D", "V"]
    combinedRegions = ["D", "V"]
    n = len(combinedRegions)
    RSS_all={}
    f_all={}
    for i in range(npoly+1):
        RSS_all[f'poly{i}'] = 0
        for htype in ["tgt",  "pre"]:
            for region in combinedRegions:
                RSS_all[f'poly{i}'] += RSS_dict[htype][region][f'poly{i}']

    # CALCULATE f
    # fine-tune
    #p1=4
    #p2=6
    #f_all[f'f{p1}{p2}'] = get_f(p1,p2,bins*n)
    #print(f'sig:{0.05}, df1:{n*(p2-p1)}, df2:{(bins-(p2+4))*n}')

    # general search
    for i in range(npoly):
        f_all[f'f{i}{i+1}'] = get_f(i,i+1,bins*n)
        print(f'sig:{0.05}, df1:{n}, df2:{(bins-(i+1+4))*n}, F_c:{scipy.stats.f.ppf(q=0.05, dfn=1*n, dfd=(bins-(i+1+4))*n)}') # significance level 0.05. p2-p1=1. n-p2. p2 = (i+1)+4. 4 params from gaussian
        
print(f_all)
