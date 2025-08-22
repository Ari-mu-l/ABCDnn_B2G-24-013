# TODO: check validation prediction and uncertainty
# Did it quick for inaugural talk

import os, sys
import numpy as np
import ROOT
import json

if len(sys.argv)>1:
    getAlphaRatio = sys.argv[1]
else:
    getAlphaRatio = "True"

year = '' # '', '_2016'
iPlot = 'BpMass_pad_full' # BpMass, BpMass_pad, BpMass_padfull

if iPlot == 'BpMass_pad':
    fileNameTag = '_2Dpad'
elif iPlot == 'BpMass_padfull':
    fileNameTag = '_2Dpadfull'
elif iPlot == 'BpMass':
    fileNameTag = ''

###############################
# calculate correction factor #
###############################
if getAlphaRatio=="True":
    # get counts
    caseName = {"case1" : "tagTjet",
                "case2" : "tagWjet",
                "case3" : "untagTlep",
                "case4" : "untagWlep",
                }

    counts = {"case14":{},
              "case23":{},
              "case1":{},
              "case2":{},
              "case3":{},
              "case4":{},
              }

    for case in counts:
        #for region in ["B", "D", "V", "BV", "highST", "BhighST"]: #, "B2", "D2"]: # general
        for region in ["B", "D", "V", "BV"]: # for year-by-year gof
            counts[case][region] = {}

    def getCounts(case, region):
        print(f'Processing {case}')
        #tempFileName  = f'/uscms/home/jmanagan/nobackup/BtoTW/CMSSW_13_0_18/src/vlq-BtoTW-SLA/makeTemplates/templates{region}_Jan2025/templates_BpMass_138fbfb.root'
        tempFileName = f'/uscms/home/xshen/nobackup/alma9/CMSSW_13_3_3/src/vlq-BtoTW-SLA/makeTemplates/templates{region}_Jan2025/templates_{iPlot}_138fbfb{year}.root'
        tFile = ROOT.TFile.Open(tempFileName, 'READ')
        hist_data  = tFile.Get(f'{iPlot}_138fbfb_isL_{caseName[case]}_{region}__data_obs')
        hist_major = tFile.Get(f'{iPlot}_138fbfb_isL_{caseName[case]}_{region}__qcd') + tFile.Get(f'{iPlot}_138fbfb_isL_{caseName[case]}_{region}__wjets') + tFile.Get(f'{iPlot}_138fbfb_isL_{caseName[case]}_{region}__singletop') + tFile.Get(f'{iPlot}_138fbfb_isL_{caseName[case]}_{region}__ttbar')
        hist_minor = tFile.Get(f'{iPlot}_138fbfb_isL_{caseName[case]}_{region}__ttx') + tFile.Get(f'{iPlot}_138fbfb_isL_{caseName[case]}_{region}__ewk')
        
        # integrate from the bin where 400 is in to the last bin
        #counts[case][region]["data"]  = hist_data.Integral(hist_data.FindBin(400), 99)
        #counts[case][region]["major"] = hist_major.Integral(hist_major.FindBin(400), 99)
        #counts[case][region]["minor"] = hist_minor.Integral(hist_minor.FindBin(400), 99)

        counts[case][region]["data"]  = hist_data.Integral()
        counts[case][region]["major"] = hist_major.Integral()
        counts[case][region]["minor"] = hist_minor.Integral()
        
        #if "D" in region or "V" in region:
        counts[case][region]["unweighted"] = hist_major.GetEntries()

        tFile.Close()

    #for region in ["B", "D", "V", "BV", "highST", "BhighST"]: #, "B2", "D2"]: # general
    for region in ["B", "D", "V", "BV"]: # for year-by-year gof test
        print(f'Getting counts for region {region}')
        getCounts("case1", region)
        getCounts("case4", region)
        getCounts("case2", region)
        getCounts("case3", region)

        counts["case14"][region]["data"]  = counts["case1"][region]["data"]  + counts["case4"][region]["data"]
        counts["case23"][region]["data"]  = counts["case2"][region]["data"]  + counts["case3"][region]["data"]

        counts["case14"][region]["major"] = counts["case1"][region]["major"] + counts["case4"][region]["major"]
        counts["case23"][region]["major"] = counts["case2"][region]["major"] + counts["case3"][region]["major"]

        counts["case14"][region]["minor"] = counts["case1"][region]["minor"] + counts["case4"][region]["minor"]
        counts["case23"][region]["minor"] = counts["case2"][region]["minor"] + counts["case3"][region]["minor"]

        if region=="D" or region=="V":
            counts["case14"][region]["unweighted"]  = counts["case1"][region]["unweighted"]  + counts["case4"][region]["unweighted"]
            counts["case23"][region]["unweighted"]  = counts["case2"][region]["unweighted"]  + counts["case3"][region]["unweighted"]

    # store counts in a json file
    print(f'Writing to counts{year}{fileNameTag}.json...')
    json_object = json.dumps(counts, indent=4)
    with open(f'counts{year}{fileNameTag}.json', "w") as outfile:
        outfile.write(json_object)

        
    # alpha-ratio prediction
    yield_pred = {"case14":{"D":{}, "V":{}, "highST":{}, "D2":{}},
                  "case23":{"D":{}, "V":{}, "highST":{}, "D2":{}},
                  "case1":{"D":{}, "V":{}, "highST":{}, "D2":{}},
                  "case2":{"D":{}, "V":{}, "highST":{}, "D2":{}},
                  "case3":{"D":{}, "V":{}, "highST":{}, "D2":{}},
                  "case4":{"D":{}, "V":{}, "highST":{}, "D2":{}},
                  }

    def getPrediction(case, region):
        if region=="V":
            B = "BV"
            D = "V"
        elif region=="D":
            B = "B"
            D = "D"
        elif region=="D2":
            B = "B2"
            D = "D2"
        elif region=="highST":
            B = "BhighST"
            D = "highST"
        else:
            print(f'Region {region} not set up for alpha ratio calculation')
        print(f'Getting prediction for {case}...')
        target_B    = counts[case][B]["data"] - counts[case][B]["minor"]
        predict_D   = target_B * counts[case][D]["major"] / counts[case][B]["major"]
        
        if region=="D2" or region=="highST": # Might need to fix for D2, but highST prediction is only a dummy
            predict_val = predict_D
            target_val = counts[case][D]["data"] - counts[case][D]["minor"]
        else:
            predict_val = target_B * counts[case]["V"]["major"] / counts[case][B]["major"]
            target_val  = counts[case]["V"]["data"] - counts[case]["V"]["minor"]
            
        yield_pred[case][region]["prediction"]  = predict_D
        yield_pred[case][region]["factor"]      = predict_D / counts[case][D]["unweighted"]
        yield_pred[case][region]["systematic"]  = predict_D * np.sqrt(1/target_B + 1/counts[case][B]["major"] + 1/counts[case][D]["major"])
        yield_pred[case][region]["statistical"] = np.sqrt(predict_D)
        yield_pred[case][region]["closure"]     = abs(predict_val-target_val)
        yield_pred[case][region]["uncertainty"] = np.sqrt(yield_pred[case][region]["systematic"]**2 + yield_pred[case][region]["statistical"]**2 + yield_pred[case][region]["closure"]**2) / predict_D

        # TEMP: commented out for D2, because not generalized yet
        # print('Data:{}'.format(counts[case]["D"]["data"]))
        # print('Minor:{}'.format(counts[case]["D"]["minor"]))
        # print(f'Data-Minor:{target_val}')

        # print('Major from MC         :{}'.format(counts[case]["D"]["major"]))
        # print(f'Major from alpha-ratio:{predict_D}')

        print('Major deviation in MC: {}%'.format(round(100*(abs(counts[case][region]["major"]-target_val)/target_val),2)))
        print(f'Major deviation in alpha-ratio: {round(100*(yield_pred[case][region]["closure"]/target_val),2)}%')
        print(f'Total uncertainty from alpha-ratio (percentage): {round(100*yield_pred[case][region]["uncertainty"],2)}%\n')


    print("\nPerforming alpha-ratio estimation.\n")
    for case in ["case14", "case23","case1", "case2", "case3", "case4"]:
        #for region in ["V", "D", "highST"]: ##, "D2"]: # general
        for region in ["V","D"]: # for year-by-year gof test
            getPrediction(case, region)

    # write alpha-ratio restuls to a json file
    print(f'Writing to alphaRatio_factors{year}{fileNameTag}.json...')
    json_object = json.dumps(yield_pred, indent=4)
    with open(f'alphaRatio_factors{year}{fileNameTag}.json', "w") as outjson:
        outjson.write(json_object)
else:
    print("Skipping counts and alpha-ratio estimation...")
