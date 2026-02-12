# python3 make1DHist_abcdnn_separateABCDnnCat.py -s rootFiles_logBpMlogST_Jan2025Run2_withMCLabel/Case14/JanMajor_all_mc_p100.root -b rootFiles_logBpMlogST_Jan2025Run2_withMCLabel/Case14/JanMinor_all_mc_p100.root -t rootFiles_logBpMlogST_Jan2025Run2_withMCLabel/Case14/JanData_all_data_p100.root -m logBpMlogST_mmd1_case14_random22

import numpy as np
import os, tqdm
import abcdnn
import uproot, ROOT
from argparse import ArgumentParser
from json import loads as load_json
from array import array
import samples
from utils import *

os.environ["KERAS_BACKEND"] = "tensorflow"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf
import config

parser = ArgumentParser()
parser.add_argument( "-s", "--source", required = True  )
parser.add_argument( "-t", "--target", required = True  )
parser.add_argument( "-b", "--minor" , required = True  )
parser.add_argument( "-m", "--tag"   , required = True  )
#parser.add_argument( "-c", "--case"  , required = False )

args = parser.parse_args()

# histogram settings
bin_lo = 400 #0
bin_hi = 2500
Nbins  = 210 # 210
validationCut = 850

byCategory = True
log = False # set to False if want BpM instead of log(BpM)
if log:
  validationCut = np.log(validationCut)
  bin_lo = np.log(bin_lo)
  bin_hi = np.log(bin_hi)

folder = config.params[ "MODEL" ][ "SAVEDIR" ]
folder_contents = os.listdir( folder )

isTest = True
if isTest:
  testDir = f'{args.tag}/'
  if not os.path.exists(testDir):
    os.makedirs(testDir)
else: testDir = ''

print( ">> Reading in {}.json for hyper parameters...".format( args.tag ) )
with open( os.path.join( folder, args.tag + ".json" ), "r" ) as f:
  params = load_json( f.read() )

print( ">> Setting up NAF model..." )

variables = [ str( key ) for key in sorted( config.variables.keys() ) if config.variables[key]["TRANSFORM"] ]
if config.regions["Y"]["VARIABLE"] in config.variables and config.regions["X"]["VARIABLE"] in config.variables:
  variables.append( config.regions["Y"]["VARIABLE"] )
  variables.append( config.regions["X"]["VARIABLE"] )
else:
  sys.exit( "[ERROR] Control variables not listed in config.variables, please add. Exiting..." )

categorical = [ config.variables[ vName ][ "CATEGORICAL" ] for vName in variables ]
lowerlimit  = [ config.variables[ vName ][ "LIMIT" ][0] for vName in variables ]
upperlimit  = [ config.variables[ vName ][ "LIMIT" ][1] for vName in variables ]

print( ">> Found {} variables: ".format( len( variables ) ) )
for i, variable in enumerate( variables ):
  print( "  + {}: [{},{}], Categorical = {}".format( variable, lowerlimit[i], upperlimit[i], categorical[i] ) )

def processInput(fileName):
  # fileName: args.target, args.minor, args.source
  print( ">> Load the data" )
  fFile = uproot.open( fileName )
  fTree = fFile[ "Events" ]
  
  if "Major" in fileName:
    inputs = fTree.arrays( variables, library="pd" )
  elif "Minor" in fileName:
    inputs = fTree.arrays( variables+["xsecWeight"], library="pd" )
    for variable in variables:
      inputs[variable] = inputs[variable].clip(upper = config.variables[variable]["LIMIT"][1])
  elif "Data" in fileName:
    inputs = fTree.arrays( variables, library="pd" )
    for variable in variables:
      inputs[variable] = inputs[variable].clip(upper = config.variables[variable]["LIMIT"][1])

  #Bdecay = fTree.arrays( ["Bdecay_obs", "pNetTtagWeight", "pNetTtagWeightUp", "pNetTtagWeightDn", "pNetWtagWeight", "pNetWtagWeightUp", "pNetWtagWeightDn","sampleCategory"], library="pd" )
  Bdecay = fTree.arrays( ["Bdecay_obs", "sampleCategory"], library="pd" )

  ##Bdecay_tgt = tTree.arrays( ["Bdecay_obs"], library="pd" )[inputs_tgt["Bprime_mass"]>400]
  ##Bdecay_mnr = mTree.arrays( ["Bdecay_obs"], library="pd" )[inputs_mnr["Bprime_mass"]>400]
  ##inputs_tgt = inputs_tgt[inputs_tgt["Bprime_mass"]>400] # take only BpM>400. pred cut made later.
  ##inputs_mnr = inputs_mnr[inputs_mnr["Bprime_mass"]>400]

  inputs_region = { region: None for region in [ "D" ] }
  Bdecay_region = { region: None for region in [ "D" ] }

  #inputs_region = { region: None for region in [ "X", "Y", "A", "B", "C", "D" ] }
  #Bdecay_region = { region: None for region in [ "X", "Y", "A", "B", "C", "D" ] }

  #for region in [ "X", "Y", "A", "B", "C", "D" ]:
  for region in [ "D" ]:
    if config.regions["X"][region][0] == ">=":
      select = inputs[ config.regions["X"]["VARIABLE"] ] >= config.regions[ "X" ][ region ][1]
    elif config.regions["X"][region][0] == "<=":
      select = inputs[ config.regions["X"]["VARIABLE"] ] <= config.regions[ "X" ][ region ][1]
    else:
      select = inputs[ config.regions["X"]["VARIABLE"] ] == config.regions[ "X" ][ region ][1]

    if config.regions["Y"][region][0] == ">=":
      select &= inputs[ config.regions["Y"]["VARIABLE"] ] >= config.regions[ "Y" ][ region ][1]
    elif config.regions["Y"][region][0] == "<=":
      select &= inputs[ config.regions["Y"]["VARIABLE"] ] <= config.regions[ "Y" ][ region ][1]
    else:
      select &= inputs[ config.regions["Y"]["VARIABLE"] ] == config.regions[ "Y" ][ region ][1]

    inputs_region[region] = inputs.loc[select]
    Bdecay_region[region] = Bdecay.loc[select]

  if "Major" in fileName:
    print( ">> Encoding and normalizing source inputs" )
    inputs_enc_region = {}
    encoder = {}
    inputs_nrm_region = {}
    inputmeans = np.hstack( [ float( mean ) for mean in params[ "INPUTMEANS" ] ] )
    inputsigmas = np.hstack( [ float( sigma ) for sigma in params[ "INPUTSIGMAS" ] ] )

    for region in inputs_region:
      encoder[region] = abcdnn.OneHotEncoder_int( categorical, lowerlimit = lowerlimit, upperlimit = upperlimit )
      inputs_enc_region[ region ] = encoder[region].encode( inputs_region[ region ].to_numpy( dtype = np.float32 ) )
      inputs_nrm_region[ region ] = ( inputs_enc_region[ region ] - inputmeans ) / inputsigmas
      inputs_region[ region ][ variables[0] ] = inputs_region[ region ][ variables[0] ].clip(upper=config.variables[variables[0]]["LIMIT"][1])
      inputs_region[ region ][ variables[1] ] = inputs_region[ region ][ variables[1] ].clip(upper=config.variables[variables[1]]["LIMIT"][1])
  
    #get predictions
    #predictions = { region: [] for region in [ "X", "Y", "A", "B", "C", "D" ] }
    predictions = { region: [] for region in [ "D" ] }

    NAF = abcdnn.NAF( 
      inputdim = params["INPUTDIM"],
      conddim = params["CONDDIM"],
      activation = params["ACTIVATION"], 
      regularizer = params["REGULARIZER"],
      initializer = params["INITIALIZER"],
      nodes_cond = params["NODES_COND"],
      hidden_cond = params["HIDDEN_COND"],
      nodes_trans = params["NODES_TRANS"],
      depth = params["DEPTH"],
      permute = bool( params["PERMUTE"] )
    )
    NAF.load_weights( os.path.join( folder, args.tag ) )

    for region in tqdm.tqdm(predictions):
      NAF_predict = np.asarray( NAF.predict( np.asarray( inputs_nrm_region[ region ] ) ) )
      predictions[ region ] = NAF_predict * inputsigmas[0:2] + inputmeans[0:2]
      ##Bdecay_src_region[region] = Bdecay_src_region[region][predictions[ region ][:,0]>400]
      ##predictions[ region ] = predictions[region][predictions[ region ][:,0]>400] # take only BpM>400
    del NAF

  if "Major" in fileName:
    return predictions, Bdecay_region
  else:
    return inputs_region, Bdecay_region

def fillFullST(fillpNetShift, inputs_array, weight_array, hist_ttbar, hist_wjets, hist_qcd, hist_singletop, category_array):
  for i in range(len(inputs_array)):
    if inputs_array[i][0]>bin_lo:
      if category_array[i]==1:
        hist_ttbar.Fill(inputs_array[i][0])
      elif category_array[i]==2:
        hist_wjets.Fill(inputs_array[i][0])
      elif category_array[i]==3:
        hist_qcd.Fill(inputs_array[i][0])
      elif category_array[i]==4:
        hist_singletop.Fill(inputs_array[i][0])
  hist_ttbar.Write()
  hist_wjets.Write()
  hist_qcd.Write()
  hist_singletop.Write()

def fillLowSTByCategory(fillpNetShift, inputs_array, weight_array, hist_ttbar, hist_wjets, hist_qcd, hist_singletop, category_array):
  for i in range(len(inputs_array)):
    if inputs_array[i][0]>bin_lo and inputs_array[i][1]<validationCut:
      if category_array[i]==1:
        hist_ttbar.Fill(inputs_array[i][0])
      elif category_array[i]==2:
        hist_wjets.Fill(inputs_array[i][0])
      elif category_array[i]==3:
        hist_qcd.Fill(inputs_array[i][0])
      elif category_array[i]==4:
        hist_singletop.Fill(inputs_array[i][0])
  hist_ttbar.Write()
  hist_wjets.Write()
  hist_qcd.Write()
  hist_singletop.Write()
  
  # if fillpNetShift:
  #   for i in range(len(inputs_array)):
  #     if inputs_array[i][0]>bin_lo and inputs_array[i][1]<validationCut:
  #       hist_pNetUp.Fill(inputs_array[i][0], pNetUp_array[i]) # pNetSF_Up
  #       hist_pNetDn.Fill(inputs_array[i][0], pNetDn_array[i]) # pNetSF_Dn
  #       hist_pNetUp.Write()
  #   hist_pNetDn.Write()
        
def make1DHists(fileName, inputs_region, Bdecay_region, region, case):
  caseValue = int(case[-1])
  
  if "Major" in fileName:
    inputs_array = inputs_region[region][ Bdecay_region[region]["Bdecay_obs"]==caseValue ] # name prediction_array as inputs_array
    weight_array = np.ones(len(inputs_array))
    category_array = Bdecay_region[region]["sampleCategory"][ Bdecay_region[region]["Bdecay_obs"]==caseValue ].to_numpy(dtype='d')
  elif "Minor" in fileName:
    inputs_array = inputs_region[region][ Bdecay_region[region]["Bdecay_obs"]==caseValue ].to_numpy(dtype='d')[:,:2]
    weight_array = inputs_region[region]["xsecWeight"][ Bdecay_region[region]["Bdecay_obs"]==caseValue ].to_numpy(dtype='d')
  else: # data
    inputs_array = inputs_region[region][ Bdecay_region[region]["Bdecay_obs"]==caseValue ].to_numpy(dtype='d')[:,:2]
    weight_array = np.ones(len(inputs_array))

  print('Gotten arrays')

  # if case=="case1" or case=="case2":
  #   if "Data" in fileName:
  #     pNet_array   = np.ones(len(inputs_array))
  #     pNetUp_array = np.ones(len(inputs_array))
  #     pNetDn_array = np.ones(len(inputs_array))
  #   else: # minor, major: case 1 or 2
  #     if case=="case1":
  #       pNetTag = "pNetTtagWeight"
  #     else:
  #       pNetTag = "pNetWtagWeight"
  #     pNet_array = Bdecay_region[region][pNetTag][ Bdecay_region[region]["Bdecay_obs"]==caseValue ].to_numpy(dtype='d')
  #     pNetUp_array = Bdecay_region[region][f'{pNetTag}Up'][ Bdecay_region[region]["Bdecay_obs"]==caseValue ].to_numpy(dtype='d')
  #     pNetDn_array = Bdecay_region[region][f'{pNetTag}Dn'][ Bdecay_region[region]["Bdecay_obs"]==caseValue ].to_numpy(dtype='d')
  # else: # case3,4
  #   pNet_array   = np.ones(len(inputs_array))
  #   pNetUp_array = np.ones(len(inputs_array))
  #   pNetDn_array = np.ones(len(inputs_array))

  print('Gotten pNetArrays')
    
  if not log:
    inputs_array = np.exp(inputs_array)
    inputs_array = np.clip(inputs_array, 0, 2500)

  if region=="D":
    #subRegionList = ['D', 'V', 'D2']
    subRegionList = ['D','V']
  #else:
  #  subRegionList = [f'{region}', f'{region}V', f'{region}2'] # low ST: append V. high ST: append 2. e.g. B, BV, B2

  for regionTag in subRegionList:
    fillpNetShift = False
    if "Data" in fileName:
      hist = ROOT.TH1D(f'Bprime_mass_dat_{regionTag}', "Bprime_mass", Nbins, bin_lo, bin_hi)
    elif "Minor" in fileName:
      hist = ROOT.TH1D(f'Bprime_mass_mnr_{regionTag}', "Bprime_mass", Nbins, bin_lo, bin_hi)
    elif "Major" in fileName:
      if byCategory:
        hist_ttbar     = ROOT.TH1D(f'Bprime_mass_pre_{regionTag}_ttbar', "Bprime_mass_ABCDnn", Nbins, bin_lo, bin_hi)
        hist_wjets     = ROOT.TH1D(f'Bprime_mass_pre_{regionTag}_wjets', "Bprime_mass_ABCDnn", Nbins, bin_lo, bin_hi)
        hist_qcd       = ROOT.TH1D(f'Bprime_mass_pre_{regionTag}_qcd', "Bprime_mass_ABCDnn", Nbins, bin_lo, bin_hi)
        hist_singletop = ROOT.TH1D(f'Bprime_mass_pre_{regionTag}_singletop', "Bprime_mass_ABCDnn", Nbins, bin_lo, bin_hi)
      else:
        hist = ROOT.TH1D(f'Bprime_mass_pre_{regionTag}', "Bprime_mass_ABCDnn", Nbins, bin_lo, bin_hi)
      #if case=="case1" or case=="case2": # fill pNetShift only for major bkg. mnr is dealt with in SLA
      #  fillpNetShift = True

    # create for all. but only fill for the necessary cases
    #hist_pNetUp = ROOT.TH1D(f'Bprime_mass_pre_{regionTag}_pNetUp', "Bprime_mass_ABCDnn", Nbins, bin_lo, bin_hi)
    #hist_pNetDn = ROOT.TH1D(f'Bprime_mass_pre_{regionTag}_pNetDn', "Bprime_mass_ABCDnn", Nbins, bin_lo, bin_hi)

    print('Gotten hist_pNetUp/Dn')
    
    if regionTag=="V":
      if byCategory and ("Major" in fileName):
        print('filling fillLowSTByCategory')
        fillLowSTByCategory(fillpNetShift, inputs_array, weight_array, hist_ttbar, hist_wjets, hist_qcd, hist_singletop, category_array)
        #fillLowSTByCategory(fillpNetShift, inputs_array, weight_array, pNet_array, pNetUp_array, pNetDn_array, hist_ttbar, hist_wjets, hist_qcd, hist_singletop, hist_pNetUp, hist_pNetDn, category_array)
        print('filled fillLowSTByCategory')
      else:
        print('filling fillLowST')
        fillLowST(fillpNetShift, inputs_array, weight_array, hist)
        #fillLowST(fillpNetShift, inputs_array, weight_array, pNet_array, pNetUp_array, pNetDn_array, hist, hist_pNetUp, hist_pNetDn)
        print('filled fillLowST')
    elif "2" in regionTag:
      fillHighST(fillpNetShift, inputs_array, weight_array, pNet_array, pNetUp_array, pNetDn_array, hist, hist_pNetUp, hist_pNetDn)
    else:
      fillFullST(fillpNetShift, inputs_array, weight_array, hist_ttbar, hist_wjets, hist_qcd, hist_singletop, category_array)
        
if 'case14' in args.tag:
  case_list = ["case1", "case4"]
elif 'case23' in args.tag:
  case_list = ["case2", "case3"]

def main(fileName, tfileOption):
  inputs_region, Bdecay_region = processInput(fileName)

  print('finished inputs_region, Bdecay_region = processInput(fileName)')

  for case in case_list:
    if log:
      outputFileName = f'{testDir}hists_ABCDnn_{case}_{bin_lo}to{bin_hi}_{Nbins}_log_pNet.root'
    else:
      outputFileName = f'{testDir}hists_ABCDnn_{case}_{bin_lo}to{bin_hi}_{Nbins}_pNet.root'

    if byCategory:
      outputFileName = outputFileName.replace('.root', '_byCategory.root')


    histFile = ROOT.TFile.Open(outputFileName, tfileOption)

    print(f'finished histFile = ROOT.TFile.Open({outputFileName}, {tfileOption})')

    #make1DHists(fileName, inputs_region, Bdecay_region, "A", case)
    #make1DHists(fileName, inputs_region, Bdecay_region, "B", case)
    #make1DHists(fileName, inputs_region, Bdecay_region, "C", case)
    #make1DHists(fileName, inputs_region, Bdecay_region, "D", case)

    make1DHists(fileName, inputs_region, Bdecay_region, "D", case)

    print(f'finished make1DHists(fileName, inputs_region, Bdecay_region, "D", {case})')
    
    histFile.Close()

#main(args.target, "recreate")
#print('finished main(args.target, "recreate")')
#main(args.minor, "update")
#print('finished main(args.minor, "update")')
main(args.source, "recreate")
print('finished main(args.source, "recreate")')
