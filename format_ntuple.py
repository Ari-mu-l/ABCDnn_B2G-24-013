# this script takes an existing step3 ROOT file and formats it for use in the ABCDnn training script
# formats three types of samples: data (no weights), major MC background (no weights), and minor MC backgrounds (weights)
# last modified April 11, 2023 by Daniel Li

# python3 format_ntuple.py -y all -n JanMajor -p 100 -c case14 --doMajorMC

import os, sys, ROOT
from array import array
from argparse import ArgumentParser
import config
import tqdm
from utils import *
from samples import *

parser = ArgumentParser()
parser.add_argument( "-y",  "--year", default = "2018", help = "Year for sample" )
parser.add_argument( "-n", "--name", required = True, help = "Output name of ROOT file" )
#parser.add_argument( "-v",  "--variables", nargs = "+", default = [ "Bprime_mass", "OS1FatJetProbJ", "gcJet_ST" ], help = "Variables to transform" ) # change for new variables
parser.add_argument( "-v",  "--variables", nargs = "+", default = [ "Bprime_mass", "gcJet_ST" ], help = "Variables to transform" )
#parser.add_argument( "-v",  "--variables", nargs = "+", default = [ "Bprime_mass", "Bprime_ptbal" ], help = "Variables to transform")
parser.add_argument( "-p",  "--pEvents", default = 100, help = "Percent of events (0 to 100) to include from each file." )
parser.add_argument( "-l",  "--location", default = "LPC", help = "Location of input ROOT files: LPC,BRUX" )
parser.add_argument( "-c", "--case", required = True, help = "decay mode")
parser.add_argument( "--doMajorMC", action = "store_true", help = "Major MC background to be weighted using ABCDnn" )
parser.add_argument( "--doMinorMC", action = "store_true", help = "Minor MC background to be weighted using traditional SF" )
parser.add_argument( "--doClosureMC", action = "store_true", help = "Closure MC background weighted using traditional SF" )
parser.add_argument( "--doData", action = "store_true" )
parser.add_argument( "--doSignalMC", action = "store_true" )
parser.add_argument( "--JECup", action = "store_true", help = "Create an MC dataset using the JECup shift for ttbar" )
parser.add_argument( "--JECdown", action = "store_true", help = "Create an MC dataset using the JECdown shift for ttbar" )
args = parser.parse_args()

if args.location not in [ "LPC", "BRUX" ]: quit( "[ERR] Invalid -l (--location) argument used. Quitting..." )

# No pNet SF added to xsecWeight. Apply it in the plotting script. 
ROOT.gInterpreter.Declare("""
    float compute_weight( float optionalWeight, float L1PreFiringWeight_Nom, float genWeight, float lumi, float xsec, float nRun, float PileupWeights, float leptonRecoSF, float leptonIDSF, float leptonIsoSF, float leptonHLTSF, float btagWeights, float puJetSF){
    return optionalWeight * PileupWeights * L1PreFiringWeight_Nom * leptonIDSF * leptonRecoSF * leptonIsoSF * leptonHLTSF * btagWeights * puJetSF * genWeight * lumi * xsec / (nRun * abs(genWeight));
    }
""")

ROOT.gInterpreter.Declare("""
float transform(float Bprime_mass){
return TMath::Log(Bprime_mass);
}
""")

class ToyTree:
  def __init__( self, name, trans_var ):
    self.name = name
    self.rFile = ROOT.TFile.Open( "{}.root".format( name ), "RECREATE" )
    self.rTree = ROOT.TTree( "Events", name )
    self.variables = { # variables that are used regardless of the transformation variables
      "xsecWeight": { "ARRAY": array( "f", [0] ), "STRING": "xsecWeight/F" },
      "Bdecay_obs": { "ARRAY": array( "i", [0] ), "STRING": "Bdecay_obs/I" },
      "pNetTtagWeight": {"ARRAY": array( "f", [0] ), "STRING": "pNetTtagWeight/F" },
      "pNetTtagWeightUp": {"ARRAY": array( "f", [0] ), "STRING": "pNetTtagWeightUp/F" },
      "pNetTtagWeightDn": {"ARRAY": array( "f", [0] ), "STRING": "pNetTtagWeightDn/F" },
      "pNetWtagWeight": {"ARRAY": array( "f", [0] ), "STRING": "pNetWtagWeight/F" },
      "pNetWtagWeightUp": {"ARRAY": array( "f", [0] ), "STRING": "pNetWtagWeightUp/F" },
      "pNetWtagWeightDn": {"ARRAY": array( "f", [0] ), "STRING": "pNetWtagWeightDn/F" },
      "sampleCategory": { "ARRAY": array( "i", [0] ), "STRING": "sampleCategory/I" },
    }
    for variable in config.variables.keys():
      if not config.variables[ variable ][ "TRANSFORM" ]:
        self.variables[ variable ] = { "ARRAY": array( "i", [0] ), "STRING": str(variable) + "/I" } # MODIFY DATATYPE
    
    for variable in trans_var:
      self.variables[ variable ] = { "ARRAY": array( "f", [0] ), "STRING": "{}/F".format( variable ) }
   
    self.selection = config.selection 
    
    for variable in self.variables:
      self.rTree.Branch( str( variable ), self.variables[ variable ][ "ARRAY" ], self.variables[ variable ][ "STRING" ] ) # create a tree with only useful branches
    
  def Fill( self, event_data ): # fill all tree branches for each event
    for variable in self.variables:
      self.variables[ variable ][ "ARRAY" ][0] = event_data[ variable ]
    self.rTree.Fill()
  
  def Write( self ):
      print( ">> Writing {} entries to {}.root".format( self.rTree.GetEntries(), self.name ) )
      self.rFile.Write()
      self.rFile.Close()

def getfChain( output, samplename, year ):
  if ( args.JECup or args.JECdown ) and "data" in output:
    print( "[WARNING] Ignoring JECup and/or JECdown arguments for data" )
    fChain = readTreeNominal( samplename, year, config.sourceDir["LPC"],"Events_Nominal" ) # read rdf for processing
  elif args.JECup and not args.JECdown:
    print( "[INFO] Running with JECup samples" )
    fChain = readTreeNominal( samplename, year, config.sourceDir["LPC"],"Events_JECup" )
    output = output.replace( "mc", "mc_JECup" )
  elif not args.JECup and args.JECdown:
    print( "[INFO] Running with JECdn samples" )
    fChain = readTreeNominal( samplename, year, config.sourceDir["LPC"],"Events_JECdn" )
    output = output.replace( "mc", "mc_JECdn" )
  elif args.JECdown and args.JECup:
    sys.exit( "[WARNING] Cannot run with both JECup and JECdown options. Select only one or none. Quitting..." )
  else:
    fChain = readTreeNominal( samplename, year, config.sourceDir["LPC"],"Events_Nominal" )
  return fChain

def format_ntuple( inputs, output, trans_var, doMCdata ):
  ntuple = ToyTree( output, trans_var )

  if (args.year == "all"):
    years = ["2016APV", "2016", "2017", "2018"]
  else:
    years = [args.year]

  for year in years:
    print("Year {}".format(year))
    for sample in inputs[year][doMCdata]:

      if (args.doData):
        samplename = sample.samplename.split('/')[1]+sample.samplename.split('-')[0][-1]
        sampleCategory = -1
      else:
        samplename = sample.samplename.split('/')[1]
        #if "JetHT" in samplename:  # not using JetHT in ABCDnn
        #  samplename += sample.samplename.split('-')[0][-1]

        # Requested by ARC: Plot separate backgrounds
        if sample.prefix in samples_ttbar_abcdnn:
          print(sample.prefix, 'ttbar')
          sampleCategory = 1
        elif sample.prefix in samples_wjets:
          print(sample.prefix, 'wjets')
          sampleCategory = 2
        elif sample.prefix in samples_qcd:
          print(sample.prefix, 'qcd')
          sampleCategory = 3
        elif sample.prefix in samples_singletop:
          print(sample.prefix, 'singletop')
          sampleCategory = 4
        else:
          sampleCategory = 0
          
      print( ">> Processing {}".format( samplename ) )
      fChain = getfChain( output, samplename, year )
      #rDF = ROOT.RDataFrame(fChain)
      # Uncomment the following only if we want to pre-transform BpM. e.g. log(BpM)
      rDF = ROOT.RDataFrame(fChain)\
                .Redefine("Bprime_mass", "transform(Bprime_mass)")\
                .Redefine("gcJet_ST", "transform(gcJet_ST)")
      sample_total = rDF.Count().GetValue()
      filter_string = "(W_MT < 200) && (isMu || isEl)" # Julie has W_MET<200. Trained with <=, but now applying with <. Removed = when addding pNetWeights
      if 'SingleMuon' in samplename:
        filter_string += "&& (isMu==1)"
      elif ('SingleElec' in samplename) or ('EGamma' in samplename):
        filter_string += "&& (isEl==1)"
      scale = 1. / ( int( args.pEvents ) / 100. ) # isTraining == 3 is 20% of the total dataset # COMMENT: What is isTraining? # what is scale used for?
      filter_string += " && ( {} ) ".format( ntuple.selection[ args.case ] )
      print('filter_string: {}'.format(filter_string))
      rDF_filter = rDF.Filter( filter_string )
      
      if args.doData:
        rDF_weight = rDF_filter.Define( "xsecWeight", "1.0")\
                               .Define("pNetTtagWeight"  , "1.0")\
                               .Define("pNetTtagWeightUp", "1.0")\
                               .Define("pNetTtagWeightDn", "1.0")\
                               .Define("pNetWtagWeight"  , "1.0")\
                               .Define("pNetWtagWeightUp", "1.0")\
                               .Define("pNetWtagWeightDn", "1.0")\
                               .Define("sampleCategory", f'{sampleCategory}')
      else:
        rDF_pNetWeight = rDF_filter.Define("pNetTtagWeight"  , "gcFatJet_pnetweights[6]")\
                                   .Define("pNetTtagWeightUp", "gcFatJet_pnetweights[7]")\
                                   .Define("pNetTtagWeightDn", "gcFatJet_pnetweights[8]")\
                                   .Define("pNetWtagWeight"  , "gcFatJet_pnetweights[9]")\
                                   .Define("pNetWtagWeightUp", "gcFatJet_pnetweights[10]")\
                                   .Define("pNetWtagWeightDn", "gcFatJet_pnetweights[11]")\
                                   .Define("sampleCategory", f'{sampleCategory}')
        # if args.case=="case14":
        #   pNetTagWeight = "pNetTtagWeight"
        # elif args.case=="case23":
        #   pNetTagWeight = "pNetWtagWeight"
        # else:
        #   print("Case not considered...")
        #   exit()
          
        if "_WJetsHT"  in samplename:
          rDF_weight = rDF_pNetWeight.Define( "xsecWeight", "compute_weight( {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {} )".format("gcHTCorr_WjetLHE[0]", "L1PreFiringWeight_Nom", "genWeight", targetlumi[year], sample.xsec, sample.nrun, "PileupWeights[0]", "leptonRecoSF[0]", "leptonIDSF[0]", "leptonIsoSF[0]", "leptonHLTSF[0]", "btagWeights[17]", "puJetSF[0]") )
        elif "TTTo" in samplename or 'TTMT' in sample.prefix:
          rDF_weight = rDF_pNetWeight.Define( "xsecWeight", "compute_weight( {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {} )".format("gcHTCorr_top[0]", "L1PreFiringWeight_Nom", "genWeight", targetlumi[year], sample.xsec, sample.nrun, "PileupWeights[0]", "leptonRecoSF[0]", "leptonIDSF[0]", "leptonIsoSF[0]", "leptonHLTSF[0]", "btagWeights[17]", "puJetSF[0]") )
        else:
          rDF_weight = rDF_pNetWeight.Define( "xsecWeight", "compute_weight( {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {} )".format("1.0", "L1PreFiringWeight_Nom", "genWeight", targetlumi[year], sample.xsec, sample.nrun, "PileupWeights[0]", "leptonRecoSF[0]", "leptonIDSF[0]", "leptonIsoSF[0]", "leptonHLTSF[0]", "btagWeights[17]", "puJetSF[0]") )

        
      if "OS1FatJetProbJ" in config.variables.keys():
        rDF_weight = rDF_weight.Define("OS1FatJetProbJ", "gcOSFatJet_pNetJ[0]")

      sample_pass = rDF_filter.Count().GetValue() # number of events passed the selection
      dict_filter = rDF_weight.AsNumpy( columns = list( ntuple.variables.keys() ) + [ "xsecWeight", "Bdecay_obs", "pNetTtagWeight", "pNetTtagWeightUp", "pNetTtagWeightDn", "pNetWtagWeight", "pNetWtagWeightUp", "pNetWtagWeightDn"] )
      del rDF, rDF_filter, rDF_weight
      n_inc = int( sample_pass * float( args.pEvents ) / 100. ) # get a specified portion of the passed events
 
      for n in tqdm.tqdm( range( n_inc ) ):
        event_data = {}
        for variable in dict_filter:
          event_data[ variable ] = dict_filter[ variable ][n] 

        ntuple.Fill( event_data )

      print( ">> {}/{} events saved...".format( n_inc, sample_total ) )
  ntuple.Write()
  
if args.doMajorMC:
  format_ntuple( inputs = config.samples_input, output = args.name + "_" + args.year + "_mc_p" + args.pEvents + "_" + args.case, trans_var = args.variables, doMCdata = "MAJOR MC" )
elif args.doMinorMC:
  format_ntuple( inputs = config.samples_input, output = args.name + "_" + args.year + "_mc_p" + args.pEvents + "_" + args.case, trans_var = args.variables, doMCdata = "MINOR MC" )
elif args.doClosureMC:
  format_ntuple( inputs = config.samples_input, output = args.name + "_" + args.year + "_mc_p" + args.pEvents + "_" + args.case, trans_var = args.variables, doMCdata = "CLOSURE" )
elif args.doData:
  format_ntuple( inputs = config.samples_input, output = args.name + "_" + args.year + "_data_p" + args.pEvents + "_" + args.case, trans_var = args.variables, doMCdata = "DATA" )
elif args.doSignalMC:
  format_ntuple( inputs = config.samples_input, output = args.name + "_" + args.year + "_mc_p" + args.pEvents + "_" + args.case, trans_var = args.variables, doMCdata = "SIGNAL MC" )
