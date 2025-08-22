# this script takes an existing step3 ROOT file and formats it for use in the ABCDnn training script
# formats three types of samples: data (no weights), major MC background (no weights), and minor MC backgrounds (weights)
# last modified April 11, 2023 by Daniel Li

#python format_ntuple.py -y all -n OctMajor -p 100 --doMajorMC

import os, sys, ROOT
from array import array
from argparse import ArgumentParser
import config
import tqdm
#import xsec
from utils import *
from samples import *

parser = ArgumentParser()
parser.add_argument( "-y",  "--year", default = "all", help = "Year for sample" )
parser.add_argument( "-n", "--name", required = True, help = "Output name of ROOT file" )
#parser.add_argument( "-v",  "--variables", nargs = "+", default = [ "Bprime_mass", "gcLeadingOSFatJet_pNetJ" ], help = "Variables to transform" ) # change for new variables
parser.add_argument( "-v",  "--variables", nargs = "+", default = [ "Bprime_mass", "gcJet_ST" ], help = "Variables to transform" )
parser.add_argument( "-r",  "--nRuns", default = 10000, help = "Number of events to include in the output file" )
parser.add_argument( "-p",  "--pEvents", default = 100, help = "Percent of events (0 to 100) to include from each file." )
parser.add_argument( "-l",  "--location", default = "LPC", help = "Location of input ROOT files: LPC,BRUX" )
parser.add_argument( "--doMajorMC", action = "store_true", help = "Major MC background to be weighted using ABCDnn" )
parser.add_argument( "--doMinorMC", action = "store_true", help = "Minor MC background to be weighted using traditional SF" )
parser.add_argument( "--doClosureMC", action = "store_true", help = "Closure MC background weighted using traditional SF" )
parser.add_argument( "--doData", action = "store_true" )
parser.add_argument( "--JECup", action = "store_true", help = "Create an MC dataset using the JECup shift for ttbar" )
parser.add_argument( "--JECdown", action = "store_true", help = "Create an MC dataset using the JECdown shift for ttbar" )
args = parser.parse_args()

if args.location not in [ "LPC", "BRUX" ]: quit( "[ERR] Invalid -l (--location) argument used. Quitting..." )

ROOT.gInterpreter.Declare("""
    float compute_weight( float optionalWeights, float genWeight, float lumi, float xsec, float nRun, float PileupWeights, float leptonRecoSF, float leptonIDSF, float leptonIsoSF, float leptonHLTSF, float btagWeights ){
    return optionalWeights * PileupWeights * leptonRecoSF * leptonIDSF * leptonIsoSF * leptonHLTSF * btagWeights * genWeight * lumi * xsec / (nRun * abs(genWeight));
    }
""")

#ROOT.gInterpreter.Declare("""                                                                                                
#    float compute_weight_noSF( float genWeight, float lumi, float xsec, float nRun ){                                  
#    return genWeight * lumi * xsec / (nRun * abs(genWeight));
#    }
#""")

class ToyTree:
  def __init__( self, name, trans_var ):
    # trans_var is transforming variables
    self.name = name
    self.rFile = ROOT.TFile.Open( "{}.root".format( name ), "RECREATE" )
    self.rTree = ROOT.TTree( "Events", name )
    self.variables = { # variables that are used regardless of the transformation variables
      "xsecWeight": { "ARRAY": array( "f", [0] ), "STRING": "xsecWeight/F" } # might not needed. # CHECK
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

def getfChain( samplename, year ):
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

def format_ntuple( inputs, output, trans_var, doMCdata):
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
      else:
        samplename = sample.samplename.split('/')[1]
        if "JetHT" in samplename:
          samplename += sample.samplename.split('-')[0][-1]
      print( ">> Processing {}".format( samplename ) )
      fChain = getfChain( samplename, year )
      rDF = ROOT.RDataFrame(fChain)
      sample_total = rDF.Count().GetValue()
      filter_string = "" 
      scale = 1. / ( int( args.pEvents ) / 100. ) # isTraining == 3 is 20% of the total dataset # COMMENT: What is isTraining? # what is scale used for?
      for variable in ntuple.selection: 
        for i in range( len( ntuple.selection[ variable ]["CONDITION"] ) ):
          if filter_string == "": 
            filter_string += "( {} {} {} ) ".format( variable, ntuple.selection[ variable ][ "CONDITION" ][i], ntuple.selection[ variable ][ "VALUE" ][i] )
          else:
            filter_string += "|| ( {} {} {} ) ".format( variable, ntuple.selection[ variable ][ "CONDITION" ][i], ntuple.selection[ variable ][ "VALUE" ][i] )
      rDF_filter = rDF.Filter( filter_string )
    
      

      if args.doData:
        rDF_weight = rDF_filter.Define( "xsecWeight", "1.0")
      elif "_WJetsToLNu"  in samplename:
        rDF_weight = rDF_filter.Define( "xsecWeight", "compute_weight( {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {} )".format("gcHTCorr_WjetLHE[0]", "genWeight", targetlumi[year], sample.xsec, sample.nrun, "PileupWeights[0]", "leptonRecoSF[0]", "leptonIDSF[0]", "leptonIsoSF[0]", "leptonHLTSF[0]", "btagWeights[17]") )
      elif "TTToSemiLeptonic" in samplename:
        rDF_weight = rDF_filter.Define( "xsecWeight", "compute_weight( {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {} )".format("gcHTCorr_top[0]", "genWeight", targetlumi[year], sample.xsec, sample.nrun, "PileupWeights[0]", "leptonRecoSF[0]", "leptonIDSF[0]", "leptonIsoSF[0]", "leptonHLTSF[0]", "btagWeights[17]") )
      else:
        rDF_weight = rDF_filter.Define( "xsecWeight", "compute_weight( {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {} )".format("1.0", "genWeight", targetlumi[year], sample.xsec, sample.nrun, "PileupWeights[0]", "leptonRecoSF[0]", "leptonIDSF[0]", "leptonIsoSF[0]", "leptonHLTSF[0]", "btagWeights[17]") )

      if "gcLeadingOSFatJet_pNetJ" in config.variables.keys():
        rDF_weight = rDF_weight.Define("gcLeadingOSFatJet_pNetJ", "gcOSFatJet_pNetJ[0]")

      sample_pass = rDF_filter.Count().GetValue() # number of events passed the selection

      dict_filter = rDF_weight.AsNumpy( columns = list( ntuple.variables.keys() + [ "xsecWeight" ] ) ) # useful rdf branches to numpy
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
  format_ntuple( inputs = config.samples_input, output = args.name + "_" + args.year + "_mc_p" + args.pEvents, trans_var = args.variables, doMCdata = "MAJOR MC" )
elif args.doMinorMC:
  format_ntuple( inputs = config.samples_input, output = args.name + "_" + args.year + "_mc_p" + args.pEvents, trans_var = args.variables, doMCdata = "MINOR MC" )
elif args.doClosureMC:
  format_ntuple( inputs = config.samples_input, output = args.name + "_" + args.year + "_mc_p" + args.pEvents, trans_var = args.variables, doMCdata = "CLOSURE" )
elif args.doData:
  format_ntuple( inputs = config.samples_input, output = args.name + "_" + args.year + "_data_p" + args.pEvents, trans_var = args.variables, doMCdata = "DATA" )
