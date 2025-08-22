# this script is run on the condor node for applying the trained ABCDnn model to ttbar samples
# last updated 11/15/2021 by Daniel Li
# python copy_untouchedTrees.py -y 2016 -t Runs

import numpy as np
import subprocess
import os
import uproot
import sys
from argparse import ArgumentParser
from json import loads as load_json
from array import array
import multiprocessing
from tqdm.auto import tqdm

os.environ["KERAS_BACKEND"] = "tensorflow"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf
import config

parser = ArgumentParser()
parser.add_argument( "-f", "--condor", default = "Single file submission for Condor jobs" )
parser.add_argument( "-y", "--year", required = True )
parser.add_argument( "-t", "--treeName", required = True )
parser.add_argument( "-l", "--location", default = "LPC" )
parser.add_argument( "-s", "--storage", default = "EOS", help = "LOCAL,EOS,BRUX" )
args = parser.parse_args()

import ROOT

# populate the step 3
def fill_tree( sample, treeName ):
  sampleDir = config.sampleDir

  print( "[START] Formatting sample: {}".format( sample ) )
  sample = os.path.join( config.sourceDir[ args.location ], sample )

  rFile_in = ROOT.TFile.Open( sample )
  rTree_in = rFile_in.Get( treeName )

  rFile_out = ROOT.TFile.Open( (sample.replace("jmanagan","xshen")).replace("BtoTW_Oct2023_fullRun2", "BtoTW_Oct2023_fullRun2_ABCDnn"),  "UPDATE" )
  if (rFile_out.GetListOfKeys().Contains(treeName)):
    del rTree_in, rFile_in, rFile_out
  else:
    rFile_out.cd()
    rTree_out = rTree_in.CloneTree(0)

    rTree_out.Write()
    #rFile_out.Write()
    rFile_out.Close()
    
    del rTree_in, rFile_in, rTree_out, rFile_out

for sample in config.samples_apply[ args.year ]:
    fill_tree( sample, args.treeName )
