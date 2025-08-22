import ROOT
import os, json
#rootFiles_logBpMlogST_Jan2025Run2_withMCLabel/Case14/JanData_all_data_p100.root

ROOT.gROOT.SetBatch(True) # suppress histogram display

case = 'Case1'
region = 'V'
year = ''

if case=='Case1' or case=='Case4':
    directory = f'rootFiles_logBpMlogST_Jan2025Run2_withMCLabel/Case14'
elif case=='Case2' or case=='Case3':
    directory = f'rootFiles_logBpMlogST_Jan2025Run2_withMCLabel/Case23'
else:
    os.exit(f'{case} undefined.')

    
with open(f'alphaRatio_factors{year}_2Dpadfull.json',"r") as alphaFile:
    alphaFactors = json.load(alphaFile)
factor = alphaFactors[f'case{case[-1]}'][region]['factor']


sampleDir = {'1':'ttbar',
             '2':'wjets',
             '3':'qcd',
             '4':'singletop'}

caseCut = f'Bdecay_obs == {case[-1]}'
regionCut = 'NJets_forward > 0 && NJets_DeepFlavL < 3' # Region D
if region=='V':
    regionCut+=' && gcJet_ST<850'

ROOT.gInterpreter.Declare("""
float transform(float Bprime_mass){
return TMath::Exp(Bprime_mass);
}
""")

rdfMajor = ROOT.RDataFrame('Events', f'{directory}/JanMajor_all_mc_p100.root')\
                .Redefine("Bprime_mass", "transform(Bprime_mass)")\
                .Redefine("gcJet_ST", "transform(gcJet_ST)")\
                .Define('factor',f'{factor}')\
                .Filter(caseCut)\
                .Filter(regionCut)

rdfMinor = ROOT.RDataFrame('Events', f'{directory}/JanMinor_all_mc_p100.root')\
                .Redefine("Bprime_mass", "transform(Bprime_mass)")\
                .Redefine("gcJet_ST", "transform(gcJet_ST)")\
                .Filter(caseCut)\
                .Filter(regionCut)

rdfData = ROOT.RDataFrame('Events', f'{directory}/JanData_all_data_p100.root')\
              .Redefine("Bprime_mass", "transform(Bprime_mass)")\
              .Redefine("gcJet_ST", "transform(gcJet_ST)")\
              .Filter(caseCut)\
              .Filter(regionCut)

rdf_ttbar     = rdfMajor.Filter('sampleCategory==1')
rdf_wjets     = rdfMajor.Filter('sampleCategory==2')
rdf_qcd       = rdfMajor.Filter('sampleCategory==3')
rdf_singletop = rdfMajor.Filter('sampleCategory==4')

hist_ttbar     = rdf_ttbar.Histo1D(('Bprime_mass','Bprime_mass',210,0,2500),'Bprime_mass','factor')
hist_wjets     = rdf_wjets.Histo1D(('Bprime_mass','Bprime_mass',210,0,2500),'Bprime_mass','factor')
hist_qcd       = rdf_qcd.Histo1D(('Bprime_mass','Bprime_mass',210,0,2500),'Bprime_mass','factor')
hist_singletop = rdf_singletop.Histo1D(('Bprime_mass','Bprime_mass',210,0,2500),'Bprime_mass','factor')
hist_minor     = rdfMinor.Histo1D(('Bprime_mass','Bprime_mass',210,0,2500),'Bprime_mass')
hist_data      = rdfData.Histo1D(('Bprime_mass','Bprime_mass',210,0,2500),'Bprime_mass')

hist_bkg = ROOT.THStack("hist_bkg","")
    
# hist_ttbar.Scale(alphaFactors[f'case{case[-1]}'][region]["factor"])
# hist_wjets.Scale(alphaFactors[f'case{case[-1]}'][region]["factor"])
# hist_qcd.Scale(alphaFactors[f'case{case[-1]}'][region]["factor"])
# hist_singletop.Scale(alphaFactors[f'case{case[-1]}'][region]["factor"])

# Roughly according to the order in ANv7 plots
hist_bkg.Add(hist_qcd.GetPtr())
hist_bkg.Add(hist_minor.GetPtr())
hist_bkg.Add(hist_wjets.GetPtr())
hist_bkg.Add(hist_singletop.GetPtr())
hist_bkg.Add(hist_ttbar.GetPtr())
hist_bkg.Add(hist_minor.GetPtr())

hist_ttbar.SetFillColor(ROOT.kAzure+8)
hist_wjets.SetFillColor(ROOT.kMagenta-6)
hist_qcd.SetFillColor(ROOT.kOrange-3)
hist_singletop.SetFillColor(ROOT.kGreen-6)
hist_minor.SetFillColor(ROOT.kOrange-9)

hist_ttbar.SetLineColor(ROOT.kAzure+8)
hist_wjets.SetLineColor(ROOT.kMagenta-6)
hist_qcd.SetLineColor(ROOT.kOrange-3)
hist_singletop.SetLineColor(ROOT.kGreen-6)
hist_minor.SetLineColor(ROOT.kOrange-9)

legend = ROOT.TLegend(0.7, 0.7, 0.9, 0.9)
legend.AddEntry(hist_ttbar.GetPtr(),'t#bar{t}_ABCDnn','f')
legend.AddEntry(hist_wjets.GetPtr(),'W+jets_ABCDnn','f')
legend.AddEntry(hist_qcd.GetPtr(),'QCD_ABCDnn','f')
legend.AddEntry(hist_singletop.GetPtr(),'single t_ABCDnn','f')


c = ROOT.TCanvas()

hist_bkg.Draw()
if region=='V':
    hist_data.Draw('SAME')
legend.Draw()

c.SaveAs(f'ABCDnnWithCategories_{region}_{case}{year}.png')
