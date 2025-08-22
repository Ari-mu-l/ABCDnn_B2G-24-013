import ROOT
import math

year = 'all'
#sample = ['Major', 'Signal']
mass = 1400
case = 'Case3'
if case in ['Case1','Case4','Case14']:
    rfileMajor = ROOT.TFile.Open(f'rootFiles_logBpMlogST_Jan2025Run2/Case14/JanMajor_{year}_mc_p100.root','READ')
    rfileSignal = ROOT.TFile.Open(f'rootFiles_logBpMlogST_Jan2025Run2/Case14/JanSignal{mass}_{year}_mc_p100.root','READ')
elif case in ['Case2','Case3','Case23']:
    rfileMajor = ROOT.TFile.Open(f'rootFiles_logBpMlogST_Jan2025Run2/Case23/JanMajor_{year}_mc_p100.root','READ')
    rfileSignal = ROOT.TFile.Open(f'rootFiles_logBpMlogST_Jan2025Run2/Case23/JanSignal{mass}_{year}_mc_p100.root','READ')

#bkgList = ['ewk','qcd','ttbar','ttx','wjets','singletop']

hist2DMajor = ROOT.TH2D('N_b_vs_N_forward_Major','N_b_vs_N_forward_Major',8,0,8,8,0,8)
hist2DSignal = ROOT.TH2D('N_b_vs_N_forward_Signal','N_b_vs_N_forward_Signal',8,0,8,8,0,8)
hist2DSignificance = ROOT.TH2D('N_b_vs_N_forward_Significance','N_b_vs_N_forward_Significance',8,0,8,8,0,8)
hist2DPurity = ROOT.TH2D('N_b_vs_N_forward_Purity','N_b_vs_N_forward_Purity',8,0,8,8,0,8)


regionBkg = {"A":0,"B":0,"C":0,"D":0,"X":0,"Y":0}
regionSig = {"A":0,"B":0,"C":0,"D":0,"X":0,"Y":0}
regionPur = {"A":0,"B":0,"C":0,"D":0,"X":0,"Y":0}

tTreeMajor = rfileMajor.Get('Events')
tTreeSignal = rfileSignal.Get('Events')

for evt in tTreeMajor:
    keepEvt = False
    if case=="Case1":
        if evt.Bdecay_obs==1:
            keepEvt = True
    elif case=="Case2":
        if evt.Bdecay_obs==2:
	        keepEvt = True
    elif case=="Case3":
        if evt.Bdecay_obs==3:
            keepEvt = True
    elif case=="Case4":
        if evt.Bdecay_obs==4:
            keepEvt = True
    elif case=="Case14" or case=="Case23":
        keepEvt = True

    if keepEvt:
        NJets_forward = evt.NJets_forward
        NJets_DeepFlavL = evt.NJets_DeepFlavL
        xsecWeight = evt.xsecWeight

        hist2DMajor.Fill(NJets_forward,NJets_DeepFlavL,xsecWeight)

        if NJets_forward == 0 and NJets_DeepFlavL == 3:
            regionBkg["A"]+=xsecWeight
        elif NJets_forward == 0 and NJets_DeepFlavL < 3:
            regionBkg["B"]+=xsecWeight
        elif NJets_forward > 0 and NJets_DeepFlavL == 3:
            regionBkg["C"]+=xsecWeight
        elif NJets_forward > 0 and NJets_DeepFlavL < 3:
            regionBkg["D"]+=xsecWeight
        elif NJets_forward == 0 and NJets_DeepFlavL > 3:
            regionBkg["X"]+=xsecWeight
        elif NJets_forward > 0 and NJets_DeepFlavL > 3:
            regionBkg["Y"]+=xsecWeight
    
print('Major histogram created.')

for evt in tTreeSignal:
    keepEvt = False
    if case=="Case1":
        if evt.Bdecay_obs==1:
            keepEvt = True
    elif case=="Case2":
        if evt.Bdecay_obs==2:
            keepEvt = True
    elif case=="Case3":
        if evt.Bdecay_obs==3:
            keepEvt = True
    elif case=="Case4":
        if evt.Bdecay_obs==4:
            keepEvt = True
    elif case=="Case14" or case=="Case23":
        keepEvt = True

    if keepEvt:
        NJets_forward = evt.NJets_forward
        NJets_DeepFlavL = evt.NJets_DeepFlavL
        xsecWeight = evt.xsecWeight

        hist2DSignal.Fill(evt.NJets_forward,evt.NJets_DeepFlavL,evt.xsecWeight)

        if NJets_forward == 0 and NJets_DeepFlavL == 3:
            regionSig["A"]+=xsecWeight
        elif NJets_forward == 0 and NJets_DeepFlavL < 3:
            regionSig["B"]+=xsecWeight
        elif NJets_forward > 0 and NJets_DeepFlavL == 3:
            regionSig["C"]+=xsecWeight
        elif NJets_forward > 0 and NJets_DeepFlavL < 3:
            regionSig["D"]+=xsecWeight
        elif NJets_forward == 0 and NJets_DeepFlavL > 3:
            regionSig["X"]+=xsecWeight
        elif NJets_forward > 0 and NJets_DeepFlavL > 3:
            regionSig["Y"]+=xsecWeight
        
print('Signal histogram created.')

c1 = ROOT.TCanvas('c1','c1')
hist2DMajor.Draw('COLZ')
c1.SaveAs(f'N_b_vs_N_forward_Major_{case}.png')

c2 = ROOT.TCanvas('c2','c2')
hist2DSignal.Draw('COLZ')
c2.SaveAs(f'N_b_vs_N_forward_Signal_{case}.png')

for i in range(1,9):
    for j in range(1,9):
        if hist2DMajor.GetBinContent(i,j)!=0:
            hist2DSignificance.SetBinContent(i,j,hist2DSignal.GetBinContent(i,j)/math.sqrt(hist2DMajor.GetBinContent(i,j)))
        else:
            hist2DSignificance.SetBinContent(i,j,0)

        if (hist2DSignal.GetBinContent(i,j)+hist2DMajor.GetBinContent(i,j))!=0:
            hist2DPurity.SetBinContent(i,j,hist2DSignal.GetBinContent(i,j)/(hist2DSignal.GetBinContent(i,j)+hist2DMajor.GetBinContent(i,j)))
        else:
            hist2DPurity.SetBinContent(i,j,0)


c3 = ROOT.TCanvas('c3','c3')
hist2DSignificance.Draw('COLZ')
c3.SaveAs(f'N_b_vs_N_forward_Significance_{case}.png')

c4 = ROOT.TCanvas('c4','c4')
hist2DPurity.Draw('COLZ')
c4.SaveAs(f'N_b_vs_N_forward_Purity_{case}.png')

for region in regionPur:
    regionPur[region] = regionSig[region]/(regionSig[region]+regionBkg[region])

print(regionPur)

rfileMajor.Close()
rfileSignal.Close()


