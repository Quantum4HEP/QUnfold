# ---------------------- Metadata ----------------------
#
# File name:  processing.py
# Author:     Gianluca Bianco (biancogianluca9@gmail.com)
# Date:       2023-11-16
# Copyright:  (c) 2023 Gianluca Bianco under the MIT license.

# Simulated process: gg -> tt~ -> WbWb -> l l~ nu nu~ b b~

# TODO: introdurre bias?
# TODO: barre di errore
# TODO: hit miss
# TODO: fakes

# STD libraries
import argparse as ap
from tqdm import tqdm
from array import array
import sys

# Data science modules
import uproot
import awkward as ak
import ROOT

# ROOT settings
ROOT.gROOT.SetBatch(True)

# RooUnfold settings
loaded_RooUnfold = ROOT.gSystem.Load("HEP_deps/RooUnfold/libRooUnfold.so")
if not loaded_RooUnfold == 0:
    print("RooUnfold not found!")
    sys.exit(0)


def get_lep_pT_at_particle_level(event):
    lepton_mask = (abs(event["Particle.PID"]) == 11) | (abs(event["Particle.PID"]) == 13)
    sorted_leptons_pT = ak.argsort(event["Particle.PT"][lepton_mask], axis=-1, ascending=False)
    leading_lepton_pT = event["Particle.PT"][lepton_mask][sorted_leptons_pT[0]]
    subleading_lepton_pT = event["Particle.PT"][lepton_mask][sorted_leptons_pT[1]]

    return leading_lepton_pT, subleading_lepton_pT


def get_trees_info(reco_file, particle_file, do_response=False):
    # Reco-level info
    file_reco = uproot.open(reco_file)
    tree_reco = file_reco["Delphes"]
    reco_info = tree_reco.arrays(["Electron.PT", "Muon.PT", "Jet_size", "Jet.PT", "Jet.Eta"])

    reco_pT_lep1_list = []
    reco_pT_lep2_list = []

    # Particle-level info
    file_particle = uproot.open(particle_file)
    tree_particle = file_particle["LHEF"]
    particle_level = tree_particle["Particle"]
    particle_info = particle_level.arrays(["Particle.PT", "Particle.PID"])

    particle_pT_lep1_list = []
    particle_pT_lep2_list = []

    # Binning
    # fmt: off
    pT_lep1_binning = [28.00, 34.00, 39.00, 45.00, 52.00, 61.00, 71.00, 83.00, 97.00, 115.00, 134.00, 158.00, 188.00, 223.00, 268.00, 338.00, 400.00]
    pT_lep2_binning = [28.00, 32.00, 37.00, 44.00, 51.00, 61.00, 73.00, 88.00, 105.00, 123.00, 150.00, 182.00, 223.00, 268.00, 338.00, 400.00]
    
    # Response matrices
    pT_lep1_response = None
    pT_lep2_response = None
    if do_response:
        binning = pT_lep1_binning
        bins = len(binning) - 1
        name = "particle_pT_lep1_vs_pT_lep1"
        pT_lep1_response = ROOT.TH2D(
            name, name, bins, array("d", binning), bins, array("d", binning)
        )
        pT_lep1_response = ROOT.RooUnfoldResponse(
            pT_lep1_response.ProjectionX(),
            pT_lep1_response.ProjectionY(),
            pT_lep1_response,
            name,
            name,
        )
        name = "particle_pT_lep2_vs_pT_lep2"
        pT_lep2_response = ROOT.TH2D(
            name, name, bins, array("d", binning), bins, array("d", binning)
        )
        pT_lep2_response = ROOT.RooUnfoldResponse(
            pT_lep2_response.ProjectionX(),
            pT_lep2_response.ProjectionY(),
            pT_lep2_response,
            name,
            name,
        )

    # Iterate over events
    print("- Reco file: {}".format(reco_file))
    print("- Particle file: {}".format(particle_file))
    for reco_event, particle_event in tqdm(
        zip(reco_info, particle_info), total=len(reco_info), ncols=100
    ):
        # Particle selection
        particle_pT_lep1, particle_pT_lep2 = get_lep_pT_at_particle_level(particle_event)
        particle_pT_lep1_list.append(particle_pT_lep1)
        particle_pT_lep2_list.append(particle_pT_lep2)
        
        # Reco selection
        fill_flag = 0
        if len(reco_event["Electron.PT"]) == 2 and len(reco_event["Muon.PT"]) == 0:
            sorted_leptons_pT = ak.argsort(reco_event["Electron.PT"], axis=-1, ascending=False)
            reco_pT_lep1_list.append(reco_event["Electron.PT"][sorted_leptons_pT[0]])
            reco_pT_lep2_list.append(reco_event["Electron.PT"][sorted_leptons_pT[1]])
            if do_response:
                fill_flag = True
        elif len(reco_event["Muon.PT"]) == 2 and len(reco_event["Electron.PT"]) == 0:
            sorted_leptons_pT = ak.argsort(reco_event["Muon.PT"], axis=-1, ascending=False)
            reco_pT_lep1_list.append(reco_event["Muon.PT"][sorted_leptons_pT[0]])
            reco_pT_lep2_list.append(reco_event["Muon.PT"][sorted_leptons_pT[1]])
            if do_response:
                fill_flag = True
        elif len(reco_event["Electron.PT"]) == 1 and len(reco_event["Muon.PT"]) == 1:
            if reco_event["Electron.PT"][0] > reco_event["Muon.PT"][0]:
                reco_pT_lep1_list.append(reco_event["Electron.PT"][0])
                reco_pT_lep2_list.append(reco_event["Muon.PT"][0])
            elif reco_event["Electron.PT"][0] < reco_event["Muon.PT"][0]:
                reco_pT_lep1_list.append(reco_event["Muon.PT"][0])
                reco_pT_lep2_list.append(reco_event["Electron.PT"][0])
            if do_response:
                fill_flag = True
        else:
            if do_response:
                fill_flag = False
        
        # Make response matrices
        if do_response:
            if fill_flag:
                pT_lep1_response.Fill(reco_pT_lep1_list[-1], particle_pT_lep1_list[-1])
                pT_lep2_response.Fill(reco_pT_lep2_list[-1], particle_pT_lep2_list[-1])
            else:
                pT_lep1_response.Miss(particle_pT_lep1_list[-1])
                pT_lep2_response.Miss(particle_pT_lep2_list[-1])

    # Save information into particle tree
    reco_tree = {
        "pT_lep1": [ak.from_iter(reco_pT_lep1_list), pT_lep1_binning],
        "pT_lep2": [ak.from_iter(reco_pT_lep2_list), pT_lep2_binning],
    }

    particle_tree = {
        "pT_lep1": [ak.from_iter(particle_pT_lep1_list), pT_lep1_binning],
        "pT_lep2": [ak.from_iter(particle_pT_lep2_list), pT_lep2_binning],
    }
    
    if do_response:
        return [
            pT_lep1_response,
            pT_lep2_response
        ]
        
    return reco_tree, particle_tree


def main():
    # Settings
    print("\nSettings:")
    print("- Saving normalized response matrices: {}".format(args.normalize_response))

    # Create trees
    print("\nCreating particle- and reco-level trees:")
    reco_level, particle_level = get_trees_info(args.reco, args.particle, do_response=False)

    # Save output trees
    output = ROOT.TFile(args.output, "RECREATE")
    reco_dir = output.mkdir("reco")
    particle_dir = output.mkdir("particle")
    for h_reco, h_particle in zip(reco_level.keys(), particle_level.keys()):
        binning = reco_level[h_reco][1]
        bins = len(binning) - 1
        reco_histo = ROOT.TH1D(h_reco, h_reco, bins, array("d", binning))
        particle_name = "particle_{}".format(h_particle)
        particle_histo = ROOT.TH1D(particle_name, particle_name, bins, array("d", binning))
        x = ak.to_numpy(reco_level[h_reco][0])
        y = ak.to_numpy(particle_level[h_particle][0])
        for i in range(len(x)):
            reco_histo.Fill(x[i])
        for i in range(len(y)):
            particle_histo.Fill(y[i])
        reco_dir.cd()
        reco_histo.Write()
        particle_dir.cd()
        particle_histo.Write()

    # Create response matrices data
    print("\nCreating response matrices:")
    response_matrices = get_trees_info(args.reco, args.particle, do_response=True)
    for response in response_matrices:
        reco_dir.cd()
        if args.normalize_response == "no":
            response.Hresponse().Write()
            response.Write()
        elif args.normalize_response == "yes":
            matrix = response.Mresponse(norm=True)
            response = response.Hresponse()
            binning = [
                response.GetXaxis().GetBinLowEdge(bin) for bin in range(1, response.GetNbinsX() + 2)
            ]
            bins = len(binning) - 1
            response_histo = ROOT.TH2D(
                response.GetName(),
                response.GetName(),
                bins,
                array("d", binning),
                bins,
                array("d", binning),
            )
            for i in range(matrix.GetNrows()):
                for j in range(matrix.GetNcols()):
                    response_histo.SetBinContent(i + 1, j + 1, matrix[i][j])
            response_histo.Write()
    output.Close()


if __name__ == "__main__":
    # Parser settings
    parser = ap.ArgumentParser(description="Parsing input arguments.")
    parser.add_argument(
        "-p",
        "--particle",
        default="",
        help="Particle-level file.",
    )
    parser.add_argument(
        "-r",
        "--reco",
        default="",
        help="Reco-level file.",
    )
    parser.add_argument(
        "-pr",
        "--particle-response",
        default="",
        help="Particle-level file for response matrix.",
    )
    parser.add_argument(
        "-rr",
        "--reco-response",
        default="",
        help="Reco-level file for response matrix.",
    )
    parser.add_argument(
        "-o",
        "--output",
        default="",
        help="Output root file name.",
    )
    parser.add_argument(
        "-n",
        "--normalize-response",
        default="no",
        help="Choose if saving normalized response or not.",
    )
    args = parser.parse_args()

    # Main part
    main()
