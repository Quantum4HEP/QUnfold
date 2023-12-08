# ---------------------- Metadata ----------------------
#
# File name:  processing.py
# Author:     Gianluca Bianco (biancogianluca9@gmail.com)
# Date:       2023-11-16
# Copyright:  (c) 2023 Gianluca Bianco under the MIT license.

# Simulated process: gg -> tt~ -> WbWb -> l l~ nu nu~ b b~

# TODO: barre di errore
# TODO: interesting variables: energy, variabili sui jets o bjets

# STD libraries
import argparse as ap
from tqdm import tqdm
from array import array
import sys

# Data science modules
import ROOT
import uproot
import awkward as ak
import numpy as np

# My modules
from paper_functions.physics import (
    compute_invariant_mass,
    compute_energy,
    compute_rapidity,
    compute_DR,
)

# ROOT settings
ROOT.gROOT.SetBatch(True)

# RooUnfold settings
loaded_RooUnfold = ROOT.gSystem.Load("HEP_deps/RooUnfold/libRooUnfold.so")
if not loaded_RooUnfold == 0:
    print("RooUnfold not found!")
    sys.exit(0)

# Constants (GeV)
m_electron = 0.0005109989461
m_muon = 0.1056583715


def get_info_at_particle_level(event, var):
    lepton_mask = (abs(event["Particle.PID"]) == 11) | (abs(event["Particle.PID"]) == 13)
    sorted_leptons = ak.argsort(event["Particle.PT"][lepton_mask], axis=-1, ascending=False)
    leading_lepton = event[var][lepton_mask][sorted_leptons[0]]
    subleading_lepton = event[var][lepton_mask][sorted_leptons[1]]
    return leading_lepton, subleading_lepton


def get_m_l1l2_at_particle_level(event):
    lepton_mask = (abs(event["Particle.PID"]) == 11) | (abs(event["Particle.PID"]) == 13)
    sorted_leptons = ak.argsort(event["Particle.PT"][lepton_mask], axis=-1, ascending=False)
    inv_mass = compute_invariant_mass(
        event["Particle.PT"][lepton_mask][sorted_leptons[0]],
        event["Particle.Eta"][lepton_mask][sorted_leptons[0]],
        event["Particle.Phi"][lepton_mask][sorted_leptons[0]],
        event["Particle.E"][lepton_mask][sorted_leptons[0]],
        event["Particle.PT"][lepton_mask][sorted_leptons[1]],
        event["Particle.Eta"][lepton_mask][sorted_leptons[1]],
        event["Particle.Phi"][lepton_mask][sorted_leptons[1]],
        event["Particle.E"][lepton_mask][sorted_leptons[1]],
    )
    return inv_mass


def get_DR_b1b2_at_particle_level(event):
    bjet_mask = abs(event["Particle.PID"]) == 5
    sorted_bjets = ak.argsort(event["Particle.PT"][bjet_mask], axis=-1, ascending=False)
    DR = compute_DR(
        event["Particle.Eta"][bjet_mask][sorted_bjets[0]],
        event["Particle.Phi"][bjet_mask][sorted_bjets[0]],
        event["Particle.Eta"][bjet_mask][sorted_bjets[1]],
        event["Particle.Phi"][bjet_mask][sorted_bjets[1]],
    )
    return DR


def create_response_matrix(binning, name):
    bins = len(binning) - 1
    response_TH2D = ROOT.TH2D(name, name, bins, array("d", binning), bins, array("d", binning))
    response = ROOT.RooUnfoldResponse(
        response_TH2D.ProjectionX(),
        response_TH2D.ProjectionY(),
        response_TH2D,
        name,
        name,
    )

    return response


def get_trees_info(reco_file, particle_file, do_response=False):
    # Reco-level info
    file_reco = uproot.open(reco_file)
    tree_reco = file_reco["Delphes"]
    reco_info = tree_reco.arrays(
        [
            "Electron.PT",
            "Electron.Eta",
            "Electron.Phi",
            "Muon.PT",
            "Muon.Eta",
            "Muon.Phi",
            "Jet_size",
            "Jet.Flavor",
            "Jet.Eta",
            "Jet.PT",
            "Jet.Phi",
        ]
    )

    reco_pT_lep1_list = []
    reco_pT_lep2_list = []
    reco_eta_lep1_list = []
    reco_eta_lep2_list = []
    reco_m_l1l2_list = []
    reco_phi_lep1_list = []
    reco_phi_lep2_list = []
    reco_y_lep1_list = []
    reco_y_lep2_list = []
    reco_DR_b1b2_list = []

    # Particle-level info
    file_particle = uproot.open(particle_file)
    tree_particle = file_particle["LHEF"]
    particle_level = tree_particle["Particle"]
    particle_info = particle_level.arrays(
        [
            "Particle.PT",
            "Particle.PID",
            "Particle.Eta",
            "Particle.Phi",
            "Particle.E",
            "Particle.Rapidity",
        ]
    )

    particle_pT_lep1_list = []
    particle_pT_lep2_list = []
    particle_eta_lep1_list = []
    particle_eta_lep2_list = []
    particle_m_l1l2_list = []
    particle_phi_lep1_list = []
    particle_phi_lep2_list = []
    particle_y_lep1_list = []
    particle_y_lep2_list = []
    particle_DR_b1b2_list = []

    # Binning
    # fmt: off
    pT_lep1_binning = np.linspace(0, 400, 30)
    pT_lep2_binning = np.linspace(0, 400, 30)
    eta_lep1_binning = np.linspace(-2.5, 2.5, 20)
    eta_lep2_binning = np.linspace(-2.5, 2.5, 20)
    m_l1l2_binning = np.linspace(0, 800, 30)
    phi_lep1_binning = np.linspace(-3.0, 3.0, 30)
    phi_lep2_binning = np.linspace(-3.0, 3.0, 30)
    y_lep1_binning = np.linspace(0, 5, 30)
    y_lep2_binning = np.linspace(0, 5, 30)
    DR_b1b2_binning = np.linspace(0, 6, 30)
    
    # Response matrices
    pT_lep1_response = None
    pT_lep2_response = None
    eta_lep1_response = None
    eta_lep2_response = None
    m_l1l2_response = None
    phi_lep1_response = None
    phi_lep2_response = None
    y_lep1_response = None
    y_lep2_response = None
    DR_b1b2_response = None
    if do_response:
        pT_lep1_response = create_response_matrix(pT_lep1_binning, "particle_pT_lep1_vs_pT_lep1")
        pT_lep2_response = create_response_matrix(pT_lep2_binning, "particle_pT_lep2_vs_pT_lep2")
        eta_lep1_response = create_response_matrix(eta_lep1_binning, "particle_eta_lep1_vs_eta_lep1")
        eta_lep2_response = create_response_matrix(eta_lep2_binning, "particle_eta_lep2_vs_eta_lep2")
        m_l1l2_response = create_response_matrix(m_l1l2_binning, "particle_m_l1l2_vs_m_l1l2")
        phi_lep1_response = create_response_matrix(phi_lep1_binning, "particle_phi_lep1_vs_phi_lep1")
        phi_lep2_response = create_response_matrix(phi_lep2_binning, "particle_phi_lep2_vs_phi_lep2")
        y_lep1_response = create_response_matrix(y_lep1_binning, "particle_y_lep1_vs_y_lep1")
        y_lep2_response = create_response_matrix(y_lep2_binning, "particle_y_lep2_vs_y_lep2")
        DR_b1b2_response = create_response_matrix(DR_b1b2_binning, "particle_DR_b1b2_vs_DR_b1b2")

    # Iterate over events
    print("- Reco file: {}".format(reco_file))
    print("- Particle file: {}".format(particle_file))
    for reco_event, particle_event in tqdm(
        zip(reco_info, particle_info), total=len(reco_info), ncols=100
    ):
        
        # Particle selection pT
        particle_pT_lep1, particle_pT_lep2 = get_info_at_particle_level(particle_event, "Particle.PT")
        particle_pT_lep1_list.append(particle_pT_lep1)
        particle_pT_lep2_list.append(particle_pT_lep2)

        # Particle selection eta
        particle_eta_lep1, particle_eta_lep2 = get_info_at_particle_level(particle_event, "Particle.Eta")
        particle_eta_lep1_list.append(particle_eta_lep1)
        particle_eta_lep2_list.append(particle_eta_lep2)
        
        # Particle selection phi
        particle_phi_lep1, particle_phi_lep2 = get_info_at_particle_level(particle_event, "Particle.Phi")
        particle_phi_lep1_list.append(particle_phi_lep1)
        particle_phi_lep2_list.append(particle_phi_lep2)
        
        # Particle selection y
        particle_y_lep1, particle_y_lep2 = get_info_at_particle_level(particle_event, "Particle.Rapidity")
        particle_y_lep1_list.append(particle_y_lep1)
        particle_y_lep2_list.append(particle_y_lep2)
        
        # Particle selection m_l1l2
        particle_m_l1l2 = get_m_l1l2_at_particle_level(particle_event)
        particle_m_l1l2_list.append(particle_m_l1l2)
        
        # Particle selection DR_b1b2
        particle_DR_b1b2 = get_DR_b1b2_at_particle_level(particle_event)
        particle_DR_b1b2_list.append(particle_DR_b1b2)
        
        # Reco selection
        fill_flag = 0
        bjet_mask = abs(reco_event["Jet.Flavor"]) == 5
        b_jets_leq_2 = len(reco_event["Jet.Flavor"][bjet_mask]) >= 2
        
        if len(reco_event["Electron.PT"]) == 2 and len(reco_event["Muon.PT"]) == 0 and b_jets_leq_2:
            sorted_leptons_pT = ak.argsort(reco_event["Electron.PT"], axis=-1, ascending=False)
            sorted_jets_pT = ak.argsort(reco_event["Jet.PT"], axis=-1, ascending=False)
            
            # pT
            reco_pT_lep1_list.append(reco_event["Electron.PT"][sorted_leptons_pT[0]])
            reco_pT_lep2_list.append(reco_event["Electron.PT"][sorted_leptons_pT[1]])

            # Eta
            reco_eta_lep1_list.append(reco_event["Electron.Eta"][sorted_leptons_pT[0]])
            reco_eta_lep2_list.append(reco_event["Electron.Eta"][sorted_leptons_pT[1]])

            # Phi
            reco_phi_lep1_list.append(reco_event["Electron.Phi"][sorted_leptons_pT[0]])
            reco_phi_lep2_list.append(reco_event["Electron.Phi"][sorted_leptons_pT[1]])
            
            # y
            energy_lep1 = compute_energy(m_electron, reco_event["Electron.PT"][sorted_leptons_pT[0]], reco_event["Electron.Phi"][sorted_leptons_pT[0]], reco_event["Electron.Eta"][sorted_leptons_pT[0]])
            energy_lep2 = compute_energy(m_electron, reco_event["Electron.PT"][sorted_leptons_pT[1]], reco_event["Electron.Phi"][sorted_leptons_pT[1]], reco_event["Electron.Eta"][sorted_leptons_pT[1]])
            p_z_lep1 = reco_event["Electron.PT"][sorted_leptons_pT[0]] * np.sinh(reco_event["Electron.Eta"][sorted_leptons_pT[0]])
            p_z_lep2 = reco_event["Electron.PT"][sorted_leptons_pT[1]] * np.sinh(reco_event["Electron.Eta"][sorted_leptons_pT[1]])
            
            reco_y_lep1_list.append(compute_rapidity(p_z_lep1, energy_lep1))
            reco_y_lep2_list.append(compute_rapidity(p_z_lep2, energy_lep2))
            
            # m_l1l2
            reco_m_l1l2 = compute_invariant_mass(
                reco_event["Electron.PT"][sorted_leptons_pT[0]], 
                reco_event["Electron.Eta"][sorted_leptons_pT[0]], 
                reco_event["Electron.Phi"][sorted_leptons_pT[0]], 
                energy_lep1, 
                reco_event["Electron.PT"][sorted_leptons_pT[1]], 
                reco_event["Electron.Eta"][sorted_leptons_pT[1]], 
                reco_event["Electron.Phi"][sorted_leptons_pT[1]], 
                energy_lep2, 
            )
            reco_m_l1l2_list.append(reco_m_l1l2)
            
            # DR_b1b2
            reco_DR_b1b2 = compute_DR(
                reco_event["Jet.Eta"][sorted_jets_pT[0]], 
                reco_event["Jet.Phi"][sorted_jets_pT[0]],
                reco_event["Jet.Eta"][sorted_jets_pT[1]], 
                reco_event["Jet.Phi"][sorted_jets_pT[1]],
            )
            reco_DR_b1b2_list.append(reco_DR_b1b2)
            
            if do_response:
                fill_flag = True
        elif len(reco_event["Muon.PT"]) == 2 and len(reco_event["Electron.PT"]) == 0 and b_jets_leq_2:
            sorted_leptons_pT = ak.argsort(reco_event["Muon.PT"], axis=-1, ascending=False)
            sorted_jets_pT = ak.argsort(reco_event["Jet.PT"], axis=-1, ascending=False)
            
            # pT
            reco_pT_lep1_list.append(reco_event["Muon.PT"][sorted_leptons_pT[0]])
            reco_pT_lep2_list.append(reco_event["Muon.PT"][sorted_leptons_pT[1]])
            
            # Eta
            reco_eta_lep1_list.append(reco_event["Muon.Eta"][sorted_leptons_pT[0]])
            reco_eta_lep2_list.append(reco_event["Muon.Eta"][sorted_leptons_pT[1]])
            
            # Phi
            reco_phi_lep1_list.append(reco_event["Muon.Phi"][sorted_leptons_pT[0]])
            reco_phi_lep2_list.append(reco_event["Muon.Phi"][sorted_leptons_pT[1]])
            
            # y
            energy_lep1 = compute_energy(m_muon, reco_event["Muon.PT"][sorted_leptons_pT[0]], reco_event["Muon.Phi"][sorted_leptons_pT[0]], reco_event["Muon.Eta"][sorted_leptons_pT[0]])
            energy_lep2 = compute_energy(m_muon, reco_event["Muon.PT"][sorted_leptons_pT[1]], reco_event["Muon.Phi"][sorted_leptons_pT[1]], reco_event["Muon.Eta"][sorted_leptons_pT[1]])
            p_z_lep1 = reco_event["Muon.PT"][sorted_leptons_pT[0]] * np.sinh(reco_event["Muon.Eta"][sorted_leptons_pT[0]])
            p_z_lep2 = reco_event["Muon.PT"][sorted_leptons_pT[1]] * np.sinh(reco_event["Muon.Eta"][sorted_leptons_pT[1]])
            
            reco_y_lep1_list.append(compute_rapidity(p_z_lep1, energy_lep1))
            reco_y_lep2_list.append(compute_rapidity(p_z_lep2, energy_lep2))
            
            # m_l1l2
            reco_m_l1l2 = compute_invariant_mass(
                reco_event["Muon.PT"][sorted_leptons_pT[0]], 
                reco_event["Muon.Eta"][sorted_leptons_pT[0]], 
                reco_event["Muon.Phi"][sorted_leptons_pT[0]], 
                energy_lep1, 
                reco_event["Muon.PT"][sorted_leptons_pT[1]], 
                reco_event["Muon.Eta"][sorted_leptons_pT[1]], 
                reco_event["Muon.Phi"][sorted_leptons_pT[1]], 
                energy_lep2, 
            )
            reco_m_l1l2_list.append(reco_m_l1l2)
            
            # DR_b1b2
            reco_DR_b1b2 = compute_DR(
                reco_event["Jet.Eta"][sorted_jets_pT[0]], 
                reco_event["Jet.Phi"][sorted_jets_pT[0]],
                reco_event["Jet.Eta"][sorted_jets_pT[1]], 
                reco_event["Jet.Phi"][sorted_jets_pT[1]],
            )
            reco_DR_b1b2_list.append(reco_DR_b1b2)
        
            if do_response:
                fill_flag = True
        elif len(reco_event["Electron.PT"]) == 1 and len(reco_event["Muon.PT"]) == 1 and b_jets_leq_2:
            sorted_jets_pT = ak.argsort(reco_event["Jet.PT"], axis=-1, ascending=False)
            
            if reco_event["Electron.PT"][0] > reco_event["Muon.PT"][0]:
                
                # pT
                reco_pT_lep1_list.append(reco_event["Electron.PT"][0])
                reco_pT_lep2_list.append(reco_event["Muon.PT"][0])
                
                # Eta
                reco_eta_lep1_list.append(reco_event["Electron.Eta"][0])
                reco_eta_lep2_list.append(reco_event["Muon.Eta"][0])

                # Phi
                reco_phi_lep1_list.append(reco_event["Electron.Phi"][0])
                reco_phi_lep2_list.append(reco_event["Muon.Phi"][0])
                
                # y
                energy_lep1 = compute_energy(m_electron, reco_event["Electron.PT"][0], reco_event["Electron.Phi"][0], reco_event["Electron.Eta"][0])
                energy_lep2 = compute_energy(m_muon, reco_event["Muon.PT"][0], reco_event["Muon.Phi"][0], reco_event["Muon.Eta"][0])
                p_z_lep1 = reco_event["Electron.PT"][0] * np.sinh(reco_event["Electron.Eta"][0])
                p_z_lep2 = reco_event["Muon.PT"][0] * np.sinh(reco_event["Muon.Eta"][0])
                
                reco_y_lep1_list.append(compute_rapidity(p_z_lep1, energy_lep1))
                reco_y_lep2_list.append(compute_rapidity(p_z_lep2, energy_lep2))

                # m_l1l2
                reco_m_l1l2 = compute_invariant_mass(
                    reco_event["Electron.PT"][0], 
                    reco_event["Electron.Eta"][0], 
                    reco_event["Electron.Phi"][0], 
                    energy_lep1, 
                    reco_event["Muon.PT"][0], 
                    reco_event["Muon.Eta"][0], 
                    reco_event["Muon.Phi"][0], 
                    energy_lep2, 
                )
                reco_m_l1l2_list.append(reco_m_l1l2)

            elif reco_event["Electron.PT"][0] < reco_event["Muon.PT"][0]:
                
                # pT
                reco_pT_lep1_list.append(reco_event["Muon.PT"][0])
                reco_pT_lep2_list.append(reco_event["Electron.PT"][0])
                
                # Eta
                reco_eta_lep1_list.append(reco_event["Electron.Eta"][0])
                reco_eta_lep2_list.append(reco_event["Muon.Eta"][0])
                
                # Phi
                reco_phi_lep1_list.append(reco_event["Electron.Phi"][0])
                reco_phi_lep2_list.append(reco_event["Muon.Phi"][0])
                
                # y
                energy_lep1 = compute_energy(m_muon, reco_event["Muon.PT"][0], reco_event["Muon.Phi"][0], reco_event["Muon.Eta"][0])
                energy_lep2 = compute_energy(m_electron, reco_event["Electron.PT"][0], reco_event["Electron.Phi"][0], reco_event["Electron.Eta"][0])
                p_z_lep1 = reco_event["Muon.PT"][0] * np.sinh(reco_event["Muon.Eta"][0])
                p_z_lep2 = reco_event["Electron.PT"][0] * np.sinh(reco_event["Electron.Eta"][0])
                
                reco_y_lep1_list.append(compute_rapidity(p_z_lep1, energy_lep1))
                reco_y_lep2_list.append(compute_rapidity(p_z_lep2, energy_lep2))

                # m_l1l2
                reco_m_l1l2 = compute_invariant_mass(
                    reco_event["Muon.PT"][0], 
                    reco_event["Muon.Eta"][0], 
                    reco_event["Muon.Phi"][0], 
                    energy_lep1, 
                    reco_event["Electron.PT"][0], 
                    reco_event["Electron.Eta"][0], 
                    reco_event["Electron.Phi"][0], 
                    energy_lep2, 
                )
                reco_m_l1l2_list.append(reco_m_l1l2)
                
            # DR_b1b2
            reco_DR_b1b2 = compute_DR(
                reco_event["Jet.Eta"][sorted_jets_pT[0]], 
                reco_event["Jet.Phi"][sorted_jets_pT[0]],
                reco_event["Jet.Eta"][sorted_jets_pT[1]], 
                reco_event["Jet.Phi"][sorted_jets_pT[1]],
            )
            reco_DR_b1b2_list.append(reco_DR_b1b2)
            
            if do_response:
                fill_flag = True
        else:
            if do_response:
                fill_flag = False
        
        # Make response matrices
        if do_response:
            if fill_flag:
                # pT
                pT_lep1_response.Fill(reco_pT_lep1_list[-1], particle_pT_lep1_list[-1])
                pT_lep2_response.Fill(reco_pT_lep2_list[-1], particle_pT_lep2_list[-1])
                
                # Eta
                eta_lep1_response.Fill(reco_eta_lep1_list[-1], particle_eta_lep1_list[-1])
                eta_lep2_response.Fill(reco_eta_lep2_list[-1], particle_eta_lep2_list[-1])
                
                # Phi
                phi_lep1_response.Fill(reco_phi_lep1_list[-1], particle_phi_lep1_list[-1])
                phi_lep2_response.Fill(reco_phi_lep2_list[-1], particle_phi_lep2_list[-1])
                
                # y
                y_lep1_response.Fill(reco_y_lep1_list[-1], particle_y_lep1_list[-1])
                y_lep2_response.Fill(reco_y_lep2_list[-1], particle_y_lep2_list[-1])
                
                # m_l1l2
                m_l1l2_response.Fill(reco_m_l1l2_list[-1], reco_m_l1l2_list[-1])
                
                # DR_b1b2
                DR_b1b2_response.Fill(reco_DR_b1b2_list[-1], reco_DR_b1b2_list[-1])
            else:
                # pT
                pT_lep1_response.Miss(particle_pT_lep1_list[-1])
                pT_lep2_response.Miss(particle_pT_lep2_list[-1])
                
                # Eta
                eta_lep1_response.Miss(particle_eta_lep1_list[-1])
                eta_lep2_response.Miss(particle_eta_lep2_list[-1])
                
                # Phi
                phi_lep1_response.Miss(particle_phi_lep1_list[-1])
                phi_lep2_response.Miss(particle_phi_lep2_list[-1])
            
                # y
                y_lep1_response.Miss(particle_y_lep1_list[-1])
                y_lep2_response.Miss(particle_y_lep2_list[-1])

                # m_l1l2
                m_l1l2_response.Miss(particle_m_l1l2_list[-1])
                
                # m_l1l2
                DR_b1b2_response.Miss(particle_DR_b1b2_list[-1])

    # Save information into particle tree
    reco_tree = {
        "pT_lep1": [ak.from_iter(reco_pT_lep1_list), pT_lep1_binning],
        "pT_lep2": [ak.from_iter(reco_pT_lep2_list), pT_lep2_binning],
        "eta_lep1": [ak.from_iter(reco_eta_lep1_list), eta_lep1_binning],
        "eta_lep2": [ak.from_iter(reco_eta_lep2_list), eta_lep2_binning],
        "m_l1l2": [ak.from_iter(reco_m_l1l2_list), m_l1l2_binning],
        "phi_lep1": [ak.from_iter(reco_phi_lep1_list), phi_lep1_binning],
        "phi_lep2": [ak.from_iter(reco_phi_lep2_list), phi_lep2_binning],
        "y_lep1": [ak.from_iter(reco_y_lep1_list), y_lep1_binning],
        "y_lep2": [ak.from_iter(reco_y_lep2_list), y_lep2_binning],
        "DR_b1b2": [ak.from_iter(reco_DR_b1b2_list), DR_b1b2_binning],
    }

    particle_tree = {
        "particle_pT_lep1": [ak.from_iter(particle_pT_lep1_list), pT_lep1_binning],
        "particle_pT_lep2": [ak.from_iter(particle_pT_lep2_list), pT_lep2_binning],
        "particle_eta_lep1": [ak.from_iter(particle_eta_lep1_list), eta_lep1_binning],
        "particle_eta_lep2": [ak.from_iter(particle_eta_lep2_list), eta_lep2_binning],
        "particle_m_l1l2": [ak.from_iter(particle_m_l1l2_list), m_l1l2_binning],
        "particle_phi_lep1": [ak.from_iter(particle_phi_lep1_list), phi_lep1_binning],
        "particle_phi_lep2": [ak.from_iter(particle_phi_lep2_list), phi_lep2_binning],
        "particle_y_lep1": [ak.from_iter(particle_y_lep1_list), y_lep1_binning],
        "particle_y_lep2": [ak.from_iter(particle_y_lep2_list), y_lep2_binning],
        "particle_DR_b1b2": [ak.from_iter(particle_DR_b1b2_list), DR_b1b2_binning],
    }
    
    if do_response:
        return [
            pT_lep1_response,
            pT_lep2_response,
            eta_lep1_response,
            eta_lep2_response,
            m_l1l2_response,
            phi_lep1_response,
            phi_lep2_response,
            y_lep1_response,
            y_lep2_response,
            DR_b1b2_response,
        ], reco_tree, particle_tree
        
    return reco_tree, particle_tree


def main():
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
        particle_histo = ROOT.TH1D(h_particle, h_particle, bins, array("d", binning))
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
    response_matrices, reco_level, mc_level = get_trees_info(
        args.reco_response, args.particle_response, do_response=True
    )
    for response in response_matrices:
        reco_dir.cd()
        response.Hresponse().Write()
    for h_reco, h_particle in zip(reco_level.keys(), mc_level.keys()):
        binning = reco_level[h_reco][1]
        bins = len(binning) - 1
        reco_histo = ROOT.TH1D(f"mc_{h_reco}", f"mc_{h_reco}", bins, array("d", binning))
        particle_histo = ROOT.TH1D(
            f"mc_{h_particle}", f"mc_{h_particle}", bins, array("d", binning)
        )
        x = ak.to_numpy(reco_level[h_reco][0])
        y = ak.to_numpy(mc_level[h_particle][0])
        for i in range(len(x)):
            reco_histo.Fill(x[i])
        for i in range(len(y)):
            particle_histo.Fill(y[i])
        reco_dir.cd()
        reco_histo.Write()
        particle_dir.cd()
        particle_histo.Write()


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
    args = parser.parse_args()

    # Main part
    main()
