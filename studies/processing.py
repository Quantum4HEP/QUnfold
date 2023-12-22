import ROOT
import uproot
import awkward as ak
import numpy as np
from array import array
from paper_functions.physics import (
    compute_invariant_mass_reco,
    compute_invariant_mass_particle,
    compute_energy,
    compute_DR_reco,
    compute_DR_particle,
)

# Constants (GeV)
m_electron = 0.0005109989461
m_muon = 0.1056583715
m_bjet = 4.18


def get_info_particle_level(particle_info, list_particleIDs, list_partvars):
    particle_mask = ak.Array(np.zeros(len(particle_info), dtype=bool))
    for ids in list_particleIDs:
        particle_mask = particle_mask | (abs(particle_info["Particle.PID"]) == ids)
    outvars = {}
    for partvars in list_partvars:
        sorted_indices = ak.argsort(
            particle_info["Particle.PT"][particle_mask], axis=-1
        )
        temp = particle_info[partvars][particle_mask][sorted_indices]
        outvars[partvars] = [temp[:, 1], temp[:, 0]]
    return outvars


def get_info_reco_level(reco_info, list_recovars):
    outvars = {}
    for recovars in list_recovars:
        outvars[recovars] = reco_info[recovars]
    return outvars


def tree_reader(reco_file, particle_file, reco_var_list, part_val_dict):
    reco_tree = uproot.open(reco_file)["Delphes"]
    particle_tree = uproot.open(particle_file)["LHEF"]
    part_val_list = list(
        set([item for sublist in part_val_dict.values() for item in sublist])
    )
    part_val_list.insert(0, "Particle.PID")
    reco_info = reco_tree.arrays(reco_var_list)
    particle_info = particle_tree.arrays(part_val_list)

    num_events = len(reco_info)
    events_idx = np.array(range(num_events))
    all_part_info = {}
    for pids in list(part_val_dict.keys()):
        all_part_info[pids] = get_info_particle_level(
            particle_info, [int(num) for num in pids.split(",")], part_val_dict[pids]
        )

    reco_info = get_info_reco_level(reco_info, reco_var_list)
    return [events_idx, reco_info, all_part_info]


def hist_resp_builder(
    name,
    reco,
    particle,
    binning,
    do_response=False,
):
    mask = reco != None
    reco = reco[mask].astype(float)

    binning = np.array([-np.inf] + binning.tolist() + [np.inf])
    hist_reco, _ = np.histogram(reco, binning)
    hist_particle, _ = np.histogram(ak.to_numpy(particle), binning)

    if do_response:
        response, _, _ = np.histogram2d(
            ak.to_numpy(reco),
            ak.to_numpy(particle[mask]).astype(float),
            binning,
        )
        response = response.T
        return name, hist_reco, hist_particle, binning, response
    else:
        return name, hist_reco, hist_particle, binning


def reco_filter_bjet(events_idx, reco_info, var):
    # Selection
    e_size = reco_info["Electron_size"]
    mu_size = reco_info["Muon_size"]
    bjet_size = np.sum(np.abs(reco_info["Jet.Flavor"] == 5), axis=1)
    mask_dilep = (
        ((e_size == 2) & (mu_size == 0))
        | ((e_size == 0) & (mu_size == 2))
        | ((e_size == 1) & (mu_size == 1))
    ) & (bjet_size >= 2)
    events_dilep_idx = events_idx[mask_dilep]
    bjet_pt_dilep = reco_info["Jet.PT"][mask_dilep]
    sorted_dilep_idx = np.argsort(bjet_pt_dilep)

    # Variables
    reco_var_dilep = None
    reco_pT_dilep = reco_info["Jet.PT"][mask_dilep][sorted_dilep_idx]
    reco_eta_dilep = reco_info["Jet.Eta"][mask_dilep][sorted_dilep_idx]
    reco_phi_dilep = reco_info["Jet.Phi"][mask_dilep][sorted_dilep_idx]
    bjet_mask = reco_info["Jet.Flavor"][mask_dilep][sorted_dilep_idx] == 5

    if var == "DR_b1b2":
        reco_var_dilep = compute_DR_reco(
            reco_eta_dilep[bjet_mask][:, -2:], reco_phi_dilep[bjet_mask][:, -2:]
        )
    elif var == "m_b1b2":
        reco_e_dilep = compute_energy(
            m_bjet,
            reco_pT_dilep[bjet_mask][:, -2:],
            reco_phi_dilep[bjet_mask][:, -2:],
            reco_eta_dilep[bjet_mask][:, -2:],
        )
        reco_var_dilep = compute_invariant_mass_reco(
            reco_pT_dilep[bjet_mask][:, -2:],
            reco_eta_dilep[bjet_mask][:, -2:],
            reco_phi_dilep[bjet_mask][:, -2:],
            reco_e_dilep,
        )
    # Result
    temp = reco_var_dilep
    concat_var = np.array([None] * len(events_idx))
    concat_idx = events_dilep_idx
    concat_var[mask_dilep] = temp[np.argsort(concat_idx)]

    return concat_var


def reco_filter_m_l1l2(events_idx, reco_info):
    # Variables
    e_size = reco_info["Electron_size"]
    mu_size = reco_info["Muon_size"]
    bjet_size = np.sum(np.abs(reco_info["Jet.Flavor"] == 5), axis=1)

    # ee / 2 b-jets
    mask_ee = (e_size == 2) & (mu_size == 0) & (bjet_size >= 2)
    events_ee_idx = events_idx[mask_ee]
    electron_pt_ee = reco_info["Electron.PT"][mask_ee]
    sorted_ee_idx = np.argsort(electron_pt_ee)

    reco_pT_ee = reco_info["Electron.PT"][mask_ee][sorted_ee_idx]
    reco_eta_ee = reco_info["Electron.Eta"][mask_ee][sorted_ee_idx]
    reco_phi_ee = reco_info["Electron.Phi"][mask_ee][sorted_ee_idx]
    reco_e_ee = compute_energy(
        m_electron,
        reco_pT_ee,
        reco_phi_ee,
        reco_eta_ee,
    )
    reco_var_ee = compute_invariant_mass_reco(
        reco_pT_ee, reco_eta_ee, reco_phi_ee, reco_e_ee
    )

    # mumu / 2 b-jets
    mask_mumu = (e_size == 0) & (mu_size == 2) & (bjet_size >= 2)
    events_mumu_idx = events_idx[mask_mumu]
    muon_pt_mumu = reco_info["Muon.PT"][mask_mumu]
    sorted_mumu_idx = np.argsort(muon_pt_mumu)

    reco_pT_mumu = reco_info["Muon.PT"][mask_mumu][sorted_mumu_idx]
    reco_eta_mumu = reco_info["Muon.Eta"][mask_mumu][sorted_mumu_idx]
    reco_phi_mumu = reco_info["Muon.Phi"][mask_mumu][sorted_mumu_idx]
    reco_e_mumu = compute_energy(
        m_muon,
        reco_pT_mumu,
        reco_phi_mumu,
        reco_eta_mumu,
    )
    reco_var_mumu = compute_invariant_mass_reco(
        reco_pT_mumu, reco_eta_mumu, reco_phi_mumu, reco_e_mumu
    )

    # emu / 2 b-jets
    mask_emu = (e_size == 1) & (mu_size == 1) & (bjet_size >= 2)
    events_emu_idx = events_idx[mask_emu]
    electron_pt_emu = reco_info["Electron.PT"][mask_emu]
    muon_pt_emu = reco_info["Muon.PT"][mask_emu]
    sorted_emu_idx = np.argsort(np.hstack((electron_pt_emu, muon_pt_emu)))

    reco_pT_emu = np.hstack(
        (reco_info["Electron.PT"][mask_emu], reco_info["Muon.PT"][mask_emu])
    )[sorted_emu_idx]
    reco_eta_emu = np.hstack(
        (reco_info["Electron.Eta"][mask_emu], reco_info["Muon.Eta"][mask_emu])
    )[sorted_emu_idx]
    reco_phi_emu = np.hstack(
        (reco_info["Electron.Phi"][mask_emu], reco_info["Muon.Phi"][mask_emu])
    )[sorted_emu_idx]
    reco_e_e = compute_energy(
        m_electron,
        reco_info["Electron.PT"][mask_emu],
        reco_info["Electron.Phi"][mask_emu],
        reco_info["Electron.Eta"][mask_emu],
    )
    reco_e_mu = compute_energy(
        m_muon,
        reco_info["Muon.PT"][mask_emu],
        reco_info["Muon.Phi"][mask_emu],
        reco_info["Muon.Eta"][mask_emu],
    )
    reco_e_emu = np.hstack((reco_e_e, reco_e_mu))[sorted_emu_idx]
    reco_var_emu = compute_invariant_mass_reco(
        reco_pT_emu, reco_eta_emu, reco_phi_emu, reco_e_emu
    )

    temp = np.concatenate((reco_var_ee, reco_var_mumu, reco_var_emu))
    concat_var = np.array([None] * len(events_idx))
    concat_idx = np.concatenate((events_ee_idx, events_mumu_idx, events_emu_idx))
    concat_var[mask_ee | mask_mumu | mask_emu] = temp[np.argsort(concat_idx)]

    return concat_var


def reco_filter_lep(events_idx, reco_info, var):
    # Variables
    reco_var_e = reco_info[f"Electron.{var}"]
    reco_var_mu = reco_info[f"Muon.{var}"]
    e_size = reco_info["Electron_size"]
    mu_size = reco_info["Muon_size"]
    bjet_size = np.sum(np.abs(reco_info["Jet.Flavor"] == 5), axis=1)

    # ee / 2 b-jets
    mask_ee = (e_size == 2) & (mu_size == 0) & (bjet_size >= 2)
    events_ee_idx = events_idx[mask_ee]
    electron_pt_ee = reco_info["Electron.PT"][mask_ee]
    sorted_ee_idx = np.argsort(electron_pt_ee)
    reco_var_ee = reco_var_e[mask_ee][sorted_ee_idx]

    # mumu / 2 b-jets
    mask_mumu = (e_size == 0) & (mu_size == 2) & (bjet_size >= 2)
    events_mumu_idx = events_idx[mask_mumu]
    muon_pt_mumu = reco_info["Muon.PT"][mask_mumu]
    sorted_mumu_idx = np.argsort(muon_pt_mumu)
    reco_var_mumu = reco_var_mu[mask_mumu][sorted_mumu_idx]

    # emu / 2 b-jets
    mask_emu = (e_size == 1) & (mu_size == 1) & (bjet_size >= 2)
    events_emu_idx = events_idx[mask_emu]
    electron_pt_emu = reco_info["Electron.PT"][mask_emu]
    muon_pt_emu = reco_info["Muon.PT"][mask_emu]
    sorted_emu_idx = np.argsort(np.hstack((electron_pt_emu, muon_pt_emu)))
    reco_var_emu = np.hstack((reco_var_e[mask_emu], reco_var_mu[mask_emu]))[
        sorted_emu_idx
    ]

    temp = np.concatenate((reco_var_ee, reco_var_mumu, reco_var_emu))
    concat_var = np.array([[None, None]] * len(events_idx))
    concat_idx = np.concatenate((events_ee_idx, events_mumu_idx, events_emu_idx))
    concat_var[mask_ee | mask_mumu | mask_emu] = temp[np.argsort(concat_idx)]

    return concat_var[:, 1], concat_var[:, 0]


def process(
    reco_tree_path, part_tree_path, list_recovars, dict_partvars, do_response=False
):
    events_idx, reco_info, particle_info = tree_reader(
        reco_tree_path, part_tree_path, list_recovars, dict_partvars
    )

    # pT_lep1/2
    reco_pT_lep1, reco_pT_lep2 = reco_filter_lep(events_idx, reco_info, "PT")
    particle_pT_lep1, particle_pT_lep2 = particle_info["11,13"]["Particle.PT"]

    # eta_lep1/2
    reco_Eta_lep1, reco_Eta_lep2 = reco_filter_lep(events_idx, reco_info, "Eta")
    particle_Eta_lep1, particle_Eta_lep2 = particle_info["11,13"]["Particle.Eta"]

    # phi_lep1/2
    reco_Phi_lep1, reco_Phi_lep2 = reco_filter_lep(events_idx, reco_info, "Phi")
    particle_Phi_lep1, particle_Phi_lep2 = particle_info["11,13"]["Particle.Phi"]

    # DR_b1b2
    reco_DR_b1b2 = reco_filter_bjet(events_idx, reco_info, "DR_b1b2")
    particle_eta_bjet1, particle_eta_bjet2 = particle_info["5"]["Particle.Eta"]
    particle_phi_bjet1, particle_phi_bjet2 = particle_info["5"]["Particle.Phi"]
    particle_DR_b1b2 = compute_DR_particle(
        particle_eta_bjet1, particle_eta_bjet2, particle_phi_bjet1, particle_phi_bjet2
    )

    # m_l1l2
    reco_m_l1l2 = reco_filter_m_l1l2(events_idx, reco_info)
    particle_e_lep1, particle_e_lep2 = particle_info["11,13"]["Particle.E"]
    particle_m_l1l2 = compute_invariant_mass_particle(
        particle_pT_lep1,
        particle_Eta_lep1,
        particle_Phi_lep1,
        particle_e_lep1,
        particle_pT_lep2,
        particle_Eta_lep2,
        particle_Phi_lep2,
        particle_e_lep2,
    )

    # m_b1b2
    reco_m_b1b2 = reco_filter_bjet(events_idx, reco_info, "m_b1b2")
    particle_e_bjet1, particle_e_bjet2 = particle_info["5"]["Particle.E"]
    particle_pT_bjet1, particle_pT_bjet2 = particle_info["5"]["Particle.PT"]
    particle_Eta_bjet1, particle_Eta_bjet2 = particle_info["5"]["Particle.Eta"]
    particle_Phi_bjet1, particle_Phi_bjet2 = particle_info["5"]["Particle.Phi"]
    particle_m_b1b2 = compute_invariant_mass_particle(
        particle_pT_bjet1,
        particle_Eta_bjet1,
        particle_Phi_bjet1,
        particle_e_bjet1,
        particle_pT_bjet2,
        particle_Eta_bjet2,
        particle_Phi_bjet2,
        particle_e_bjet2,
    )

    # Binning
    binning_leading_pT = np.linspace(20, 180, 30)
    binning_subleading_pT = np.linspace(20, 140, 30)
    binning_leading_eta = np.linspace(-2.2, 2.2, 20)
    binning_subleading_eta = np.linspace(-2.2, 2.2, 20)
    binning_leading_phi = np.linspace(-3.0, 3.0, 20)
    binning_subleading_phi = np.linspace(-3.0, 3.0, 20)
    binning_m_l1l2 = np.linspace(0, 400, 30)
    binning_m_b1b2 = np.linspace(20, 500, 30)
    binning_DR_b1b2 = np.linspace(1, 5, 20)

    # Result
    variables = [
        ("pT_lep1", reco_pT_lep1, particle_pT_lep1, binning_leading_pT),
        ("pT_lep2", reco_pT_lep2, particle_pT_lep2, binning_subleading_pT),
        ("eta_lep1", reco_Eta_lep1, particle_Eta_lep1, binning_leading_eta),
        ("eta_lep2", reco_Eta_lep2, particle_Eta_lep2, binning_subleading_eta),
        ("phi_lep1", reco_Phi_lep1, particle_Phi_lep1, binning_leading_phi),
        ("phi_lep2", reco_Phi_lep2, particle_Phi_lep2, binning_subleading_phi),
        ("DR_b1b2", reco_DR_b1b2, particle_DR_b1b2, binning_DR_b1b2),
        ("m_l1l2", reco_m_l1l2, particle_m_l1l2, binning_m_l1l2),
        ("m_b1b2", reco_m_b1b2, particle_m_b1b2, binning_m_b1b2),
    ]

    processed = []
    for var in variables:
        processed.append(hist_resp_builder(*var, do_response))

    return processed


if __name__ == "__main__":
    # Input
    reco_file = "data/simulated/input/reco_ATLAS.root"
    particle_file = "data/simulated/input/particle_ATLAS.root"
    reco_response_file = "data/simulated/input/reco_ATLAS_response.root"
    particle_response_file = "data/simulated/input/particle_ATLAS_response.root"

    # Output
    outname = "data/simulated/output/unfolding_input.root"

    # Reco variables
    list_recovars = [
        "Electron_size",
        "Muon_size",
        "Electron.PT",
        "Muon.PT",
        "Electron.Eta",
        "Muon.Eta",
        "Electron.Phi",
        "Muon.Phi",
        "Jet.Flavor",
        "Jet.PT",
        "Jet.Eta",
        "Jet.Phi",
    ]

    # Particle variables
    dict_partvars = {
        "11,13": [
            "Particle.PT",
            "Particle.Eta",
            "Particle.Phi",
            "Particle.E",
            "Particle.PID",
        ],
        "5": ["Particle.Eta", "Particle.Phi", "Particle.PT", "Particle.E"],
    }

    # Create and save reco and particle info
    out_hist_list = process(
        reco_tree_path=reco_file,
        part_tree_path=particle_file,
        list_recovars=list_recovars,
        dict_partvars=dict_partvars,
        do_response=False,
    )

    output = ROOT.TFile(outname, "RECREATE")
    reco_dir = output.mkdir("reco")
    particle_dir = output.mkdir("particle")
    for hist_tuple in out_hist_list:
        binning = hist_tuple[3][1:-1]
        bins = len(binning) - 1
        reco_histo = ROOT.TH1D(hist_tuple[0], hist_tuple[0], bins, array("d", binning))
        particle_histo = ROOT.TH1D(
            f"particle_{hist_tuple[0]}",
            f"particle_{hist_tuple[0]}",
            bins,
            array("d", binning),
        )
        for i, (reco_entries, particle_entries) in enumerate(
            zip(hist_tuple[1], hist_tuple[2])
        ):
            reco_histo.SetBinContent(i, reco_entries)
            particle_histo.SetBinContent(i, particle_entries)
        reco_dir.cd()
        reco_histo.Write()
        particle_dir.cd()
        particle_histo.Write()

    our_res_list = process(
        reco_tree_path=reco_response_file,
        part_tree_path=particle_response_file,
        list_recovars=list_recovars,
        dict_partvars=dict_partvars,
        do_response=True,
    )

    # Create and save response info
    for hist_tuple in our_res_list:
        binning = hist_tuple[3][1:-1]
        bins = len(binning) - 1
        reco_histo = ROOT.TH1D(
            f"mc_{hist_tuple[0]}", f"mc_{hist_tuple[0]}", bins, array("d", binning)
        )
        particle_histo = ROOT.TH1D(
            f"mc_particle_{hist_tuple[0]}",
            f"mc_particle_{hist_tuple[0]}",
            bins,
            array("d", binning),
        )
        response = ROOT.TH2D(
            f"particle_{hist_tuple[0]}_vs_{hist_tuple[0]}",
            f"particle_{hist_tuple[0]}_vs_{hist_tuple[0]}",
            bins,
            array("d", binning),
            bins,
            array("d", binning),
        )
        for i, (reco_entries, particle_entries) in enumerate(
            zip(hist_tuple[1], hist_tuple[2])
        ):
            reco_histo.SetBinContent(i, reco_entries)
            particle_histo.SetBinContent(i, particle_entries)
        for i in range(0, bins + 2):
            for j in range(0, bins + 2):
                response.SetBinContent(i, j, hist_tuple[4][j][i])
        reco_dir.cd()
        reco_histo.Write()
        response.Write()
        particle_dir.cd()
        particle_histo.Write()
