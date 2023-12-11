import uproot
import awkward as ak
import numpy as np


def get_info_particle_level(particle_info, list_particleIDs, list_partvars):
    lepton_mask = ak.Array(np.zeros(len(particle_info), dtype=bool))
    for ids in list_particleIDs:
        lepton_mask = lepton_mask | (abs(particle_info["Particle.PID"]) == ids)
    outvars = {}
    for partvars in list_partvars:
        temp = np.sort(ak.mask(particle_info[partvars], lepton_mask), axis=-1)
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
    # [int(num) for num in list(part_val_dict.keys())[0].split(',')]
    # print([int(num) for num in list(part_val_dict.keys())[0].split(',')])
    # print(all_part_info["11,13"]["Particle.PT"])
    return [events_idx, reco_info, all_part_info]


def hist_resp_builder(
    reco_lead, reco_sublead, part_lead, part_sublead, do_response=False
):
    num_bins = 19
    lead_binning = np.linspace(50, 200, num_bins + 1)
    sublead_binning = np.linspace(30, 120, num_bins + 1)
    mask_lead = reco_lead != None
    mask_sublead = reco_sublead != None
    # print(reco_lead)
    # print(ak.to_numpy(reco_lead))
    reco_lead = reco_lead[mask_lead].astype(float)
    reco_sublead = reco_sublead[mask_sublead].astype(float)

    hist_reco_lead, _ = np.histogram(reco_lead, lead_binning)
    hist_reco_sublead, _ = np.histogram(reco_sublead, sublead_binning)
    hist_part_lead, _ = np.histogram(ak.to_numpy(part_lead), lead_binning)
    hist_part_sublead, _ = np.histogram(ak.to_numpy(part_sublead), sublead_binning)

    if do_response:
        lead_response, _, _ = np.histogram2d(
            ak.to_numpy(reco_lead),
            ak.to_numpy(part_lead[mask_lead]).astype(float),
            lead_binning,
        )

        sublead_response, _, _ = np.histogram2d(
            ak.to_numpy(reco_sublead),
            ak.to_numpy(part_sublead[mask_sublead]),
            sublead_binning,
        )
        return lead_response, hist_part_lead, sublead_response, hist_part_sublead
    else:
        return hist_reco_lead, hist_part_lead, hist_reco_sublead, hist_part_sublead


def process(
    reco_tree_path, part_tree_path, list_recovars, dict_partvars, do_response=False
):
    events_idx, reco_info, particle_info = tree_reader(
        reco_tree_path, part_tree_path, list_recovars, dict_partvars
    )

    num_events = len(events_idx)
    electron_size = reco_info["Electron_size"]
    muon_size = reco_info["Muon_size"]
    electron_pt = reco_info["Electron.PT"]
    muon_pt = reco_info["Muon.PT"]

    # emu_part_vars = get_info_particle_level(particle_info,[11,13],["Particle.PID", "Particle.PT", "Particle.Eta","Particle.Phi","Particle.E"])
    # part_level_vars = get_info_particle_level(particle_info,[int(num) for num in list(dict_partvars.keys())[0].split(',')],["Particle.PID", "Particle.PT", "Particle.Eta","Particle.Phi","Particle.E"])
    # part_level_vars_jet = get_info_particle_level(particle_info,[5],["Particle.PID",  "Particle.Eta","Particle.Phi"])
    # print(part_level_vars["Particle.PT"])

    mask_ee = (electron_size == 2) & (muon_size == 0)
    events_ee_idx = events_idx[mask_ee]
    reco_pt_ee = np.sort(electron_pt[mask_ee], axis=-1)

    mask_mumu = (electron_size == 0) & (muon_size == 2)
    events_mumu_idx = events_idx[mask_mumu]
    reco_pt_mumu = np.sort(muon_pt[mask_mumu], axis=-1)

    mask_emu = (electron_size == 1) & (muon_size == 1)
    events_emu_idx = events_idx[mask_emu]
    reco_pt_emu = np.sort(
        np.hstack((electron_pt[mask_emu], muon_pt[mask_emu])), axis=-1
    )

    temp = np.concatenate((reco_pt_ee, reco_pt_mumu, reco_pt_emu))
    concat_pt = np.array([[None, None]] * num_events)
    concat_idx = np.concatenate((events_ee_idx, events_mumu_idx, events_emu_idx))
    concat_pt[mask_ee | mask_mumu | mask_emu] = temp[np.argsort(concat_idx)]
    reco_pT_lep1, reco_pT_lep2 = concat_pt[:, 1], concat_pt[:, 0]

    # reco_pT_lep1______ = np.loadtxt("reco_pT_lep1.txt", delimiter=",")
    # reco_pT_lep2______ = np.loadtxt("reco_pT_lep2.txt", delimiter=",")
    # assert np.all(reco_pT_lep1[reco_pT_lep1 != None] == reco_pT_lep1______)
    # assert np.all(reco_pT_lep2[reco_pT_lep2 != None] == reco_pT_lep2______)

    # particle_pT_lep1______ = np.loadtxt("particle_pT_lep1.txt", delimiter=",")
    # particle_pT_lep2______ = np.loadtxt("particle_pT_lep2.txt", delimiter=",")
    # assert np.all(particle_pT_lep1 == particle_pT_lep1______)
    # assert np.all(particle_pT_lep2 == particle_pT_lep2______)
    particle_pT_lep1, particle_pT_lep2 = particle_info["11,13"]["Particle.PT"]
    return hist_resp_builder(
        reco_pT_lep1, reco_pT_lep2, particle_pT_lep1, particle_pT_lep2, do_response
    )

    # resp_pt_lep1______ = np.loadtxt("resp_pt_lep1.txt")
    # resp_pt_lep2______ = np.loadtxt("resp_pt_lep2.txt")
    # assert np.all(pt_lep1_response == resp_pt_lep1______)
    # assert np.all(pt_lep2_response == resp_pt_lep2______)


if __name__ == "__main__":
    list_recovars = [
        "Electron_size",
        "Muon_size",
        "Electron.PT",
        "Muon.PT",
        "Electron.Eta",
    ]
    # list_partvars = ["Particle.PID", "Particle.PT", "Particle.Eta","Particle.Phi","Particle.E"]
    dict_partvars = {
        "11,13": ["Particle.PT", "Particle.Eta", "Particle.Phi", "Particle.E"],
        "5": ["Particle.Eta", "Particle.Phi"],
    }
    reco_tree_path = "tag_1_delphes_events_small.root"
    part_tree_path = "unweighted_events_small.root"
    outp = process(
        reco_tree_path, part_tree_path, list_recovars, dict_partvars, do_response=False
    )
    # print(outp[0],outp[1],outp[2],outp[3])
    outp = process(
        reco_tree_path, part_tree_path, list_recovars, dict_partvars, do_response=True
    )
    # print(outp[0],outp[1],outp[2],outp[3])
    print("ho fatto la response")
