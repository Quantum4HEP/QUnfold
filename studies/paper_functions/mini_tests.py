import uproot
import numpy as np

if __name__ == "__main__":
    old_file = "data/simulated/output/unfolding_input.root"
    new_file = "data/simulated/output/new_unfolding_input.root"

    # Reco check
    old_reco = uproot.open(old_file)["reco"]
    new_reco = uproot.open(new_file)["reco"]

    for key in old_reco.keys():
        for el_old, el_new in zip(old_reco[key].to_numpy(), new_reco[key].to_numpy()):
            if not np.array_equal(el_old, el_new):
                print(f"Problem with {key}")

    # Particle check
    old_particle = uproot.open(old_file)["particle"]
    new_particle = uproot.open(new_file)["particle"]

    for key in old_particle.keys():
        for el_old, el_new in zip(
            old_particle[key].to_numpy(), new_particle[key].to_numpy()
        ):
            if not np.array_equal(el_old, el_new):
                print(f"Problem with {key}")
