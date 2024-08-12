import numpy as np


def compute_energy(mass, pt, phi, eta):
    p = compute_total_momentum(pt, phi, eta)
    e = np.sqrt(mass**2 + p**2)
    return e


def compute_total_momentum(p_t, phi, eta):
    p_x = p_t * np.cos(phi)
    p_y = p_t * np.sin(phi)
    p_z = p_t * np.sinh(eta)
    p = np.sqrt(p_x**2 + p_y**2 + p_z**2)
    return p


def compute_rapidity(p_z, e):
    y = 0.5 * np.log((e + p_z / (e - p_z)))
    return y


def compute_DR(eta1, phi1, eta2, phi2):
    delta_eta = eta1 - eta2

    delta_phi = phi1 - phi2
    if delta_phi > np.pi:
        delta_phi -= 2 * np.pi
    if delta_phi < -np.pi:
        delta_phi += 2 * np.pi

    delta_R = np.sqrt(delta_eta**2 + delta_phi**2)

    return delta_R


def compute_DR_reco(eta, phi):
    delta_eta = eta[:, 0] - eta[:, 1]
    delta_phi = phi[:, 0] - phi[:, 1]

    delta_phi = np.where(delta_phi > np.pi, delta_phi - 2 * np.pi, delta_phi)
    delta_phi = np.where(delta_phi < -np.pi, delta_phi + 2 * np.pi, delta_phi)

    DR_b1b2 = np.sqrt(delta_eta**2 + delta_phi**2)

    return DR_b1b2


def compute_DR_particle(eta1, eta2, phi1, phi2):
    delta_eta = eta1 - eta2
    delta_phi = phi1 - phi2

    delta_phi = np.where(delta_phi > np.pi, delta_phi - 2 * np.pi, delta_phi)
    delta_phi = np.where(delta_phi < -np.pi, delta_phi + 2 * np.pi, delta_phi)

    DR_b1b2 = np.sqrt(delta_eta**2 + delta_phi**2)

    return DR_b1b2


def compute_invariant_mass(obj_1_pt, obj_1_eta, obj_1_phi, obj_1_e, obj_2_pt, obj_2_eta, obj_2_phi, obj_2_e):
    obj_1_px = obj_1_pt * np.cos(obj_1_phi)
    obj_1_py = obj_1_pt * np.sin(obj_1_phi)
    obj_1_pz = obj_1_pt * np.sinh(obj_1_eta)

    obj_2_px = obj_2_pt * np.cos(obj_2_phi)
    obj_2_py = obj_2_pt * np.sin(obj_2_phi)
    obj_2_pz = obj_2_pt * np.sinh(obj_2_eta)

    invariant_mass = np.sqrt(
        (obj_1_e + obj_2_e) ** 2 - (obj_1_px + obj_2_px) ** 2 - (obj_1_py + obj_2_py) ** 2 - (obj_1_pz + obj_2_pz) ** 2
    )
    return invariant_mass


def compute_invariant_mass_particle(pt1, eta1, phi1, e1, pt2, eta2, phi2, e2):
    px1 = pt1 * np.cos(phi1)
    py1 = pt1 * np.sin(phi1)
    pz1 = pt1 * np.sinh(eta1)

    px2 = pt2 * np.cos(phi2)
    py2 = pt2 * np.sin(phi2)
    pz2 = pt2 * np.sinh(eta2)

    invariant_mass = np.sqrt((e1 + e2) ** 2 - (px1 + px2) ** 2 - (py1 + py2) ** 2 - (pz1 + pz2) ** 2)
    return invariant_mass


def compute_invariant_mass_reco(pt, eta, phi, e):
    eta1, eta2 = eta[:, 0], eta[:, 1]
    phi1, phi2 = phi[:, 0], phi[:, 1]
    pt1, pt2 = pt[:, 0], pt[:, 1]
    e1, e2 = e[:, 0], e[:, 1]

    px1 = pt1 * np.cos(phi1)
    py1 = pt1 * np.sin(phi1)
    pz1 = pt1 * np.sinh(eta1)

    px2 = pt2 * np.cos(phi2)
    py2 = pt2 * np.sin(phi2)
    pz2 = pt2 * np.sinh(eta2)

    invariant_mass = np.sqrt((e1 + e2) ** 2 - (px1 + px2) ** 2 - (py1 + py2) ** 2 - (pz1 + pz2) ** 2)
    return invariant_mass
