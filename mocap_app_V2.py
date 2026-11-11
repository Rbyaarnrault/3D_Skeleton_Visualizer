import sys
import os
import numpy as np
from scipy.signal import savgol_filter
from scipy.spatial import ConvexHull
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas

from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QPushButton, QSlider, QLabel,
                             QFileDialog, QSpinBox, QAbstractSpinBox, QTreeWidget, 
                             QTreeWidgetItem, QScrollArea, QMessageBox, QStyle, 
                             QSizePolicy, QGroupBox, QFormLayout, QSplitter,
                             QStackedWidget, QCheckBox, QProgressDialog)
from PyQt6.QtCore import Qt, QTimer, QRectF
from PyQt6.QtGui import QAction, QShortcut, QKeySequence, QPdfWriter, QPainter, QPageSize, QTextOption
import pyqtgraph as pg
import pyqtgraph.opengl as gl

# S'assure que matplotlib utilise le bon backend PyQt6
matplotlib.use('QtAgg')

class Descriptors:
    # Calculs biomécaniques à partir des positions MKF
    def __init__(self, mkf_data, joint_names, freq=200):
        self.data = mkf_data * 1000.0  # en millimètres
        self.names = joint_names
        self.freq = freq
        self.dt = 1.0 / freq

    def get_pos(self, nom_joint):
        """ Récupère les positions 3D d'un joint , lisse les données si nécessaire. """
        if nom_joint not in self.names:
            return np.zeros((self.data.shape[0], 3)) 
        idx = self.names.index(nom_joint)
        return self.lisser_donnees(self.data[:, idx, :])

    def lisser_donnees(self, vecteur, window_length=15, polyorder=3):
        """Applique un filtre de Savitzky-Golay"""
        if len(vecteur) < window_length: return vecteur 
        if vecteur.ndim > 1:
            return savgol_filter(vecteur, window_length, polyorder, axis=0)
        return savgol_filter(vecteur, window_length, polyorder)

    def norme_vectorielle(self, vector, axis=-1):
        return np.linalg.norm(vector, axis=axis)

    # --- Cinématique ---
    def calculer_vecteur_vitesse_3D_joint(self, nom_joint):
        positions = self.get_pos(nom_joint)
        if len(positions) == 0:
            return np.zeros((0,3))
        vitesses = np.vstack([np.zeros((1, 3)), np.diff(positions, axis=0)]) / self.dt
        return vitesses

    def calculer_vecteur_acceleration_3D_joint(self, nom_joint):
        vitesses = self.calculer_vecteur_vitesse_3D_joint(nom_joint)
        if len(vitesses) == 0: 
            return np.zeros((0,3))
        accel = np.vstack([np.zeros((1, 3)), np.diff(vitesses, axis=0)]) / self.dt
        return accel

    def calculer_vecteur_jerk_3D_joint(self, nom_joint):
        accel = self.calculer_vecteur_acceleration_3D_joint(nom_joint)
        if len(accel) == 0:
            return np.zeros((0,3))
        jerk = np.vstack([np.zeros((1, 3)), np.diff(accel, axis=0)]) / self.dt
        return jerk / 1000.0 # en m/s³

    # --- Energies ---
    def calculer_energie_cinetique(self, mass_approx=70.0):
        total_energy = np.zeros(self.data.shape[0])
        segment_mass = mass_approx / len(self.names)
        for joint in self.names:
            v_joint = self.calculer_vecteur_vitesse_3D_joint(joint) / 1000.0 # v en m/s
            v_norm_sq = np.sum(v_joint**2, axis=1)
            total_energy += 0.5 * segment_mass * v_norm_sq
        return total_energy

    # --- NOUVEAU DESCRIPTEUR: Quadrilatère ---
    def calculer_aire_quadrilatere(self):
        """ 
        Calcule l'aire du quadrilatère formé par les 4 extrémités du corps 
        en utilisant le produit vectoriel des diagonales.
        """
        lh = self.get_pos("LeftHand")
        rh = self.get_pos("RightHand")
        lf = self.get_pos("LeftFoot")
        rf = self.get_pos("RightFoot")
        
        # Diagonale 1: Main Gauche -> Pied Droit
        diag1 = rf - lh
        # Diagonale 2: Main Droite -> Pied Gauche
        diag2 = lf - rh
        
        # L'aire d'un quadrilatère dans l'espace est la moitié de la norme du produit vectoriel de ses diagonales
        cross_prod = np.cross(diag1, diag2)
        area_mm2 = 0.5 * np.linalg.norm(cross_prod, axis=1)
        
        # Conversion en cm² pour une meilleure lisibilité
        return area_mm2 / 100.0


    # --- Contact et marche ---
    def calculer_hauteur_minimale_pied(self, side="Left"):
        z_foot = self.get_pos(f"{side}Foot")[:, 2]
        z_toe = self.get_pos(f"{side}ToeBase")[:, 2]
        return np.minimum(z_foot, z_toe)
        
    def heel_strike_angle(self, side="Left"):
        heel_pos = self.get_pos(f"{side}Foot")
        toe_pos = self.get_pos(f"{side}ToeBase")
        delta_z = toe_pos[:, 2] - heel_pos[:, 2]
        delta_xy = np.linalg.norm(toe_pos[:, :2] - heel_pos[:, :2], axis=1)
        angle_rad = np.arctan2(delta_z, delta_xy)
        return np.degrees(angle_rad)

    def vitesse_angulaire_pied(self, side="Left"):
        angle = self.heel_strike_angle(side)
        window = int(0.055 * self.freq)
        if window % 2 == 0: window += 1
        if len(angle) > window:
            angle = savgol_filter(angle, window, 3)
        return np.gradient(angle, self.dt)

    def detecter_phases_contact(self, side="Left"):
        """
        Nouvelle détection robuste du contact (basée sur l'analyse visuelle de l'outil d'optimisation IA) :
- Vitesse de balayage du pied très faible (< seuil)
- Vitesse angulaire du pied très faible (< seuil, indiquant le pied à plat)
- Hauteur proche du sol
        """
        z_foot = self.get_pos(f"{side}Foot")[:, 2]
        z_toe = self.get_pos(f"{side}ToeBase")[:, 2]
        z_min = np.minimum(z_foot, z_toe)
        
        v_foot_norm = np.linalg.norm(self.calculer_vecteur_vitesse_3D_joint(f"{side}Foot"), axis=1)
        v_ang = np.abs(self.vitesse_angulaire_pied(side))
        
        # Seuils déduits de l'observation visuelle
        z_threshold = np.percentile(z_min, 30) # Doit être dans les 30% les plus bas
        v_thresh = 400.0 # Vitesse de balayage sous 400 mm/s
        w_thresh = 50.0  # Vitesse angulaire sous 50 °/s
        
        contact = np.zeros(len(z_min))
        
        for i in range(len(z_min)):
            if z_min[i] < z_threshold and v_foot_norm[i] < v_thresh and v_ang[i] < w_thresh:
                contact[i] = 1

        # 1. Fill gaps (Filtre anti-trous : comble les petites ruptures de < 120ms)
        gap_limit = int(0.120 * self.freq)
        last_one = -1
        for i in range(len(contact)):
            if contact[i] == 1:
                if last_one != -1 and (i - last_one) < gap_limit:
                    contact[last_one:i] = 1
                last_one = i

        # 2. Anti-bruit (Enlève les micro-contacts fantômes < 100ms)
        min_frames = int(0.100 * self.freq)
        cleaned = contact.copy()
        current_val, count = cleaned[0], 0
        for i in range(len(cleaned)):
            if cleaned[i] == current_val:
                count += 1
            else:
                if current_val == 1 and count < min_frames: 
                    cleaned[i-count:i] = 0
                current_val = cleaned[i]
                count = 1

        return cleaned

    # --- Angles articul. ---
    def calculer_angle_3D(self, nom_a, nom_b, nom_c):
        a, b, c = self.get_pos(nom_a), self.get_pos(nom_b), self.get_pos(nom_c)
        ba, bc = a - b, c - b
        norm_ba, norm_bc = np.linalg.norm(ba, axis=1), np.linalg.norm(bc, axis=1)
        cos_angle = np.sum(ba * bc, axis=1) / (norm_ba * norm_bc + 1e-8)
        return np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0)))


    # --- Centre de masse et stabilité ---
    def calcul_deviation_laterale(self, trajectoire):
        window_size = int(0.5 / self.dt)
        if window_size % 2 == 0: window_size += 1 
        if window_size > len(trajectoire):
            window_size = len(trajectoire) - 1 if len(trajectoire)%2 == 0 else len(trajectoire)
            
        if window_size < 3: return np.zeros(len(trajectoire))

        path_x = np.convolve(trajectoire[:, 0], np.ones(window_size)/window_size, mode='same')
        path_y = np.convolve(trajectoire[:, 1], np.ones(window_size)/window_size, mode='same')
        
        v_forward_x = np.gradient(path_x)
        v_forward_y = np.gradient(path_y)
        v_lat_x = -v_forward_y
        v_lat_y = v_forward_x
        
        norms = np.sqrt(v_lat_x**2 + v_lat_y**2)
        v_lat_x = np.divide(v_lat_x, norms, out=np.zeros_like(v_lat_x), where=norms!=0)
        v_lat_y = np.divide(v_lat_y, norms, out=np.zeros_like(v_lat_y), where=norms!=0)

        rel_x = trajectoire[:, 0] - path_x
        rel_y = trajectoire[:, 1] - path_y
        deviation_laterale = rel_x * v_lat_x + rel_y * v_lat_y
        return deviation_laterale

    def center_of_mass_oscillation(self):
        hips_pos = self.get_pos("Hips")
        z_rel = hips_pos[:, 2] - np.mean(hips_pos[:, 2])
        osc_z = np.std(z_rel) * 2 
        
        deviation_laterale = self.calcul_deviation_laterale(hips_pos)
        osc_x = np.std(deviation_laterale) * 2

        return osc_x, osc_z, deviation_laterale

    def compute_symmetry_index(self, val_left, val_right):
        if (val_left + val_right) == 0:
            return 0 
        return (np.abs(val_left - val_right) / (0.5 * (val_left + val_right))) * 100

    def compute_fft(self, signal_1d):
        fft_vals = np.fft.rfft(signal_1d)
        fft_freqs = np.fft.rfftfreq(len(signal_1d), d=self.dt)
        return fft_freqs, np.abs(fft_vals)

    def calculate_rom_spheres(self, bones):
        rom_features = {}
        
        for p, c in bones:
            if p in self.names and c in self.names:
                p_idx = self.names.index(p)
                c_idx = self.names.index(c)
                local_vec = self.data[:, c_idx] - self.data[:, p_idx]
                try:
                    hull = ConvexHull(local_vec)
                    # Conversion des mm³ en cm³ et mm² en cm² pour ne pas avoir de chiffres illisibles
                    vol_cm3 = hull.volume / 1000.0
                    area_cm2 = hull.area / 100.0
                except:
                    vol_cm3 = 0.0
                    area_cm2 = 0.0
                rom_features[f"{p}->{c}"] = {"Volume (cm³)": vol_cm3, "Surface (cm²)": area_cm2}
        return rom_features

    def get_results(self):
        # Utilise la nouvelle détection multimodale ultra-robuste
        contact_l = self.detecter_phases_contact("Left")
        contact_r = self.detecter_phases_contact("Right")

        impacts_l = np.where(np.diff(contact_l) == 1)[0]
        impacts_r = np.where(np.diff(contact_r) == 1)[0]
        impacts_all = np.sort(np.concatenate([impacts_l, impacts_r]))
        
        nb_pas = len(impacts_all)
        duree_min = (len(self.data) * self.dt) / 60.0
        cadence = (nb_pas / duree_min) if duree_min > 0 else 0
        
        step_duration = np.mean(np.diff(impacts_all)) * self.dt if len(impacts_all) > 1 else 0
        freq_pas = (1.0 / step_duration) if step_duration > 0 else 0

        # L'approche indépendante permet ENFIN le double appui (l==1 ET r==1)
        double_appui_frames = np.sum((contact_l == 1) & (contact_r == 1))
        double_appui_sec = double_appui_frames * self.dt

        LFoot = self.get_pos("LeftFoot")
        RFoot = self.get_pos("RightFoot")
        LToe = self.get_pos("LeftToeBase")
        RToe = self.get_pos("RightToeBase")
        Hips = self.get_pos("Hips")

        dist_xy = np.linalg.norm(LFoot[:, :2] - RFoot[:, :2], axis=1)
        longueur_pas = np.mean(dist_xy[impacts_all]) if len(impacts_all) > 0 else 0

        osc_x, osc_z, dev_lat = self.center_of_mass_oscillation()
        dev_lat_lfoot = self.calcul_deviation_laterale(LFoot)
        dev_lat_rfoot = self.calcul_deviation_laterale(RFoot)

        std_com_x = np.std(Hips[:, 0])
        std_com_y = np.std(Hips[:, 1])

        angle_knee_l = self.calculer_angle_3D("LeftUpLeg", "LeftLeg", "LeftFoot")
        angle_knee_r = self.calculer_angle_3D("RightUpLeg", "RightLeg", "RightFoot")
        si_genou = self.compute_symmetry_index(np.mean(angle_knee_l), np.mean(angle_knee_r))

        hs_angle_l = self.heel_strike_angle("Left")
        hs_angle_r = self.heel_strike_angle("Right")
        
        # NOUVEAU DESCRIPTEUR: Vitesse Angulaire
        v_ang_l = self.vitesse_angulaire_pied("Left")
        v_ang_r = self.vitesse_angulaire_pied("Right")
        
        # NOUVEAU DESCRIPTEUR: Quadrilatère / Amplitude Corporelle globale
        aire_quad_cm2 = self.calculer_aire_quadrilatere()
        aire_quad_max = np.max(aire_quad_cm2)
        aire_quad_min = np.min(aire_quad_cm2)
        
        dist_heel_toe_l = np.mean(np.linalg.norm(LFoot - LToe, axis=1))
        dist_heel_toe_r = np.mean(np.linalg.norm(RFoot - RToe, axis=1))

        fft_f, fft_z = self.compute_fft(Hips[:, 2] - np.mean(Hips[:, 2]))
        
        valid_idx = np.where(fft_f <= 10.0)[0]
        fft_f = fft_f[valid_idx]
        fft_z = fft_z[valid_idx]

        v_LFoot = self.calculer_vecteur_vitesse_3D_joint("LeftFoot")
        v_RFoot = self.calculer_vecteur_vitesse_3D_joint("RightFoot")
        j_LFoot = self.calculer_vecteur_jerk_3D_joint("LeftFoot")
        j_RFoot = self.calculer_vecteur_jerk_3D_joint("RightFoot")

        angle_cou = self.calculer_angle_3D("Spine2", "Neck", "HeadTop")
        
        energie_cinetique = self.calculer_energie_cinetique()

        scalaires = {
            "Taille Acteur approx. (mm)": (float(self.get_pos("HeadTop")[0][2]), 
                "Hauteur totale estimée de la personne à l'arrêt."),
            "Cadence (pas/min)": (cadence, 
                "Rythme de la marche. C'est le nombre de pas qu'effectue la personne en une minute."),
            "Freq. Pas (Hz)": (freq_pas, 
                "Nombre de pas réalisés par seconde."),
            "Durée Moyenne Pas (s)": (step_duration, 
                "Temps moyen nécessaire pour effectuer un pas complet."),
            "Longueur Pas Moyenne (mm)": (longueur_pas, 
                "Distance séparant les deux pieds au moment exact où le talon frappe le sol."),
            "Temps Double Appui (s)": (double_appui_sec, 
                "Durée totale passée avec les deux pieds en contact simultané avec le sol (recherche de stabilité)."),
            "Aire Quadrilatère Max (cm²)": (aire_quad_max, 
                "Amplitude globale maximale du corps lors du mouvement (Mains-Pieds)."),
            "Aire Quadrilatère Min (cm²)": (aire_quad_min, 
                "Resserrement global minimal du corps lors du mouvement."),
            "Oscillation Hips Z (mm)": (osc_z, 
                "Amplitude de rebond du bassin de haut en bas."),
            "Oscillation Hips Latéral (mm)": (osc_x, 
                "Balancement gauche-droite du bassin (déhanchement)."),
            "Stabilité COM X (Ecart-type)": (std_com_x, 
                "Dispersion du centre de gravité vers l'avant. Reflète la régularité de l'avancée."),
            "Stabilité COM Y (Ecart-type)": (std_com_y, 
                "Dispersion du centre de gravité sur les côtés. Révèle l'équilibre latéral."),
            "Assymétrie Genou (SI %)": (si_genou, 
                "Différence de pliure entre le genou gauche et droit (0% = symétrie parfaite)."),
            "Assymétrie Vitesse Pieds (SI %)": (self.compute_symmetry_index(np.mean(np.linalg.norm(v_LFoot, axis=1)), np.mean(np.linalg.norm(v_RFoot, axis=1))), 
                "Différence de dynamisme entre les deux pieds (peut révéler un membre traînant)."),
            "Dist. Talon-Pointe Gauche (mm)": (dist_heel_toe_l, 
                "Taille apparente du pied gauche pendant le mouvement."),
            "Dist. Talon-Pointe Droite (mm)": (dist_heel_toe_r, 
                "Taille apparente du pied droit pendant le mouvement."),
            "Angle Impact Gauche (°)": (np.mean(hs_angle_l[impacts_l]) if len(impacts_l) > 0 else 0, 
                "Inclinaison du pied gauche au moment de toucher le sol."),
            "Angle Impact Droit (°)": (np.mean(hs_angle_r[impacts_r]) if len(impacts_r) > 0 else 0, 
                "Inclinaison du pied droit au moment de toucher le sol."),
            "Energie Cinétique Globale Moyenne (J)": (np.mean(energie_cinetique),
                "Estimation de l'énergie cinétique (1/2 mv²) répartie sur l'ensemble du corps.")
        }

        graphiques = {
            "Phases de Contact Sol": {
                "unit": "0/1", "x_label": "Temps (s)",
                "lines": [(contact_l, "Pied Gauche", 'y', "yellow"), (contact_r, "Pied Droit", 'm', "magenta")],
                "desc": "Graphique en créneaux montrant très précisément les instants où chaque pied est en l'air (valeur basse) ou posé sur le sol (valeur haute)."
            },
            "Aire de l'enveloppe corporelle (Quadrilatère)": {
                "unit": "cm²", "x_label": "Temps (s)",
                "lines": [(aire_quad_cm2, "Aire (Mains-Pieds)", 'y', "yellow")],
                "desc": "Surface du quadrilatère formé par les deux mains et les deux pieds. Indique l'amplitude et l'expression du mouvement."
            },
            "Vitesse Angulaire Pied": {
                "unit": "°/s", "x_label": "Temps (s)",
                "lines": [(v_ang_l, "Vitesse Ang. G", 'c', "cyan"),
                          (v_ang_r, "Vitesse Ang. D", 'm', "magenta")],
                "desc": "Vitesse de rotation du pied. Reste proche de zéro lorsque le pied est à plat, bien ancré au sol."
            },
            "Déviation Latérale du Hips (COM)": {
                "unit": "mm", "x_label": "Temps (s)",
                "lines": [(dev_lat, "Déviation Bassin", 'w', "white")],
                "desc": "Écart du bassin par rapport à une trajectoire droite lissée. Si la ligne serpente beaucoup, la marche zigzague."
            },
            "Déviation Latérale des Pieds": {
                "unit": "mm", "x_label": "Temps (s)",
                "lines": [(dev_lat_lfoot, "Déviation P. Gauche", 'c', "cyan"),
                          (dev_lat_rfoot, "Déviation P. Droit", 'm', "magenta")],
                "desc": "Écart de la position des pieds par rapport à l'axe de marche. Utile pour voir si la personne marche les pieds écartés."
            },
            "Vitesse Pieds (Norme)": {
                "unit": "mm/s", "x_label": "Temps (s)",
                "lines": [(np.linalg.norm(v_LFoot, axis=1), "Vitesse G", 'c', "cyan"),
                          (np.linalg.norm(v_RFoot, axis=1), "Vitesse D", 'm', "magenta")],
                "desc": "Évolution de la vitesse de balayage des pieds. Le pic indique la vitesse maximale atteinte par le pied pendant qu'il est en l'air."
            },
            "Jerk Pieds (Norme)": {
                "unit": "m/s³", "x_label": "Temps (s)",
                "lines": [(np.linalg.norm(j_LFoot, axis=1), "Jerk G", 'c', "cyan"),
                          (np.linalg.norm(j_RFoot, axis=1), "Jerk D", 'm', "magenta")],
                "desc": "Le Jerk mesure les variations brusques d'accélération. Des pics pointus signifient un pas saccadé et un impact brutal."
            },
            "Heel Strike Angle (Attaque du pas)": {
                "unit": "°", "x_label": "Temps (s)",
                "lines": [(hs_angle_l, "Angle G", 'c', "cyan"), (hs_angle_r, "Angle D", 'm', "magenta")],
                "desc": "L'angle d'inclinaison du pied. Le moment où l'angle est le plus grand correspond souvent au moment où le talon frappe le sol."
            },
            "Angle Genou (UpLeg-Leg-Foot)": {
                "unit": "°", "x_label": "Temps (s)",
                "lines": [(angle_knee_l, "Genou G", 'c', "cyan"), (angle_knee_r, "Genou D", 'm', "magenta")],
                "desc": "L'évolution de la pliure des genoux. C'est la signature mécanique la plus forte pour identifier le style d'un individu."
            },
            "Angle Cheville (Leg-Foot-Toe)": {
                "unit": "°", "x_label": "Temps (s)",
                "lines": [(self.calculer_angle_3D("LeftLeg", "LeftFoot", "LeftToeBase"), "Cheville G", 'c', "cyan"),
                          (self.calculer_angle_3D("RightLeg", "RightFoot", "RightToeBase"), "Cheville D", 'm', "magenta")],
                "desc": "L'évolution de la flexion de la cheville."
            },
            "Angle de flexion du cou (Spine2-Neck-HeadTop)": {
                "unit": "°", "x_label": "Temps (s)",
                "lines": [(angle_cou, "Flexion Cou", 'y', "yellow")],
                "desc": "L'angle de la tête par rapport au buste. Permet de savoir si l'acteur marche la tête droite, ou s'il a tendance à regarder vers le bas."
            },
            "Energie Cinétique Globale (1/2 mv²)": {
                "unit": "Joules (approx)", "x_label": "Temps (s)",
                "lines": [(energie_cinetique, "Energie Totale", 'g', "lime")],
                "desc": "Répartition et évolution de l'énergie liée au mouvement de l'ensemble des segments corporels."
            },
            "Spectre FFT (Hanche Z)": {
                "unit": "Amplitude", "x_label": "Fréquence (Hz)",
                "lines": [(fft_z, "Spectre Z", 'y', "yellow")],
                "x_data": fft_f, 
                "desc": "L'empreinte digitale du rythme ! Le grand pic identifie la cadence principale parfaite, les petits pics dévoilent les petites boiteries et irrégularités du style."
            },
        }

        return scalaires, graphiques


# Interface 3D
class CustomGLViewWidget(gl.GLViewWidget):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.rPos = None
    def mousePressEvent(self, ev):
        if ev.button() == Qt.MouseButton.RightButton:
            self.rPos = ev.position()
        super().mousePressEvent(ev)
    def mouseMoveEvent(self, ev):
        if ev.buttons() == Qt.MouseButton.RightButton and self.rPos is not None:
            diff = ev.position() - self.rPos
            self.rPos = ev.position()
            self.pan(diff.x(), diff.y(), 0, relative='view')
        else: super().mouseMoveEvent(ev)
    def mouseReleaseEvent(self, ev):
        if ev.button() == Qt.MouseButton.RightButton: 
            self.rPos = None
        super().mouseReleaseEvent(ev)


#----------------------------------
# --- Application pricipale ---

class MocapApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Analyse de la Marche - Descripteurs et Visualisation 3D")
        
        self.current_frame = 0
        self.max_frames = 0
        self.is_playing = False
        
        self.speeds = [("x0.16", 1), ("x1", 6), ("x2", 12)]
        self.current_speed_idx = 1
        
        self.tsv_data, self.mkf_data = None, None
        self.tsv_marker_names, self.mkf_joint_names = [], []
        self.current_sequence_name = "Aucune Séquence"
        
        self.show_markers, self.show_skeleton, self.show_floor = True, True, False
        self.show_rom_spheres = False
        self.tick_counter = 0 

        self.rom_clouds_local = []
        self.rom_features_computed = {} 
        self.bones = [
            ("Hips", "Spine"), ("Spine", "Spine1"), ("Spine1", "Spine2"),
            ("Spine2", "Neck"), ("Neck", "Head"), ("Head", "HeadTop"),
            ("Spine2", "LeftShoulder"), ("LeftShoulder", "LeftArm"), ("LeftArm", "LeftForeArm"), ("LeftForeArm", "LeftHand"),
            ("Spine2", "RightShoulder"), ("RightShoulder", "RightArm"), ("RightArm", "RightForeArm"), ("RightForeArm", "RightHand"),
            ("Hips", "LeftUpLeg"), ("LeftUpLeg", "LeftLeg"), ("LeftLeg", "LeftFoot"), ("LeftFoot", "LeftToeBase"),
            ("Hips", "RightUpLeg"), ("RightUpLeg", "RightLeg"), ("RightLeg", "RightFoot"), ("RightFoot", "RightToeBase")
        ]
        
        # Structure de données pour dessiner l'enveloppe TSV (déplacée de define_tsv_topology vers __init__)
        self.tsv_connections = [
            ("HeadTop", "HeadFront"), ("HeadTop", "HeadL"), ("HeadTop", "HeadR"),
            ("HeadL", "HeadR"), ("HeadL", "HeadFront"), ("HeadR", "HeadFront"),
            ("SpineTop", "Chest"), ("SpineTop", "BackL"), ("SpineTop", "BackR"),
            ("Chest", "BackL"), ("Chest", "BackR"), ("BackL", "BackR"),
            ("Chest", "WaistLFront"), ("Chest", "WaistRFront"), # CORRECTION ICI : Ajout du 2eme element du tuple
            ("BackL", "WaistLBack"), ("BackR", "WaistRBack"),
            ("WaistLFront", "WaistLBack"), ("WaistRBack", "WaistRFront"), 
            ("WaistLFront", "WaistRFront"), ("WaistLBack", "WaistRBack"),
            ("SpineTop", "LShoulderTop"), ("BackL", "LShoulderBack"),
            ("LShoulderTop", "LShoulderBack"), 
            ("LShoulderTop", "LElbowOut"), ("LShoulderBack", "LElbowBack"),
            ("LElbowOut", "LElbowBack"), 
            ("LElbowOut", "LWristOut"), ("LElbowBack", "LWristIn"),
            ("LWristOut", "LWristIn"),
            ("LWristIn", "LThumb"), ("LThumb", "LForeFinger"), 
            ("LForeFinger", "LMiddleFinger"), ("LMiddleFinger", "LRingFinger"),
            ("LRingFinger", "LLittleFinger"), ("LLittleFinger", "LWristOut"),
            ("LWristOut", "LHandOut"), ("LHandOut", "LWristIn"),
            ("SpineTop", "RShoulderTop"), ("BackR", "RShoulderBack"),
            ("RShoulderTop", "RShoulderBack"),
            ("RShoulderTop", "RElbowOut"), ("RShoulderBack", "RElbowBack"),
            ("RElbowOut", "RElbowBack"), 
            ("RElbowOut", "RWristOut"), ("RElbowBack", "RWristIn"),
            ("RWristOut", "RWristIn"),
            ("RWristIn", "RThumb"), ("RThumb", "RForeFinger"),
            ("RForeFinger", "RMiddleFinger"), ("RMiddleFinger", "RRingFinger"),
            ("RRingFinger", "RLittleFinger"), ("RLittleFinger", "RWristOut"),
            ("RWristOut", "RHandOut"), ("RHandOut", "RWristIn"),
            ("WaistLFront", "LThigh"), ("WaistLBack", "LThigh"),
            ("LThigh", "LKneeOut"),
            ("LKneeOut", "LShin"), ("LShin", "LAnkleOut"), ("LKneeOut", "LAnkleOut"),
            ("LAnkleOut", "LHeelBack"), ("LHeelBack", "LAnkleIn"),
            ("LAnkleIn", "LForefootIn"), ("LForefootIn", "LToeTip"),
            ("LToeTip", "LForefootOut"), ("LForefootOut", "LAnkleOut"),
            ("LAnkleOut", "LAnkleIn"),
            ("WaistRFront", "RThigh"), ("WaistRBack", "RThigh"),
            ("RThigh", "RKneeOut"),
            ("RKneeOut", "RShin"), ("RShin", "RAnkleOut"), ("RKneeOut", "RAnkleOut"),
            ("RAnkleOut", "RHeelBack"), ("RHeelBack", "RAnkleIn"),
            ("RAnkleIn", "RForefootIn"), ("RForefootIn", "RToeTip"),
            ("RToeTip", "RForefootOut"), ("RForefootOut", "RAnkleOut"),
            ("RAnkleOut", "RAnkleIn")
        ]

        self.perceptual_data = {} 
        self.perceptual_plots = {}
        
        self.init_ui()
        self.init_shortcuts()
        # define_tsv_topology n'est plus appelé car la liste a été déplacée pour éviter le bug

    def init_ui(self):
        menubar = self.menuBar()
        file_menu = menubar.addMenu("Fichier")
        file_menu.addAction(QAction("Ouvrir le répertoire 'Data'...", self, triggered=self.open_directory))
        file_menu.addSeparator()
        file_menu.addAction(QAction("Exporter en PDF...", self, triggered=self.export_pdf))

        study_menu = menubar.addMenu("Étude Perceptuelle")
        study_menu.addAction(QAction("Vue Analyse 3D", self, triggered=lambda: self.stacked_widget.setCurrentIndex(0)))
        study_menu.addAction(QAction("Ouvrir Vue Étude Perceptuelle", self, triggered=self.show_perceptual_view))

        nn_menu = menubar.addMenu("Réseaux de Neurones")
        nn_menu.addAction(QAction("Paramètres du Modèle", self))
        nn_menu.addAction(QAction("Lancer l'inférence", self))

        help_menu = menubar.addMenu("Aide")
        help_menu.addAction(QAction("Key bindings (Raccourcis)", self, triggered=self.show_keybindings))

        self.stacked_widget = QStackedWidget()
        self.setCentralWidget(self.stacked_widget)
        
        # === VUE 1 : ANALYSE 3D ===
        view_3d_widget = QWidget()
        main_layout = QVBoxLayout(view_3d_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        self.main_splitter = QSplitter(Qt.Orientation.Horizontal)
        main_layout.addWidget(self.main_splitter)

        self.tree_files = QTreeWidget()
        self.tree_files.setHeaderLabel("Explorateur de Données")
        self.tree_files.setMinimumWidth(150) 
        self.tree_files.itemClicked.connect(self.load_selected_data)
        self.main_splitter.addWidget(self.tree_files)

        center_widget = QWidget()
        center_widget.setMinimumWidth(400) 
        center_layout = QVBoxLayout(center_widget)
        
        self.view_3d = CustomGLViewWidget()
        self.view_3d.opts['distance'] = 4 
        self.view_3d.opts['elevation'] = 20
        self.view_3d.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.grid = gl.GLGridItem()
        self.view_3d.addItem(self.grid)

        verts = np.array([[-10, -10, 0], [10, -10, 0], [10, 10, 0], [-10, -10, 0], [10, 10, 0], [-10, 10, 0]])
        self.solid_floor = gl.GLMeshItem(vertexes=verts, faces=np.array([[0, 1, 2], [3, 4, 5]]), color=(0.5, 0.5, 0.5, 0.8), smooth=False)
        self.solid_floor.setVisible(False)
        self.view_3d.addItem(self.solid_floor)
        
        self.scatter_3d = gl.GLScatterPlotItem(color=(0.2, 0.8, 1, 0.8), size=8)
        self.tsv_envelope_lines = gl.GLLinePlotItem(color=pg.mkColor('g'), width=2.0, antialias=False) 
        self.skeleton_lines = gl.GLLinePlotItem(color=pg.mkColor(255, 200, 50, 255), width=3.0, antialias=True)
        
        self.rom_spheres = gl.GLScatterPlotItem(color=(0.0, 1.0, 1.0, 0.4), size=6)
        self.rom_spheres.setVisible(False)

        self.view_3d.addItem(self.scatter_3d)
        self.view_3d.addItem(self.tsv_envelope_lines)
        self.view_3d.addItem(self.skeleton_lines)
        self.view_3d.addItem(self.rom_spheres)

        self.camera_overlay = QWidget(self.view_3d)
        cam_layout = QHBoxLayout(self.camera_overlay)
        cam_layout.setContentsMargins(5, 5, 5, 5)
        for name, ele, azi in [("Face", 0, 90), ("Dessus", 90, 0), ("Coté", 0, 0), ("Iso Libre", 20, 45)]:
            btn = QPushButton(name)
            btn.clicked.connect(lambda _, e=ele, a=azi: self.reset_camera(e, a))
            cam_layout.addWidget(btn)
        self.camera_overlay.setGeometry(10, 10, 300, 40)
        
        center_layout.addWidget(self.view_3d, stretch=1)

        player_layout = QHBoxLayout()
        for offset in [-10, -1]:
            btn = QPushButton(str(offset))
            btn.setFixedWidth(35)
            btn.clicked.connect(lambda _, o=offset: self.set_frame(self.current_frame + o))
            player_layout.addWidget(btn)
        
        self.btn_play_pause = QPushButton()
        self.btn_play_pause.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_MediaPlay))
        self.btn_play_pause.setFixedWidth(40)
        self.btn_play_pause.clicked.connect(self.toggle_play)
        player_layout.addWidget(self.btn_play_pause)

        self.btn_speed = QPushButton(self.speeds[self.current_speed_idx][0])
        self.btn_speed.setFixedWidth(60)
        self.btn_speed.clicked.connect(self.toggle_speed)
        player_layout.addWidget(self.btn_speed)

        for offset in [1, 10]:
            btn = QPushButton(f"+{offset}")
            btn.setFixedWidth(35)
            btn.clicked.connect(lambda _, o=offset: self.set_frame(self.current_frame + o))
            player_layout.addWidget(btn)

        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.sliderMoved.connect(self.set_frame)
        player_layout.addWidget(self.slider)

        frame_container = QHBoxLayout()
        self.frame_spinbox = QSpinBox()
        self.frame_spinbox.setButtonSymbols(QAbstractSpinBox.ButtonSymbols.NoButtons)
        self.frame_spinbox.setFixedWidth(50)
        self.frame_spinbox.setAlignment(Qt.AlignmentFlag.AlignRight)
        self.frame_spinbox.valueChanged.connect(self.set_frame)
        self.lbl_total_frames = QLabel(" / 0")
        
        frame_container.addWidget(self.frame_spinbox)
        frame_container.addWidget(self.lbl_total_frames)
        player_layout.addLayout(frame_container)
        
        center_layout.addLayout(player_layout, stretch=0)
        self.main_splitter.addWidget(center_widget)

        self.right_panel = QWidget()
        self.right_panel.setMinimumWidth(300)
        right_layout = QVBoxLayout(self.right_panel)
        right_layout.setContentsMargins(0,0,0,0)

        self.right_splitter = QSplitter(Qt.Orientation.Vertical)
        
        self.scalars_scroll = QScrollArea()
        self.scalars_scroll.setWidgetResizable(True)
        self.group_scalars = QGroupBox("Valeurs Numériques")
        self.layout_scalars = QFormLayout(self.group_scalars)
        self.scalars_scroll.setWidget(self.group_scalars)
        self.right_splitter.addWidget(self.scalars_scroll)

        graphs_wrapper = QWidget()
        wrapper_layout = QVBoxLayout(graphs_wrapper)
        wrapper_layout.setContentsMargins(0,0,0,0)
        wrapper_layout.setSpacing(0)

        toggle_bar = QWidget()
        toggle_bar.setStyleSheet("background-color: #3b4252; border: 1px solid #2e3440;")
        toggle_layout = QHBoxLayout(toggle_bar)
        toggle_layout.setContentsMargins(0, 2, 0, 2)
        btn_up = QPushButton("▲")
        btn_down = QPushButton("▼")
        btn_up.setStyleSheet("color: white; background-color: #4c566a; border: none;")
        btn_down.setStyleSheet("color: white; background-color: #4c566a; border: none;")
        btn_up.setFixedSize(30, 20)
        btn_down.setFixedSize(30, 20)
        btn_up.clicked.connect(lambda: self.right_splitter.setSizes([0, 1000]))
        btn_down.clicked.connect(lambda: self.right_splitter.setSizes([300, 700]))
        toggle_layout.addStretch()
        toggle_layout.addWidget(btn_up)
        toggle_layout.addWidget(btn_down)
        toggle_layout.addStretch()
        wrapper_layout.addWidget(toggle_bar)

        self.graphs_scroll = QScrollArea()
        self.graphs_scroll.setWidgetResizable(True)
        scroll_content = QWidget()
        self.graphs_layout = QVBoxLayout(scroll_content)
        self.graphs_layout.addStretch()
        self.graphs_scroll.setWidget(scroll_content)
        wrapper_layout.addWidget(self.graphs_scroll)

        self.right_splitter.addWidget(graphs_wrapper)
        right_layout.addWidget(self.right_splitter)

        self.main_splitter.addWidget(self.right_panel)
        self.main_splitter.setSizes([200, 800, 400])

        self.plots, self.cursors, self.graph_containers = [], [], []
        self.scalar_labels = {} 
        self.timer = QTimer()
        self.timer.timeout.connect(self.next_frame)

        self.stacked_widget.addWidget(view_3d_widget)

        # === VUE 2 : ETUDE PERCEPTUELLE ===
        self.perceptual_widget = QWidget()
        perc_layout = QHBoxLayout(self.perceptual_widget)
        
        perc_left = QWidget()
        perc_left.setMinimumWidth(250)
        perc_left_layout = QVBoxLayout(perc_left)
        
        btn_refresh_perc = QPushButton("Rafraîchir / Calculer les données")
        btn_refresh_perc.clicked.connect(self.compute_all_sequences_perceptual)
        perc_left_layout.addWidget(btn_refresh_perc)
        
        self.cb_all_desc = QCheckBox("Tout sélectionner/désélectionner")
        self.cb_all_desc.setChecked(True)
        self.cb_all_desc.stateChanged.connect(self.toggle_all_perceptual_plots)
        perc_left_layout.addWidget(self.cb_all_desc)
        
        perc_left_layout.addWidget(QLabel("<b>Descripteurs à afficher :</b>"))
        
        self.perc_checkboxes_layout = QVBoxLayout()
        self.perc_checkboxes = [] 
        
        perc_scroll_left = QScrollArea()
        perc_scroll_left.setWidgetResizable(True)
        perc_scroll_content = QWidget()
        perc_scroll_content.setLayout(self.perc_checkboxes_layout)
        perc_scroll_left.setWidget(perc_scroll_content)
        
        perc_left_layout.addWidget(perc_scroll_left)
        perc_layout.addWidget(perc_left)
        
        self.perc_graphs_scroll = QScrollArea()
        self.perc_graphs_scroll.setWidgetResizable(True)
        self.perc_graphs_scroll.setStyleSheet("QScrollArea { background-color: white; }")
        
        perc_graphs_content = QWidget()
        perc_graphs_content.setStyleSheet("background-color: white;")
        self.perc_graphs_layout = QVBoxLayout(perc_graphs_content)
        self.perc_graphs_scroll.setWidget(perc_graphs_content)
        
        perc_layout.addWidget(self.perc_graphs_scroll, stretch=1)
        self.stacked_widget.addWidget(self.perceptual_widget)

    def init_shortcuts(self):
        QShortcut(QKeySequence("M"), self).activated.connect(self.toggle_markers)
        QShortcut(QKeySequence("S"), self).activated.connect(self.toggle_skeleton)
        QShortcut(QKeySequence("F"), self).activated.connect(self.toggle_floor)
        QShortcut(QKeySequence("V"), self).activated.connect(self.toggle_rom_spheres)

    def reset_camera(self, elevation, azimuth):
        self.view_3d.opts['center'] = pg.Vector(0, 0, 0)
        self.view_3d.setCameraPosition(elevation=elevation, azimuth=azimuth)

    def open_directory(self):
        dir_name = QFileDialog.getExistingDirectory(self, "Sélectionner le répertoire 'Data'")
        if not dir_name: return

        self.tree_files.clear()
        dir_tsv = os.path.join(dir_name, "TSV")
        dir_mkf = os.path.join(dir_name, "MKF")
        
        tsvs = set(os.listdir(dir_tsv)) if os.path.exists(dir_tsv) else set()
        mkfs = set(os.listdir(dir_mkf)) if os.path.exists(dir_mkf) else set()
        
        def get_matching_id(filename):
            base = filename.rsplit('.', 1)[0]
            if '_' in base: 
                return base.split('_', 1)[1]
            return base

        tsv_dict = {get_matching_id(f): f for f in tsvs if f.endswith('.tsv')}
        mkf_dict = {get_matching_id(f): f for f in mkfs if f.endswith('.mkf')}

        paired_ids = set(tsv_dict.keys()).intersection(mkf_dict.keys())
        isolated_tsv_ids = set(tsv_dict.keys()) - paired_ids
        isolated_mkf_ids = set(mkf_dict.keys()) - paired_ids

        item_paired = QTreeWidgetItem(["Paires (TSV + MKF)"])
        for uid in sorted(paired_ids):
            nom_complet_mkf = mkf_dict[uid].rsplit('.', 1)[0]
            child = QTreeWidgetItem([nom_complet_mkf])
            
            child.setData(0, Qt.ItemDataRole.UserRole, {
                "tsv": os.path.join(dir_tsv, tsv_dict[uid]), 
                "mkf": os.path.join(dir_mkf, mkf_dict[uid]), 
                "name": uid,
                "display_name": nom_complet_mkf
            })
            item_paired.addChild(child)
            
        item_isolated = QTreeWidgetItem(["Fichiers Isolés"])
        for uid in sorted(isolated_tsv_ids):
            nom_complet_tsv = tsv_dict[uid].rsplit('.', 1)[0]
            child = QTreeWidgetItem([nom_complet_tsv + " (TSV seul)"])
            child.setData(0, Qt.ItemDataRole.UserRole, {
                "tsv": os.path.join(dir_tsv, tsv_dict[uid]), 
                "mkf": None, 
                "name": uid,
                "display_name": nom_complet_tsv
            })
            item_isolated.addChild(child)

        for uid in sorted(isolated_mkf_ids):
            nom_complet_mkf = mkf_dict[uid].rsplit('.', 1)[0]
            child = QTreeWidgetItem([nom_complet_mkf + " (MKF seul)"])
            child.setData(0, Qt.ItemDataRole.UserRole, {
                "tsv": None, 
                "mkf": os.path.join(dir_mkf, mkf_dict[uid]), 
                "name": uid,
                "display_name": nom_complet_mkf
            })
            item_isolated.addChild(child)

        self.tree_files.addTopLevelItem(item_paired)
        self.tree_files.addTopLevelItem(item_isolated)
        self.tree_files.expandAll()

    def build_rom_clouds_cache(self):
        self.rom_clouds_local = []
        if self.mkf_data is None: return
        desc = Descriptors(self.mkf_data, self.mkf_joint_names, freq=200)
        self.rom_features_computed = desc.calculate_rom_spheres(self.bones)
        
        for p, c in self.bones:
            if p in self.mkf_joint_names and c in self.mkf_joint_names:
                p_idx = self.mkf_joint_names.index(p)
                c_idx = self.mkf_joint_names.index(c)
                local_vec = self.mkf_data[:, c_idx] - self.mkf_data[:, p_idx]
                
                mean_pos = np.mean(local_vec, axis=0)
                centered_vec = local_vec - mean_pos
                
                mean_len = np.mean(np.linalg.norm(local_vec, axis=1))
                if mean_len > 0:
                    normalized = (centered_vec / mean_len) * 0.15 
                    self.rom_clouds_local.append((p_idx, normalized))

    def load_selected_data(self, item, column):
        data_paths = item.data(0, Qt.ItemDataRole.UserRole)
        if not data_paths: return

        self.stop()
        self.tsv_data, self.mkf_data = None, None
        self.tsv_marker_names.clear()
        self.current_sequence_name = data_paths["name"]

        if data_paths["tsv"]:
            try:
                raw_tsv = []
                with open(data_paths["tsv"], 'r') as f:
                    for line in f:
                        if line.startswith("MARKER_NAMES"):
                            parts = line.split('	')
                            self.tsv_marker_names = [p.strip() for p in parts[1:] if p.strip()]
                        elif not line.startswith("TRAJECTORY_TYPES") and not line.startswith("AL_"):
                            parts = line.strip().split('	')
                            if parts and parts[0]:
                                try:
                                    float(parts[0])
                                    row = []
                                    for x in parts:
                                        if x.strip():
                                            try: row.append(float(x.strip()))
                                            except ValueError: row.append(np.nan)
                                    raw_tsv.append(row)
                                except ValueError: pass
                if raw_tsv and self.tsv_marker_names:
                    expected_len = len(self.tsv_marker_names) * 3
                    cleaned_tsv = []
                    for r in raw_tsv:
                        if len(r) >= expected_len:
                            cleaned_tsv.append(r[:expected_len])
                        else: cleaned_tsv.append(r + [np.nan] * (expected_len - len(r)))
                    self.tsv_data = np.array(cleaned_tsv).reshape(len(cleaned_tsv), len(self.tsv_marker_names), 3) / 1000.0
            except Exception as e: print("Erreur TSV:", e)

        if data_paths["mkf"]:
            try:
                with open(data_paths["mkf"], 'r') as f:
                    lines = f.readlines()
                header = lines[0].strip().split()
                frames, joints = int(header[0]), int(header[1])
                self.mkf_joint_names = header[2:]
                
                raw_mkf = []
                for line in lines[1:]:
                    parts = line.strip().split()
                    if parts: raw_mkf.append([float(x) for x in parts])
                mkf_full = np.array(raw_mkf).reshape(frames, joints, 6)
                self.mkf_data = mkf_full[:, :, :3] / 1000.0
            except Exception as e: print("Erreur MKF:", e)

        self.max_frames = 0
        if self.tsv_data is not None: self.max_frames = self.tsv_data.shape[0]
        elif self.mkf_data is not None: self.max_frames = self.mkf_data.shape[0]

        if self.max_frames > 0:
            self.slider.setRange(0, self.max_frames - 1)
            self.frame_spinbox.setRange(0, self.max_frames - 1)
            self.lbl_total_frames.setText(f" / {self.max_frames - 1}")
        else:
            self.lbl_total_frames.setText(" / 0")
        
        try:
            if self.mkf_data is not None:
                self.build_rom_clouds_cache()
            self.build_descriptors_graphs()
        except Exception as e:
            print("Erreur Graphiques:", e)
        
        self.rom_spheres.setVisible(False)
        self.set_frame(0)

    def build_descriptors_graphs(self):
        for container in getattr(self, 'graph_containers', []):
            container.setParent(None)
            self.graphs_layout.removeWidget(container)

        self.graph_containers = []
        self.plots.clear()
        self.cursors.clear()
        
        while self.layout_scalars.rowCount() > 0:
            self.layout_scalars.removeRow(0)
        self.scalar_labels.clear()

        if self.mkf_data is None:
            return

        desc = Descriptors(self.mkf_data, self.mkf_joint_names, freq=200)
        scalars, time_series = desc.get_results()

        for title, (value, description) in scalars.items():
            val_str = f"{value:.2f}"
            lbl_titre_val = QLabel(f"<b>{title} : {val_str}</b>")
            lbl_desc = QLabel(f"<small><i>{description}</i></small>")
            lbl_desc.setWordWrap(True)
            lbl_desc.setStyleSheet("color: #666; margin-bottom: 5px;")
            
            val_container = QWidget()
            val_layout = QVBoxLayout(val_container)
            val_layout.setContentsMargins(0, 0, 0, 0)
            val_layout.setSpacing(2)
            val_layout.addWidget(lbl_titre_val)
            val_layout.addWidget(lbl_desc)
            
            self.layout_scalars.addRow(val_container)
            self.scalar_labels[title] = (val_str, description)

        for title, config in time_series.items():
            unit = config.get("unit", "")
            x_label = config.get("x_label", "Temps (s)")
            lines = config["lines"]
            x_data = config.get("x_data", None)
            desc_text = config.get("desc", "")

            container = QWidget()
            container_layout = QVBoxLayout(container)
            container_layout.setContentsMargins(0, 10, 0, 20)
            
            lbl_title = QLabel(f"<b>{title}</b>")
            lbl_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
            container_layout.addWidget(lbl_title)

            if desc_text:
                lbl_desc = QLabel(f"<small><i>{desc_text}</i></small>")
                lbl_desc.setWordWrap(True)
                lbl_desc.setAlignment(Qt.AlignmentFlag.AlignCenter)
                lbl_desc.setStyleSheet("color: #666;")
                container_layout.addWidget(lbl_desc)
            
            plot = pg.PlotWidget()
            plot.setMinimumHeight(150)
            plot.setMaximumHeight(250) 
            plot.setLabel('left', text="", units=unit)
            plot.setLabel('bottom', text=x_label)
            plot.getViewBox().setDefaultPadding(padding=0.05)
            plot.setMouseEnabled(x=True, y=False)
            plot.enableAutoRange(axis='y') 

            if x_data is not None:
                time_axis = x_data
            else:
                time_axis = np.arange(len(lines[0][0])) / 200.0

            if len(time_axis) > 0:
                plot.setXRange(0, time_axis[-1], padding=0.05)

            for data, name, pyqt_color, html_color in lines:
                plot.plot(time_axis, data, pen=pyqt_color)
            
            if x_data is None:
                cursor = pg.InfiniteLine(angle=90, movable=False, pen=pg.mkPen('r', width=2))
                plot.addItem(cursor)
                if self.plots:
                    plot.setXLink(self.plots[0])
                self.cursors.append(cursor)
                self.plots.append(plot)
            
            container_layout.addWidget(plot)
            
            legend_layout = QHBoxLayout()
            legend_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

            for data, name, pyqt_color, html_color in lines:
                legend_layout.addWidget(QLabel(f"<font color='{html_color}'>■</font> {name}"))
                legend_layout.addSpacing(15)
            container_layout.addLayout(legend_layout)
            
            self.graphs_layout.insertWidget(self.graphs_layout.count() - 1, container)
            self.graph_containers.append(container)


    def show_perceptual_view(self):
        self.stacked_widget.setCurrentIndex(1)
        if not self.perceptual_data:
            QMessageBox.information(self, "Info", "Cliquez sur 'Rafraîchir' pour calculer les données de toutes les séquences.")


    def toggle_all_perceptual_plots(self, state):
        for cb in self.perc_checkboxes:
            cb.setChecked(state == Qt.CheckState.Checked.value)


    def compute_all_sequences_perceptual(self):
        item_paired = self.tree_files.topLevelItem(0)
        if not item_paired or item_paired.childCount() == 0:
            QMessageBox.warning(self, "Erreur", "Aucune paire TSV/MKF dans le répertoire courant.")
            return

        progress = QProgressDialog("Calcul des descripteurs pour l'étude perceptuelle...", "Annuler", 0, item_paired.childCount(), self)
        progress.setWindowModality(Qt.WindowModality.WindowModal)

        self.perceptual_data = {}
        self.actor_names_mapping = {}

        for i in range(item_paired.childCount()):
            if progress.wasCanceled(): break
            progress.setValue(i)
            
            child = item_paired.child(i)
            paths = child.data(0, Qt.ItemDataRole.UserRole)
            uid = paths["name"]
            
            parts = uid.split('_')
            acteur = parts[0]
            seq_type = " ".join(parts[1:]) if len(parts) > 1 else "Sequence"
            
            full_mkf_name = paths.get("display_name", "")
            if full_mkf_name:
                initials = full_mkf_name.split('_')[0]
                self.actor_names_mapping[acteur] = f"{initials}_{acteur}"
            else:
                self.actor_names_mapping[acteur] = acteur

            if paths["mkf"]:
                try:
                    with open(paths["mkf"], 'r') as f:
                        lines = f.readlines()
                    header = lines[0].strip().split()
                    frames, joints = int(header[0]), int(header[1])
                    j_names = header[2:]
                    raw_mkf = [[float(x) for x in line.strip().split()] for line in lines[1:] if line.strip()]
                    mkf_arr = np.array(raw_mkf).reshape(frames, joints, 6)[:, :, :3] / 1000.0
                    
                    desc = Descriptors(mkf_arr, j_names, freq=200)
                    scalars, _ = desc.get_results()
                    
                    for desc_name, (val, text) in scalars.items():
                        if desc_name not in self.perceptual_data:
                            self.perceptual_data[desc_name] = {}
                        if acteur not in self.perceptual_data[desc_name]:
                            self.perceptual_data[desc_name][acteur] = {}
                        self.perceptual_data[desc_name][acteur][seq_type] = val

                except Exception as e:
                    print(f"Erreur perceptuelle sur {uid}:", e)

        progress.setValue(item_paired.childCount())
        self.update_perceptual_ui()


    def update_perceptual_ui(self):
        while self.perc_checkboxes_layout.count():
            w = self.perc_checkboxes_layout.takeAt(0).widget()
            if w: w.deleteLater()
            
        for w in self.perceptual_plots.values():
            w.setParent(None)
            self.perc_graphs_layout.removeWidget(w)
        self.perceptual_plots.clear()
        self.perc_checkboxes.clear()

        if not self.perceptual_data: return

        actors = []
        for desc_data in self.perceptual_data.values():
            for act in desc_data.keys():
                if act not in actors: actors.append(act)
        actors = sorted(actors)
        
        colors_rgb = plt.cm.tab10.colors
        def rgb_to_hex(rgb):
            return '#%02x%02x%02x' % (int(rgb[0]*255), int(rgb[1]*255), int(rgb[2]*255))
        
        actor_colors_hex = {act: rgb_to_hex(colors_rgb[i % len(colors_rgb)]) for i, act in enumerate(actors)}
        actor_colors_rgb = {act: colors_rgb[i % len(colors_rgb)] for i, act in enumerate(actors)}

        leg_group = QGroupBox("Acteurs (Sujets)")
        leg_lay = QVBoxLayout(leg_group)
        for act in actors:
            display_name = self.actor_names_mapping.get(act, act)
            lbl = QLabel(f"<font color='{actor_colors_hex[act]}'>■</font> {display_name}")
            leg_lay.addWidget(lbl)
        self.perc_checkboxes_layout.addWidget(leg_group)

        desc_group = QGroupBox("Afficher / Masquer")
        desc_lay = QVBoxLayout(desc_group)

        pg.setConfigOption('background', 'w')
        pg.setConfigOption('foreground', 'k')

        for desc_name, data_by_actor in self.perceptual_data.items():
            cb = QCheckBox(desc_name.split(' (')[0])
            cb.setChecked(True)
            desc_lay.addWidget(cb)
            self.perc_checkboxes.append(cb)
            
            fig, ax = plt.subplots(figsize=(12, 5))
            canvas = FigureCanvas(fig)
            canvas.wheelEvent = lambda event: event.ignore()
            
            canvas.setMinimumHeight(400)
            canvas.setMaximumHeight(400) 
            
            all_y_data = []
            x_labels = []
            
            for act in actors:
                if act in data_by_actor:
                    seqs_for_act = sorted(data_by_actor[act].keys())
                    
                    for seq_idx, seq in enumerate(seqs_for_act):
                        all_y_data.append(data_by_actor[act][seq])
                        clean_seq = seq.replace('_', ' ')
                        actor_index = actors.index(act) + 1
                        x_labels.append(f"{clean_seq} {actor_index}")

            x = range(1, len(all_y_data) + 1)
            group_size = len(seqs_for_act) if len(seqs_for_act) > 0 else 3 
            
            for i in range(0, len(all_y_data), group_size):
                xs = list(x)[i:i+group_size]
                act_index = i // group_size
                if act_index < len(actors):
                    color = actor_colors_rgb[actors[act_index]]
                    ax.plot(xs, all_y_data[i:i+group_size], color=color, marker='o', markersize=4)

            title = f"Mean Computed {desc_name.split(' (')[0]}"
            ax.set_xlabel("Movement and Subject", fontsize=10)
            ax.set_ylabel(desc_name.split(' (')[0], fontsize=10)
            ax.set_title(title, fontsize=12)
            
            ax.set_xticks(list(x))
            ax.set_xticklabels(x_labels, rotation=45, ha="right", fontsize=8)
            
            ax.set_facecolor('#f4f4f4')
            ax.grid(True, linestyle='--', alpha=0.5)
            fig.tight_layout()

            self.perc_graphs_layout.addWidget(canvas, stretch=0)
            self.perceptual_plots[desc_name] = canvas
            
            cb.stateChanged.connect(lambda state, widget=canvas: widget.setVisible(state == Qt.CheckState.Checked.value))

        self.perc_checkboxes_layout.addWidget(desc_group)
        self.perc_checkboxes_layout.addStretch()
        
        pg.setConfigOption('background', 'k')
        pg.setConfigOption('foreground', 'd')


    def export_pdf(self):
        if not getattr(self, 'graph_containers', None):
            QMessageBox.warning(self, "Erreur", "Aucune donnée calculée à exporter.")
            return

        path, _ = QFileDialog.getSaveFileName(self, "Exporter en PDF", f"Rapport_{self.current_sequence_name}.pdf", "PDF Files (*.pdf)")
        if not path:
            return
    
        writer = QPdfWriter(path)
        writer.setPageSize(QPageSize(QPageSize.PageSizeId.A4))
        writer.setResolution(300)
        
        painter = QPainter(writer)

        margin = 200
        page_width = writer.width()
        page_height = writer.height()
        usable_width = page_width - (2 * margin)
        y_cursor = margin + 100
        
        font = painter.font()
        font.setPointSize(18)
        font.setBold(True)
        painter.setFont(font)
        painter.drawText(margin, y_cursor, f"Rapport d'Analyse : {self.current_sequence_name}")
        y_cursor += 300
        
        if hasattr(self, 'rom_features_computed') and self.rom_features_computed:
            font.setPointSize(14)
            font.setBold(True)
            painter.setFont(font)
            painter.drawText(margin, y_cursor, "Volume d'Amplitude Articulaire (Kinematic Workspace)")
            y_cursor += 100
            
            font.setPointSize(10)
            font.setBold(False)
            painter.setFont(font)
            
            vols = [f["Volume (cm³)"] for f in self.rom_features_computed.values()]
            areas = [f["Surface (cm²)"] for f in self.rom_features_computed.values()]
            mean_vol = np.mean(vols) if vols else 0
            mean_area = np.mean(areas) if areas else 0
            
            painter.drawText(margin, y_cursor, f"Moyenne Globale du Squelette -> Volume : {mean_vol:.2f} cm³ | Surface : {mean_area:.2f} cm²")
            y_cursor += 150
            
            col_w = usable_width // 3
            font.setBold(True)
            painter.setFont(font)
            painter.drawText(margin, y_cursor, "Articulation")
            painter.drawText(margin + col_w, y_cursor, "Volume (cm³)")
            painter.drawText(margin + 2*col_w, y_cursor, "Surface (cm²)")
            y_cursor += 80
            
            font.setBold(False)
            painter.setFont(font)
            for j_name, feats in self.rom_features_computed.items():
                if y_cursor > page_height - margin - 100:
                    writer.newPage()
                    y_cursor = margin
                painter.drawText(margin, y_cursor, j_name)
                painter.drawText(margin + col_w, y_cursor, f"{feats['Volume (cm³)']:.2f}")
                painter.drawText(margin + 2*col_w, y_cursor, f"{feats['Surface (cm²)']:.2f}")
                y_cursor += 60
                
            y_cursor += 150

        for title, (val, desc) in self.scalar_labels.items():
            if y_cursor > page_height - margin - 200:
                writer.newPage()
                y_cursor = margin
                
            font.setPointSize(12)
            font.setBold(True)
            painter.setFont(font)
            painter.drawText(margin, y_cursor, f"• {title} : {val}")
            y_cursor += 60
            
            font.setPointSize(10)
            font.setItalic(True)
            font.setBold(False)
            painter.setFont(font)
            
            rect = QRectF(margin + 20, y_cursor, usable_width - 20, 150)
            option = QTextOption()
            option.setWrapMode(QTextOption.WrapMode.WordWrap)
            painter.drawText(rect, desc, option)
            y_cursor += 120 

        y_cursor += 200

        for container in self.graph_containers:
            pixmap = container.grab()
            img_w = pixmap.width()
            img_h = pixmap.height()
            
            target_w = usable_width
            target_h = int((img_h / img_w) * target_w)
            
            if y_cursor + target_h > page_height - margin:
                writer.newPage()
                y_cursor = margin
                
            painter.drawPixmap(margin, y_cursor, target_w, target_h, pixmap)
            y_cursor += target_h + 150 

        painter.end()
        QMessageBox.information(self, "Succès", "Rapport PDF professionnel généré avec succès.")


    def show_keybindings(self):
        QMessageBox.information(self, "Aide - Raccourcis Clavier",
            "<b>M</b> : Afficher / Masquer les Marqueurs <br>"
            "<b>S</b> : Afficher / Masquer le Squelette <br>"
            "<b>V</b> : Afficher / Masquer le Volume d'Amplitude (Sphères 3D)<br>"
            "<b>F</b> : Afficher / Masquer le Sol Solide<br><br>"
            "<b>Clic Droit (Maintenu)</b> : Déplacer la caméra <br>"
            "<b>Clic Gauche (Maintenu)</b> : Tourner la caméra<br>"
            "<b>Molette</b> : Zoomer / Dézoomer graphiques"
        )


    def toggle_play(self):
        if self.is_playing: self.stop()
        else: self.play()

    def toggle_speed(self):
        self.current_speed_idx = (self.current_speed_idx + 1) % len(self.speeds)
        self.btn_speed.setText(self.speeds[self.current_speed_idx][0])

    def toggle_markers(self):
        self.show_markers = not self.show_markers
        self.scatter_3d.setVisible(self.show_markers)
        self.tsv_envelope_lines.setVisible(self.show_markers)

    def toggle_skeleton(self):
        self.show_skeleton = not self.show_skeleton
        self.skeleton_lines.setVisible(self.show_skeleton)
        
    def toggle_rom_spheres(self):
        self.show_rom_spheres = not self.show_rom_spheres
        if self.current_frame == self.max_frames - 1:
            self.rom_spheres.setVisible(self.show_rom_spheres)
        else:
            self.rom_spheres.setVisible(False)

    def toggle_floor(self):
        self.show_floor = not self.show_floor
        self.solid_floor.setVisible(self.show_floor)
        self.grid.setVisible(not self.show_floor)

    def play(self):
        self.is_playing = True
        self.rom_spheres.setVisible(False) 
        self.btn_play_pause.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_MediaPause))
        self.timer.start(30)

    def stop(self):
        self.is_playing = False
        self.btn_play_pause.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_MediaPlay))
        self.timer.stop()

    def next_frame(self):
        step = self.speeds[self.current_speed_idx][1]
        next_f = self.current_frame + step
        if next_f < self.max_frames:
            self.set_frame(next_f)
        else:
            self.set_frame(self.max_frames - 1)
            self.stop()

    def set_frame(self, frame):
        if self.max_frames == 0:
            return
        
        frame = max(0, min(frame, self.max_frames - 1))
        self.current_frame = frame
        
        self.slider.setValue(frame)
        self.frame_spinbox.blockSignals(True)
        self.frame_spinbox.setValue(frame)
        self.frame_spinbox.blockSignals(False)

        self.tick_counter += 1
        if not self.is_playing or self.tick_counter % 3 == 0:
            time_sec = frame / 200.0

            for cursor in self.cursors:
                cursor.setValue(time_sec)

        if self.tsv_data is not None and self.show_markers:
            self.scatter_3d.setData(pos=self.tsv_data[frame])
            self.draw_tsv_envelope(frame)

        if self.mkf_data is not None and self.show_skeleton:
            self.draw_skeleton(frame)
            
            if self.show_rom_spheres and frame == self.max_frames - 1:
                self.rom_spheres.setVisible(True)
                self.draw_rom_spheres(frame)
            else:
                self.rom_spheres.setVisible(False)


    def draw_tsv_envelope(self, frame):
        if not self.tsv_marker_names:
            return
        
        lines_pos = []
        clean_names = [n.strip() for n in self.tsv_marker_names]
        
        def get_index(base_name):
            for i, name in enumerate(clean_names):
                if name.endswith(base_name): return i
            return -1

        for parent, child in self.tsv_connections:
            p_idx = get_index(parent)
            c_idx = get_index(child)
            
            if p_idx != -1 and c_idx != -1:
                if p_idx < self.tsv_data.shape[1] and c_idx < self.tsv_data.shape[1]:
                    p_pos = self.tsv_data[frame, p_idx]
                    c_pos = self.tsv_data[frame, c_idx]

                    if not (np.isnan(p_pos[0]) or np.isnan(c_pos[0])):
                        lines_pos.append(p_pos)
                        lines_pos.append(c_pos)
        
        if lines_pos:
            pos_array = np.array(lines_pos, dtype=np.float32)
            self.tsv_envelope_lines.setData(pos=pos_array, mode='lines')
        else:
            self.tsv_envelope_lines.setData(pos=np.empty((0,3), dtype=np.float32), mode='lines')


    def draw_skeleton(self, frame):
        bones = [
            ("Hips", "Spine"), ("Spine", "Spine1"), ("Spine1", "Spine2"),
            ("Spine2", "Neck"), ("Neck", "Head"), ("Head", "HeadTop"),
            ("Spine2", "LeftShoulder"), ("LeftShoulder", "LeftArm"), ("LeftArm", "LeftForeArm"), ("LeftForeArm", "LeftHand"),
            ("Spine2", "RightShoulder"), ("RightShoulder", "RightArm"), ("RightArm", "RightForeArm"), ("RightForeArm", "RightHand"),
            ("Hips", "LeftUpLeg"), ("LeftUpLeg", "LeftLeg"), ("LeftLeg", "LeftFoot"), ("LeftFoot", "LeftToeBase"),
            ("Hips", "RightUpLeg"), ("RightUpLeg", "RightLeg"), ("RightLeg", "RightFoot"), ("RightFoot", "RightToeBase")
        ]
        
        lines_pos = []
        for parent, child in bones:
            if parent in self.mkf_joint_names and child in self.mkf_joint_names:
                p_idx = self.mkf_joint_names.index(parent)
                c_idx = self.mkf_joint_names.index(child)
                lines_pos.append(self.mkf_data[frame, p_idx])
                lines_pos.append(self.mkf_data[frame, c_idx])

        if lines_pos:
            self.skeleton_lines.setData(pos=np.array(lines_pos, dtype=np.float32), mode='lines')
        else:
            self.skeleton_lines.setData(pos=np.empty((0,3), dtype=np.float32), mode='lines')
            
    def draw_rom_spheres(self, frame):
        all_pts = []
        for p_idx, local_cloud in self.rom_clouds_local:
            current_parent_pos = self.mkf_data[frame, p_idx]
            all_pts.append(local_cloud + current_parent_pos)
            
        if all_pts:
            self.rom_spheres.setData(pos=np.vstack(all_pts))

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MocapApp()
    window.showMaximized() 
    sys.exit(app.exec())
