import sys
import os
import numpy as np
from scipy.signal import savgol_filter

from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QPushButton, QSlider, QLabel,
                             QFileDialog, QSpinBox, QAbstractSpinBox, QTreeWidget, 
                             QTreeWidgetItem, QScrollArea, QMessageBox, QStyle, 
                             QSizePolicy, QGroupBox, QFormLayout, QSplitter)
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QAction, QShortcut, QKeySequence, QPdfWriter, QPainter, QPageSize
import pyqtgraph as pg
import pyqtgraph.opengl as gl

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
        """Calcule la norme vectorielle pour un axe."""
        return np.linalg.norm(vector, axis=axis)


    # --- Cinématique ---
    def calculer_vecteur_vitesse_3D_joint(self, nom_joint):
        """Calcule le vecteur de vitesse 3D pour un joint."""
        positions = self.get_pos(nom_joint)
        if len(positions) == 0:
            return np.zeros((0,3))
        
        vitesses = np.vstack([np.zeros((1, 3)), np.diff(positions, axis=0)]) / self.dt
        return vitesses


    def calculer_vecteur_acceleration_3D_joint(self, nom_joint):
        """Calcule le vecteur d'accélération 3D pour un joint."""
        vitesses = self.calculer_vecteur_vitesse_3D_joint(nom_joint)
        if len(vitesses) == 0: 
            return np.zeros((0,3))
        accel = np.vstack([np.zeros((1, 3)), np.diff(vitesses, axis=0)]) / self.dt

        return accel


    def calculer_vecteur_jerk_3D_joint(self, nom_joint):
        """Calcule le vecteur de jerk (dérivée de l'accélération) 3D pour un joint ."""
        accel = self.calculer_vecteur_acceleration_3D_joint(nom_joint)
        if len(accel) == 0:
            return np.zeros((0,3))
        jerk = np.vstack([np.zeros((1, 3)), np.diff(accel, axis=0)]) / self.dt

        return jerk / 1000.0 # en m/s³


    # --- Contact et marche ---
    def calculer_hauteur_minimale_pied(self, side="Left"):
        """Calcule la hauteur minimale du pied (Z) pour détecter les phases de contact."""
        z_foot = self.get_pos(f"{side}Foot")[:, 2]
        z_toe = self.get_pos(f"{side}ToeBase")[:, 2]
        return np.minimum(z_foot, z_toe)


    def detecter_phases_contact(self, z_min):
        """Détecte les phases de contact du pied avec le sol en utilisant un filtre de Savitzky-Golay."""
        # Hauteur, vitesse et accélération du pied pour repérer l'appui
        window = int(0.055 * self.freq) 
        if window % 2 == 0: 
            window += 1
        if len(z_min) < window:
            return np.zeros_like(z_min)
            
        z_smooth = savgol_filter(z_min, window, polyorder=3)
        dz = savgol_filter(z_min, window, polyorder=3, deriv=1)  
        ddz = savgol_filter(z_min, window, polyorder=3, deriv=2) 
        
        z_threshold = np.percentile(z_smooth, 33)
        slope_tolerance = np.max(np.abs(dz)) * 0.10
        contact = np.zeros_like(z_min)
        
        for i in range(len(z_min)):
            dans_zone_basse = z_smooth[i] < z_threshold
            est_stable = np.abs(dz[i]) < slope_tolerance
            est_choc_impact = ddz[i] > 0
            if dans_zone_basse and (est_stable or est_choc_impact):
                contact[i] = 1

        # J'enlève les phases trop courtes
        min_frames = int(0.030 * self.freq)
        cleaned = contact.copy()
        current_val, count = cleaned[0], 0
        for i in range(len(cleaned)):
            if cleaned[i] == current_val:
                count += 1
            else:
                if count < min_frames: 
                    cleaned[i-count:i] = 1 - current_val
                current_val = cleaned[i]
                count = 1

        return cleaned


    def heel_strike_angle(self, side="Left"):
        """Calcule l'angle d'attaque du pas pour un pied"""
        # Angle du pied au moment du contact
        heel_pos = self.get_pos(f"{side}Foot")
        toe_pos = self.get_pos(f"{side}ToeBase")
        delta_z = toe_pos[:, 2] - heel_pos[:, 2]
        delta_xy = np.linalg.norm(toe_pos[:, :2] - heel_pos[:, :2], axis=1)
        angle_rad = np.arctan2(delta_z, delta_xy)
        return np.degrees(angle_rad)


    # --- Angles articul. ---
    def calculer_angle_3D(self, nom_a, nom_b, nom_c):
        """Calcule l'angle formé par trois joints dans l'espace 3D."""
        a, b, c = self.get_pos(nom_a), self.get_pos(nom_b), self.get_pos(nom_c)
        ba, bc = a - b, c - b
        norm_ba, norm_bc = np.linalg.norm(ba, axis=1), np.linalg.norm(bc, axis=1)
        cos_angle = np.sum(ba * bc, axis=1) / (norm_ba * norm_bc + 1e-8)
        return np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0)))


    # --- Centre de masse et stabilité ---
    def center_of_mass_oscillation(self):
        """Calcule l'oscillation du centre de masse (Hips) en Z et latéralement."""
        # Trajectoire moyenne des hanches puis écart latéral
        hips_pos = self.get_pos("Hips")
        z_rel = hips_pos[:, 2] - np.mean(hips_pos[:, 2])
        osc_z = np.std(z_rel) * 2 
        
        window_size = int(0.5 / self.dt)
        if window_size % 2 == 0: 
            window_size += 1 
        if window_size > len(hips_pos):
            if len(hips_pos)%2 == 0:
                window_size = len(hips_pos) - 1 
            else:
                window_size = len(hips_pos)
        
        if window_size < 3: 
            return 0, osc_z, np.zeros(len(hips_pos))

        path_x = np.convolve(hips_pos[:, 0], np.ones(window_size)/window_size, mode='same')
        path_y = np.convolve(hips_pos[:, 1], np.ones(window_size)/window_size, mode='same')
        
        v_forward_x = np.gradient(path_x)
        v_forward_y = np.gradient(path_y)
        v_lat_x = -v_forward_y
        v_lat_y = v_forward_x
        
        norms = np.sqrt(v_lat_x**2 + v_lat_y**2)
        v_lat_x = np.divide(v_lat_x, norms, out=np.zeros_like(v_lat_x), where=norms!=0)
        v_lat_y = np.divide(v_lat_y, norms, out=np.zeros_like(v_lat_y), where=norms!=0)

        rel_x = hips_pos[:, 0] - path_x
        rel_y = hips_pos[:, 1] - path_y
        deviation_laterale = rel_x * v_lat_x + rel_y * v_lat_y
        osc_x = np.std(deviation_laterale) * 2

        return osc_x, osc_z, deviation_laterale


    def compute_symmetry_index(self, val_left, val_right):
        """Calcule l'indice de symétrie entre deux valeurs (gauche et droite)."""
        # Ecart relatif entre les deux cotés
        if (val_left + val_right) == 0:
            return 0 
        return (np.abs(val_left - val_right) / (0.5 * (val_left + val_right))) * 100


    def compute_fft(self, signal_1d):
        """ Passage du signal temporel vers les fréquences"""
        fft_vals = np.fft.rfft(signal_1d)
        fft_freqs = np.fft.rfftfreq(len(signal_1d), d=self.dt)
        return fft_freqs, np.abs(fft_vals)


    def get_results(self):
        """Regroupement des mesures pour les valeurs et les graphes"""
        contact_l = self.detecter_phases_contact(self.calculer_hauteur_minimale_pied("Left"))
        contact_r = self.detecter_phases_contact(self.calculer_hauteur_minimale_pied("Right"))

        #Débuts des phases de contact = impacts
        impacts_l = np.where(np.diff(contact_l) == 1)[0]
        impacts_r = np.where(np.diff(contact_r) == 1)[0]
        impacts_all = np.sort(np.concatenate([impacts_l, impacts_r]))
        
        nb_pas = len(impacts_all)
        duree_min = (len(self.data) * self.dt) / 60.0
        cadence = (nb_pas / duree_min) if duree_min > 0 else 0
        
        step_duration = np.mean(np.diff(impacts_all)) * self.dt if len(impacts_all) > 1 else 0
        freq_pas = (1.0 / step_duration) if step_duration > 0 else 0

        # Frames où les deux pieds touchent le sol
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
        std_com_x = np.std(Hips[:, 0])
        std_com_y = np.std(Hips[:, 1])

        angle_knee_l = self.calculer_angle_3D("LeftUpLeg", "LeftLeg", "LeftFoot")
        angle_knee_r = self.calculer_angle_3D("RightUpLeg", "RightLeg", "RightFoot")
        si_genou = self.compute_symmetry_index(np.mean(angle_knee_l), np.mean(angle_knee_r))

        hs_angle_l = self.heel_strike_angle("Left")
        hs_angle_r = self.heel_strike_angle("Right")
        
        dist_heel_toe_l = np.mean(np.linalg.norm(LFoot - LToe, axis=1))
        dist_heel_toe_r = np.mean(np.linalg.norm(RFoot - RToe, axis=1))

        fft_f, fft_z = self.compute_fft(Hips[:, 2] - np.mean(Hips[:, 2]))

        v_LFoot = self.calculer_vecteur_vitesse_3D_joint("LeftFoot")
        v_RFoot = self.calculer_vecteur_vitesse_3D_joint("RightFoot")
        j_LFoot = self.calculer_vecteur_jerk_3D_joint("LeftFoot")
        j_RFoot = self.calculer_vecteur_jerk_3D_joint("RightFoot")

        angle_cou = self.calculer_angle_3D("Spine2", "Neck", "HeadTop")

        # une valeur moyenne ou globale par indicateur
        scalaires = {
            "Taille Acteur approx. (mm)": float(self.get_pos("HeadTop")[0] [2]),
            "Cadence (pas/min)": cadence,
            "Freq. Pas (Hz)": freq_pas,
            "Durée Moyenne Pas (s)": step_duration,
            "Longueur Pas Moyenne (mm)": longueur_pas,
            "Temps Double Appui (s)": double_appui_sec,
            "Oscillation Hips Z (mm)": osc_z,
            "Oscillation Hips Latéral (mm)": osc_x,
            "Stabilité COM X (Ecart-type)": std_com_x,
            "Stabilité COM Y (Ecart-type)": std_com_y,
            "Assymétrie Genou (SI %)": si_genou,
            "Assymétrie Vitesse Pieds (SI %)": self.compute_symmetry_index(np.mean(np.linalg.norm(v_LFoot, axis=1)), np.mean(np.linalg.norm(v_RFoot, axis=1))),
            "Dist. Talon-Pointe Gauche (mm)": dist_heel_toe_l,
            "Dist. Talon-Pointe Droite (mm)": dist_heel_toe_r,
            "Angle Impact Gauche (°)": np.mean(hs_angle_l[impacts_l]) if len(impacts_l) > 0 else 0,
            "Angle Impact Droit (°)": np.mean(hs_angle_r[impacts_r]) if len(impacts_r) > 0 else 0,
        }

        #  Données conservées frame par frame pour l'affichage
        graphiques = {
            "Phases de Contact Sol": {
                "unit": "0/1", "x_label": "Temps (s)",
                "lines": [(contact_l, "Pied Gauche", 'y', "yellow"), (contact_r, "Pied Droit", 'm', "magenta")]
            },
            "Déviation Latérale du Hips (COM)": {
                "unit": "mm", "x_label": "Temps (s)",
                "lines": [(dev_lat, "Déviation", 'w', "white")]
            },
            "Vitesse Pieds (Norme)": {
                "unit": "mm/s", "x_label": "Temps (s)",
                "lines": [(np.linalg.norm(v_LFoot, axis=1), "Vitesse G", 'c', "cyan"),
                          (np.linalg.norm(v_RFoot, axis=1), "Vitesse D", 'm', "magenta")]
            },
            "Jerk Pieds (Norme)": {
                "unit": "m/s³", "x_label": "Temps (s)",
                "lines": [(np.linalg.norm(j_LFoot, axis=1), "Jerk G", 'c', "cyan"),
                          (np.linalg.norm(j_RFoot, axis=1), "Jerk D", 'm', "magenta")]
            },
            "Heel Strike Angle (Attaque du pas)": {
                "unit": "°", "x_label": "Temps (s)",
                "lines": [(hs_angle_l, "Angle G", 'c', "cyan"), (hs_angle_r, "Angle D", 'm', "magenta")]
            },
            "Angle Genou (UpLeg-Leg-Foot)": {
                "unit": "°", "x_label": "Temps (s)",
                "lines": [(angle_knee_l, "Genou G", 'c', "cyan"), (angle_knee_r, "Genou D", 'm', "magenta")]
            },
            "Angle Cheville (Leg-Foot-Toe)": {
                "unit": "°", "x_label": "Temps (s)",
                "lines": [(self.calculer_angle_3D("LeftLeg", "LeftFoot", "LeftToeBase"), "Cheville G", 'c', "cyan"),
                          (self.calculer_angle_3D("RightLeg", "RightFoot", "RightToeBase"), "Cheville D", 'm', "magenta")]
            },
            "Spectre FFT (Hanche Z)": {
                "unit": "Amplitude", "x_label": "Fréquence (Hz)",
                "lines": [(fft_z, "Spectre Z", 'y', "yellow")],
                "x_data": fft_f # Axe X spécial pour FFT
            },
            "Angle de flexion du cou (Spine2-Neck-HeadTop)": {
                "unit": "°", "x_label": "Temps (s)",
                "lines": [(angle_cou, "Flexion Cou", 'y', "yellow")]
},
        }

        return scalaires, graphiques


# Interface
class CustomGLViewWidget(gl.GLViewWidget):
    # Gestion du déplacement de la vue avec le clic droit
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
    # Fenetre principale et liens entre les données et l'affichage
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
        self.tick_counter = 0 # Compteur pour optimiser les performances graphiques

        self.init_ui()
        self.init_shortcuts()
        self.define_tsv_topology()

    def init_ui(self):
        # Construction des panneaux, du lecteur et de la vue 3D
        menubar = self.menuBar()
        file_menu = menubar.addMenu("Fichier")
        file_menu.addAction(QAction("Ouvrir le répertoire 'Data'...", self, triggered=self.open_directory))
        file_menu.addSeparator()
        file_menu.addAction(QAction("Exporter en PDF...", self, triggered=self.export_pdf))

        nn_menu = menubar.addMenu("Réseaux de Neurones")
        nn_menu.addAction(QAction("Paramètres du Modèle", self))
        nn_menu.addAction(QAction("Lancer l'inférence", self))

        help_menu = menubar.addMenu("Aide")
        help_menu.addAction(QAction("Key bindings (Raccourcis)", self, triggered=self.show_keybindings))

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        main_layout = QVBoxLayout(central_widget)
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

        # Deux triangles pour sol plein
        verts = np.array([[-10, -10, 0], [10, -10, 0], [10, 10, 0], [-10, -10, 0], [10, 10, 0], [-10, 10, 0]])
        self.solid_floor = gl.GLMeshItem(vertexes=verts, faces=np.array([[0, 1, 2], [3, 4, 5]]), color=(0.5, 0.5, 0.5, 0.8), smooth=False)
        self.solid_floor.setVisible(False)
        self.view_3d.addItem(self.solid_floor)
        
        self.scatter_3d = gl.GLScatterPlotItem(color=(0.2, 0.8, 1, 0.8), size=8)
        self.tsv_envelope_lines = gl.GLLinePlotItem(color=pg.mkColor('g'), width=2.0, antialias=False) 
        self.skeleton_lines = gl.GLLinePlotItem(color=pg.mkColor(255, 200, 50, 255), width=3.0, antialias=True)
        self.view_3d.addItem(self.scatter_3d)
        self.view_3d.addItem(self.tsv_envelope_lines)
        self.view_3d.addItem(self.skeleton_lines)

        # Changement de vue
        self.camera_overlay = QWidget(self.view_3d)
        cam_layout = QHBoxLayout(self.camera_overlay)
        cam_layout.setContentsMargins(5, 5, 5, 5)
        for name, ele, azi in [("Face", 0, 90), ("Dessus", 90, 0), ("Coté", 0, 0), ("Iso Libre", 20, 45)]:
            btn = QPushButton(name)
            btn.clicked.connect(lambda _, e=ele, a=azi: self.reset_camera(e, a))
            cam_layout.addWidget(btn)
        self.camera_overlay.setGeometry(10, 10, 300, 40)
        
        center_layout.addWidget(self.view_3d, stretch=1)

        # Barre de lecture
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

        # DROITE = SPliiter vertical 
        # Valeurs numériques en haut, courbes en dessous
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


    def init_shortcuts(self):
        """ Keybindings"""
        QShortcut(QKeySequence("M"), self).activated.connect(self.toggle_markers)
        QShortcut(QKeySequence("S"), self).activated.connect(self.toggle_skeleton)
        QShortcut(QKeySequence("F"), self).activated.connect(self.toggle_floor)


    def reset_camera(self, elevation, azimuth):
        self.view_3d.opts['center'] = pg.Vector(0, 0, 0)
        self.view_3d.setCameraPosition(elevation=elevation, azimuth=azimuth)


    def define_tsv_topology(self):
        # Connexions entre marqueurs pour dessiner l'enveloppe du corps
        self.tsv_connections = [
            ("HeadTop", "HeadFront"), ("HeadTop", "HeadL"), ("HeadTop", "HeadR"),
            ("HeadL", "HeadR"), ("HeadL", "HeadFront"), ("HeadR", "HeadFront"),
            ("SpineTop", "Chest"), ("SpineTop", "BackL"), ("SpineTop", "BackR"),
            ("Chest", "BackL"), ("Chest", "BackR"), ("BackL", "BackR"),
            ("Chest", "WaistLFront"), ("Chest", "WaistRFront"),
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


    def open_directory(self):
        """Permet à l'utilisateur de sélectionner le répertoire DAta. Recherche des fichiers TSV et MKF puis association par identifiant"""
        dir_name = QFileDialog.getExistingDirectory(self, "Sélectionner le répertoire 'Data'")
        if not dir_name: return

        self.tree_files.clear()
        dir_tsv = os.path.join(dir_name, "TSV")
        dir_mkf = os.path.join(dir_name, "MKF")
        
        tsvs = set(os.listdir(dir_tsv)) if os.path.exists(dir_tsv) else set()
        mkfs = set(os.listdir(dir_mkf)) if os.path.exists(dir_mkf) else set()

        
        def get_identifier(filename):
            """Extrait l'identifiant d'un fichier en supprimant l'extension et le préfixe."""
            base = filename.rsplit('.', 1)[0]
            if '_' in base: return base.split('_', 1)[1]
            return base


        tsv_dict = {get_identifier(f): f for f in tsvs if f.endswith('.tsv')}
        mkf_dict = {get_identifier(f): f for f in mkfs if f.endswith('.mkf')}

        # Séparation des paires et des fichiers qui restent seuls
        paired_ids = set(tsv_dict.keys()).intersection(mkf_dict.keys())
        isolated_tsv_ids = set(tsv_dict.keys()) - paired_ids
        isolated_mkf_ids = set(mkf_dict.keys()) - paired_ids

        item_paired = QTreeWidgetItem(["Paires (TSV + MKF)"])
        for uid in sorted(paired_ids):
            child = QTreeWidgetItem([uid])
            child.setData(0, Qt.ItemDataRole.UserRole, {"tsv": os.path.join(dir_tsv, tsv_dict[uid]), "mkf": os.path.join(dir_mkf, mkf_dict[uid]), "name": uid})
            item_paired.addChild(child)
            
        item_isolated = QTreeWidgetItem(["Fichiers Isolés"])
        for uid in sorted(isolated_tsv_ids):
            child = QTreeWidgetItem([tsv_dict[uid] + " (TSV seul)"])
            child.setData(0, Qt.ItemDataRole.UserRole, {"tsv": os.path.join(dir_tsv, tsv_dict[uid]), "mkf": None, "name": uid})
            item_isolated.addChild(child)

        for uid in sorted(isolated_mkf_ids):
            child = QTreeWidgetItem([mkf_dict[uid] + " (MKF seul)"])
            child.setData(0, Qt.ItemDataRole.UserRole, {"tsv": None, "mkf": os.path.join(dir_mkf, mkf_dict[uid]), "name": uid})
            item_isolated.addChild(child)

        self.tree_files.addTopLevelItem(item_paired)
        self.tree_files.addTopLevelItem(item_isolated)
        self.tree_files.expandAll()


    def load_selected_data(self, item, column):
        """Charge les données TSV et MKF sélectionnées et remet à zéro la lecture."""
        data_paths = item.data(0, Qt.ItemDataRole.UserRole)
        if not data_paths: 
            return

        self.stop()
        self.tsv_data, self.mkf_data = None, None
        self.tsv_marker_names.clear()
        self.current_sequence_name = data_paths["name"]

        # Lecture des marqueurs TSV
        if data_paths["tsv"]:
            try:
                raw_tsv = []
                with open(data_paths["tsv"], 'r') as f:
                    for line in f:
                        if line.startswith("MARKER_NAMES"):
                            parts = line.split('\t')
                            self.tsv_marker_names = [p.strip() for p in parts[1:] if p.strip()]
                        elif not line.startswith("TRAJECTORY_TYPES") and not line.startswith("AL_"):
                            parts = line.strip().split('\t')
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

        # Lecture du squelette via MKF
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

                # Les 3 premières valeurs correspondent aux coordonnées XYZ
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
            self.build_descriptors_graphs()
        except Exception as e:
            print("Erreur Graphiques:", e)
        self.set_frame(0)


    def build_descriptors_graphs(self):
        """Construit les graphiques et les valeurs numériques à partir des données MKF. Nettoyage des anciens graphes avant de reconstruire l'affichage"""
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

        # Calcul des indicateurs à partir des données MKF
        desc = Descriptors(self.mkf_data, self.mkf_joint_names, freq=200)
        scalars, time_series = desc.get_results()

        for title, value in scalars.items():
            val_str = f"{value:.2f}"
            self.layout_scalars.addRow(title + " :", QLabel(f"<b>{val_str}</b>"))
            self.scalar_labels[title] = val_str

        # un widget par série temporelle
        for title, config in time_series.items():
            unit = config.get("unit", "")
            x_label = config.get("x_label", "Temps (s)")
            lines = config["lines"]
            x_data = config.get("x_data", None)

            container = QWidget()
            container_layout = QVBoxLayout(container)
            container_layout.setContentsMargins(0, 10, 0, 20)
            
            lbl_title = QLabel(f"<b>{title}</b>")
            lbl_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
            container_layout.addWidget(lbl_title)
            
            plot = pg.PlotWidget()
            plot.setMinimumHeight(180)
            plot.setLabel('left', text="", units=unit)
            plot.setLabel('bottom', text=x_label)
            plot.getPlotItem().setDefaultPadding(0.05)
            plot.setMouseEnabled(x=True, y=False)
            plot.enableAutoRange(axis='y') 

            if x_data is not None:
                time_axis = x_data
            else:
                time_axis = np.arange(len(lines[0][0])) / 200.0

            for data, name, pyqt_color, html_color in lines:
                plot.plot(time_axis, data, pen=pyqt_color)
            
            # Ne pas lier et ne pas mettre de curseur si c'est un graphe fréquentiel (FFT)
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

    def export_pdf(self):
        """ Export des valeurs et des graphes dans un rapport """
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

        # Marges et dimensions A4 à 300 DPI
        margin = 200
        page_width = writer.width()
        page_height = writer.height()
        usable_width = page_width - (2 * margin)
        y_cursor = margin + 100
        
        # Titre
        font = painter.font()
        font.setPointSize(18)
        font.setBold(True)
        painter.setFont(font)
        painter.drawText(margin, y_cursor, f"Rapport d'Analyse : {self.current_sequence_name}")
        y_cursor += 300
        
        # Valeurs numériques avec 2 col.
        font.setPointSize(11)
        font.setBold(False)
        painter.setFont(font)
        
        items = list(self.scalar_labels.items())
        mid_point = (len(items) + 1) // 2
        col1_x, col2_x = margin, margin + (usable_width // 2)
        
        start_y = y_cursor

        for i, (title, val) in enumerate(items):
            x_pos = col1_x if i < mid_point else col2_x
            y_pos = start_y + ((i % mid_point) * 120)
            painter.drawText(x_pos, y_pos, f"• {title} : {val}")
            
        y_cursor = start_y + (mid_point * 120) + 300

        # Graphiques
        for container in self.graph_containers:
            pixmap = container.grab()
            img_w = pixmap.width()
            img_h = pixmap.height()
            
            # Mise à l'échelle en conservant le ratio
            target_w = usable_width
            target_h = int((img_h / img_w) * target_w)
            
            # Saut de page si ça dépasse
            if y_cursor + target_h > page_height - margin:
                writer.newPage()
                y_cursor = margin
                
            painter.drawPixmap(margin, y_cursor, target_w, target_h, pixmap)
            y_cursor += target_h + 150 # Petit espace entre les graphiques

        painter.end()
        QMessageBox.information(self, "Succès", "Rapport PDF généré avec succès.")


    def show_keybindings(self):
        # Rappel du keybinding
        QMessageBox.information(self, "Aide - Raccourcis Clavier",
            "<b>M</b> : Afficher / Masquer les Marqueurs <br>"
            "<b>S</b> : Afficher / Masquer le Squelette <br>"
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
        
    def toggle_floor(self):
        self.show_floor = not self.show_floor
        self.solid_floor.setVisible(self.show_floor)
        self.grid.setVisible(not self.show_floor)

    def play(self):
        self.is_playing = True
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

        # Maj des curseurs qu'une fois sur 3 pendant l'animation pour opti
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


    def draw_tsv_envelope(self, frame):
        """ Dessine les lignes reliant les marqueurs à la frame voulue."""
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
        """ Dessin des Os définis par les couples d'articulations MKF"""
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

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MocapApp()
    window.showMaximized() 
    sys.exit(app.exec())