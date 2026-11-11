import sys
import os
import numpy as np
from scipy.signal import savgol_filter
from scipy.spatial import ConvexHull

from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QPushButton, QSlider, QLabel,
                             QFileDialog, QSplitter, QMessageBox, QGroupBox, 
                             QTextEdit, QScrollArea)
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QAction, QShortcut, QKeySequence
import pyqtgraph as pg
import pyqtgraph.opengl as gl

# =====================================================================
# CLASSE D'EXTRACTION DES FEATURES
# =====================================================================
class FeatureExtractor:
    def __init__(self, mkf_data, joint_names, freq=200):
        self.data = mkf_data * 1000.0
        self.names = joint_names
        self.freq = freq
        self.dt = 1.0 / freq

    def get_pos(self, nom_joint):
        if nom_joint not in self.names: return np.zeros((self.data.shape[0], 3))
        idx = self.names.index(nom_joint)
        # Lissage pour éviter trop de bruit sur les dérivées
        return savgol_filter(self.data[:, idx, :], window_length=11, polyorder=3, axis=0)

    def extract_features(self, side="Left"):
        """ Extrait UNIQUEMENT les 4 variables demandées : Hauteur Pointe, Vitesse Pointe, Angle, Jerk Pointe """
        toe_pos = self.get_pos(f"{side}ToeBase")
        z_toe = toe_pos[:, 2]
        
        # Vitesse Z de la pointe
        vz_toe = np.gradient(z_toe, self.dt)
        
        # Accélération Z de la pointe (étape intermédiaire)
        az_toe = np.gradient(vz_toe, self.dt)
        
        # Jerk Z de la pointe (Dérivée de l'accélération, en m/s^3)
        jz_toe = np.gradient(az_toe, self.dt) / 1000.0
        
        # Angle de la cheville (Leg -> Foot -> ToeBase)
        nom_a = f"{side}Leg"
        nom_b = f"{side}Foot"
        nom_c = f"{side}ToeBase"
        
        a, b, c = self.get_pos(nom_a), self.get_pos(nom_b), self.get_pos(nom_c)
        ba, bc = a - b, c - b
        norm_ba, norm_bc = np.linalg.norm(ba, axis=1), np.linalg.norm(bc, axis=1)
        cos_angle = np.sum(ba * bc, axis=1) / (norm_ba * norm_bc + 1e-8)
        angle_cheville = np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0)))
        
        # Matrice X : Cols : 0: Z pointe, 1: VZ pointe, 2: Angle cheville, 3: Jerk pointe
        X = np.column_stack([z_toe, vz_toe, angle_cheville, jz_toe])
        feature_names = ["Hauteur", "Vitesse", "Angle", "Jerk"]
        
        return X, feature_names


# =====================================================================
# VUE 3D
# =====================================================================
class CustomGLViewWidget(gl.GLViewWidget):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.rPos = None
    def mousePressEvent(self, ev):
        if ev.button() == Qt.MouseButton.RightButton: self.rPos = ev.position()
        super().mousePressEvent(ev)
    def mouseMoveEvent(self, ev):
        if ev.buttons() == Qt.MouseButton.RightButton and self.rPos is not None:
            diff = ev.position() - self.rPos
            self.rPos = ev.position()
            self.pan(diff.x(), diff.y(), 0, relative='view')
        else: super().mouseMoveEvent(ev)
    def mouseReleaseEvent(self, ev):
        if ev.button() == Qt.MouseButton.RightButton: self.rPos = None
        super().mouseReleaseEvent(ev)


# =====================================================================
# APPLICATION D'ANALYSE STATISTIQUE ET CORRÉLATION
# =====================================================================
class AnnotationApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Contact Optimizer - Corrélation Statistique")
        
        self.mkf_data, self.mkf_joint_names = None, []
        self.max_frames = 0
        self.current_frame = 0
        self.is_playing = False
        self.freq = 200.0

        self.bones = [
            ("Hips", "Spine"), ("Spine", "Spine1"), ("Spine1", "Spine2"),
            ("Spine2", "Neck"), ("Neck", "Head"), ("Head", "HeadTop"),
            ("Spine2", "LeftShoulder"), ("LeftShoulder", "LeftArm"), ("LeftArm", "LeftForeArm"), ("LeftForeArm", "LeftHand"),
            ("Spine2", "RightShoulder"), ("RightShoulder", "RightArm"), ("RightArm", "RightForeArm"), ("RightForeArm", "RightHand"),
            ("Hips", "LeftUpLeg"), ("LeftUpLeg", "LeftLeg"), ("LeftLeg", "LeftFoot"), ("LeftFoot", "LeftToeBase"),
            ("Hips", "RightUpLeg"), ("RightUpLeg", "RightLeg"), ("RightLeg", "RightFoot"), ("RightFoot", "RightToeBase")
        ]

        self.init_ui()

    def init_ui(self):
        menubar = self.menuBar()
        file_menu = menubar.addMenu("Fichier")
        file_menu.addAction(QAction("Ouvrir Fichier MKF...", self, triggered=self.load_mkf))

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        
        self.splitter = QSplitter(Qt.Orientation.Horizontal)
        main_layout.addWidget(self.splitter)

        # --- PANNEAU GAUCHE : 3D & LECTEUR ---
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        
        self.view_3d = CustomGLViewWidget()
        self.view_3d.opts['distance'] = 4 
        self.view_3d.opts['elevation'] = 20
        self.grid = gl.GLGridItem()
        self.view_3d.addItem(self.grid)
        self.skeleton_lines = gl.GLLinePlotItem(color=pg.mkColor(255, 200, 50, 255), width=3.0, antialias=True)
        self.view_3d.addItem(self.skeleton_lines)
        
        left_layout.addWidget(self.view_3d, stretch=1)
        
        # LECTEUR PRECISION
        player_layout = QHBoxLayout()
        
        for offset in [-10, -1]:
            btn = QPushButton(str(offset))
            btn.setFixedWidth(35)
            btn.clicked.connect(lambda _, o=offset: self.set_frame(self.current_frame + o))
            player_layout.addWidget(btn)
            
        self.btn_play = QPushButton("▶ / ⏸")
        self.btn_play.setFixedWidth(50)
        self.btn_play.clicked.connect(self.toggle_play)
        player_layout.addWidget(self.btn_play)
        
        for offset in [1, 10]:
            btn = QPushButton(f"+{offset}")
            btn.setFixedWidth(35)
            btn.clicked.connect(lambda _, o=offset: self.set_frame(self.current_frame + o))
            player_layout.addWidget(btn)
        
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.sliderMoved.connect(self.set_frame)
        player_layout.addWidget(self.slider)
        
        self.lbl_frame = QLabel("0 / 0")
        player_layout.addWidget(self.lbl_frame)
        left_layout.addLayout(player_layout)
        
        self.splitter.addWidget(left_panel)

        # --- PANNEAU DROIT : GRAPHIQUES ET CORRÉLATION ---
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)

        # Les 3 Graphes Demandés
        self.plot_1 = pg.PlotWidget(title="Graphe 1 : Hauteur et Vitesse (Pointe Z) - Valeurs Normalisées")
        self.plot_1.addLegend()
        
        self.plot_2 = pg.PlotWidget(title="Graphe 2 : Angle Cheville - Valeurs Normalisées")
        self.plot_2.addLegend()
        
        self.plot_3 = pg.PlotWidget(title="Graphe 3 : Jerk (Pointe Z) - Valeurs Normalisées")
        self.plot_3.addLegend()
        
        # Lier les axes X pour zoomer/naviguer ensemble
        self.plot_2.setXLink(self.plot_1)
        self.plot_3.setXLink(self.plot_1)
        
        # --- LIGNES FINES (width=1) ---
        
        # Courbes Graphe 1 (Hauteur et Vitesse)
        self.curve_h_toe_l = self.plot_1.plot(pen=pg.mkPen('c', width=1), name="Hauteur G")
        self.curve_h_toe_r = self.plot_1.plot(pen=pg.mkPen('m', width=1), name="Hauteur D")
        
        self.curve_v_toe_l = self.plot_1.plot(pen=pg.mkPen((0, 100, 255), width=1, style=Qt.PenStyle.DotLine), name="Vitesse G")
        self.curve_v_toe_r = self.plot_1.plot(pen=pg.mkPen((255, 50, 100), width=1, style=Qt.PenStyle.DotLine), name="Vitesse D")
        
        # Courbes Graphe 2 (Angle Cheville)
        self.curve_ang_l = self.plot_2.plot(pen=pg.mkPen('c', width=1), name="Angle G")
        self.curve_ang_r = self.plot_2.plot(pen=pg.mkPen('m', width=1), name="Angle D")
        
        # Courbes Graphe 3 (Jerk)
        self.curve_j_toe_l = self.plot_3.plot(pen=pg.mkPen('c', width=1), name="Jerk G")
        self.curve_j_toe_r = self.plot_3.plot(pen=pg.mkPen('m', width=1), name="Jerk D")
        
        # Curseurs verticaux (barre de lecture synchronisée)
        self.cursor_1 = pg.InfiniteLine(angle=90, movable=False, pen=pg.mkPen('y', width=2))
        self.cursor_2 = pg.InfiniteLine(angle=90, movable=False, pen=pg.mkPen('y', width=2))
        self.cursor_3 = pg.InfiniteLine(angle=90, movable=False, pen=pg.mkPen('y', width=2))
        self.plot_1.addItem(self.cursor_1)
        self.plot_2.addItem(self.cursor_2)
        self.plot_3.addItem(self.cursor_3)

        right_layout.addWidget(self.plot_1)
        right_layout.addWidget(self.plot_2)
        right_layout.addWidget(self.plot_3)

        # Panneau Corrélation
        corr_group = QGroupBox("Analyse des Corrélations Statistiques")
        corr_layout = QVBoxLayout(corr_group)
        
        btn_calc = QPushButton("📊 Calculer la Corrélation entre les Descripteurs")
        btn_calc.setStyleSheet("background-color: #2980b9; color: white; font-weight: bold; padding: 10px;")
        btn_calc.clicked.connect(self.calculate_correlations)
        corr_layout.addWidget(btn_calc)
        
        self.lbl_results = QLabel("En attente de calcul...")
        self.lbl_results.setWordWrap(True)
        self.lbl_results.setStyleSheet("font-family: monospace; font-size: 14px;")
        
        # Utilisation d'un QScrollArea au cas où la matrice est grosse
        scroll_res = QScrollArea()
        scroll_res.setWidgetResizable(True)
        scroll_res.setWidget(self.lbl_results)
        scroll_res.setMaximumHeight(150)
        
        corr_layout.addWidget(scroll_res)

        right_layout.addWidget(corr_group)
        
        self.splitter.addWidget(right_panel)
        self.splitter.setSizes([400, 800])

        self.timer = QTimer()
        self.timer.timeout.connect(self.next_frame)

    def load_mkf(self):
        path, _ = QFileDialog.getOpenFileName(self, "Ouvrir fichier MKF", "", "MKF Files (*.mkf)")
        if not path: return
        
        try:
            with open(path, 'r') as f:
                lines = f.readlines()
            header = lines[0].strip().split()
            frames, joints = int(header[0]), int(header[1])
            self.mkf_joint_names = header[2:]
            
            raw_mkf = [[float(x) for x in line.strip().split()] for line in lines[1:] if line.strip()]
            self.mkf_data = np.array(raw_mkf).reshape(frames, joints, 6)[:, :, :3] / 1000.0
            self.max_frames = frames
            
            self.slider.setRange(0, self.max_frames - 1)
            self.lbl_frame.setText(f"0 / {self.max_frames - 1}")
            
            self.extractor = FeatureExtractor(self.mkf_data, self.mkf_joint_names, self.freq)
            self.update_plots()
            self.set_frame(0)
            
            self.lbl_results.setText("Données chargées. Analysez visuellement les courbes ou lancez le calcul de corrélation.")
            
        except Exception as e:
            QMessageBox.critical(self, "Erreur", f"Impossible de charger : {e}")

    def update_plots(self):
        time_axis = np.arange(self.max_frames) / self.freq
        
        X_left, _ = self.extractor.extract_features("Left")
        X_right, _ = self.extractor.extract_features("Right")
        
        # Fonction de Normalisation Min-Max (Ramène les valeurs entre 0.0 et 1.0)
        def normalize(arr):
            ptp = np.ptp(arr)
            return (arr - np.min(arr)) / ptp if ptp != 0 else arr
        
        # Graphe 1 (Hauteur et Vitesse)
        self.curve_h_toe_l.setData(time_axis, normalize(X_left[:, 0]))
        self.curve_h_toe_r.setData(time_axis, normalize(X_right[:, 0]))
        
        self.curve_v_toe_l.setData(time_axis, normalize(X_left[:, 1]))
        self.curve_v_toe_r.setData(time_axis, normalize(X_right[:, 1]))
        
        # Graphe 2 (Angle Cheville)
        self.curve_ang_l.setData(time_axis, normalize(X_left[:, 2]))
        self.curve_ang_r.setData(time_axis, normalize(X_right[:, 2]))
        
        # Graphe 3 (Jerk)
        self.curve_j_toe_l.setData(time_axis, normalize(X_left[:, 3]))
        self.curve_j_toe_r.setData(time_axis, normalize(X_right[:, 3]))

    def calculate_correlations(self):
        if self.mkf_data is None: return
        
        X_left, feature_names = self.extractor.extract_features("Left")
        
        # Calcul de la matrice de corrélation (Pearson)
        corr_matrix = np.corrcoef(X_left, rowvar=False)
        
        # Construction d'un tableau HTML propre pour afficher les corrélations
        html = "<table width='100%' border='1' cellspacing='0' cellpadding='5' style='border-collapse: collapse; text-align: center;'>"
        
        # En-tête
        html += "<tr style='background-color: #3b4252; color: white;'><th></th>"
        for name in feature_names:
            html += f"<th>{name}</th>"
        html += "</tr>"
        
        # Lignes
        for i, row_name in enumerate(feature_names):
            html += f"<tr><td style='background-color: #3b4252; color: white; font-weight: bold;'>{row_name}</td>"
            for j in range(len(feature_names)):
                val = corr_matrix[i, j]
                # Code couleur selon la force de la corrélation
                if val >= 0.7 or val <= -0.7:
                    color = "#27ae60" if val > 0 else "#c0392b"
                elif val >= 0.3 or val <= -0.3:
                    color = "#f39c12"
                else:
                    color = "black"
                
                html += f"<td style='color: {color}; font-weight: bold;'>{val:.2f}</td>"
            html += "</tr>"
        
        html += "</table><br>"
        html += "<i>Vert/Rouge = Forte corrélation (+/-) | Orange = Moyenne | Noir = Faible. <br>Exemple : Un Jerk fortement corrélé à la Hauteur indique un lien mécanique direct lors de la frappe.</i>"
        
        self.lbl_results.setText(html)

    def toggle_play(self):
        if self.is_playing:
            self.is_playing = False
            self.timer.stop()
        else:
            self.is_playing = True
            self.timer.start(30)

    def set_frame(self, frame):
        if self.max_frames == 0: return
        frame = max(0, min(frame, self.max_frames - 1))
        self.current_frame = frame
        
        self.slider.setValue(frame)
        self.lbl_frame.setText(f"{frame} / {self.max_frames - 1}")
        
        time_sec = frame / self.freq
        self.cursor_1.setValue(time_sec)
        self.cursor_2.setValue(time_sec)
        self.cursor_3.setValue(time_sec)
        
        self.draw_skeleton(frame)

    def next_frame(self):
        step = 1 # Play frame by frame
        if self.current_frame + step < self.max_frames:
            self.set_frame(self.current_frame + step)
        else:
            self.set_frame(0)

    def draw_skeleton(self, frame):
        lines_pos = []
        for parent, child in self.bones:
            if parent in self.mkf_joint_names and child in self.mkf_joint_names:
                p_idx = self.mkf_joint_names.index(parent)
                c_idx = self.mkf_joint_names.index(child)
                lines_pos.append(self.mkf_data[frame, p_idx])
                lines_pos.append(self.mkf_data[frame, c_idx])

        if lines_pos:
            self.skeleton_lines.setData(pos=np.array(lines_pos, dtype=np.float32), mode='lines')

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = AnnotationApp()
    window.showMaximized() 
    sys.exit(app.exec())
