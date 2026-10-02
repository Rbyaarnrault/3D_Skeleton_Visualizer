import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog

def lire_mkf(filepath):
    """Parse un fichier MKF et retourne les données brutes et les noms des joints."""
    try:
        with open(filepath, 'r') as f:
            lines = f.readlines()
            
        header = lines[0].strip().split()
        frames = int(header[0])
        joints = int(header[1])
        joint_names = header[2:]
        
        raw_mkf = []
        for line in lines[1:]:
            parts = line.strip().split()
            if parts:
                raw_mkf.append([float(x) for x in parts])
                
        # positions 3D (X, Y, Z) et conversion en mm
        mkf_data = np.array(raw_mkf).reshape(frames, joints, 6)[:, :, :3]
        return mkf_data, joint_names
    except Exception as e:
        print(f"Erreur lors de la lecture du fichier : {e}")
        return None, None

def calculer_et_afficher_angle(filepath):
    data, names = lire_mkf(filepath)
    if data is None:
        return
    try:
        idx_spine2 = names.index("Spine2")
        idx_neck = names.index("Neck")
        idx_headtop = names.index("HeadTop")
    except ValueError:
        print("Erreur : Les joints Spine2, Neck ou HeadTop sont introuvables dans le fichier.")
        return

    pos_spine2 = data[:, idx_spine2, :]
    pos_neck = data[:, idx_neck, :]
    pos_headtop = data[:, idx_headtop, :]

    # Spine2 -> Neck)
    v1 = pos_neck - pos_spine2
    # Neck -> HeadTop)
    v2 = pos_headtop - pos_neck

    angles_deg = np.zeros(len(data))
    
    for i in range(len(data)):
        # Produit scalaire
        dot_product = np.dot(v1[i], v2[i])
        
        # Normes
        norm_v1 = np.linalg.norm(v1[i])
        norm_v2 = np.linalg.norm(v2[i])
        
        # Sauf div par 0
        if norm_v1 == 0 or norm_v2 == 0:
            angles_deg[i] = 0.0
            continue
            
        cos_theta = dot_product / (norm_v1 * norm_v2)
        angle_rad = np.arccos(cos_theta)
        angles_deg[i] = np.degrees(angle_rad)

    #Plot 1D
    time_axis = np.arange(len(angles_deg)) / 200.0  # en sec

    plt.figure(figsize=(10, 5))
    plt.plot(time_axis, angles_deg, color='b', linewidth=1.5)
    plt.title(f"Angle de flexion du cou (Spine2 -> Neck -> HeadTop)\nFichier: {filepath.split('/')[-1]}")
    plt.xlabel("Temps (secondes)")
    plt.ylabel("Angle (Degrés)")
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Ouvre une fenêtre de dialogue simple pour choisir un fichier MKF
    root = tk.Tk()
    root.withdraw() 
    fichier_mkf = filedialog.askopenfilename(
        title="Sélectionner un fichier MKF",
        filetypes=[("Mocap MKF Files", "*.mkf")]
    )
    
    if fichier_mkf:
        calculer_et_afficher_angle(fichier_mkf)