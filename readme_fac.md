Barrault Ryan

barraultryan@hotmail.fr

# Application de calculs de descripteurs (et classification dans le futur)

Cette application permet la visualisation 3D de données de Motion Capture  l'nalyse biomécanique de descripteurs. Elle permet d'extraire, de filtrer et d'afficher des descripteurs de mouvement bas niveau et haut niveau pour étudier les caractéristiques de la marche et le style d'un acteur.

---

## Architecture du répertoire de Données

Pour que l'explorateur de fichiers de l'application fonctionne correctement et associe automatiquement les données brutes aux squelettes, le répertoire source (sélectionné via `Fichier > Ouvrir le répertoire 'Data'...`) doit respecter une arborescence stricte :

```text
Data/
├── TSV/                  # Données des marqueurs
│   ├── sujet1_marche.tsv
│   └── sujet2_course.tsv
└── MKF/                  # Données des jointsq
    ├── sujet1_marche.mkf
    └── sujet2_course.mkf
```

L'application parse les noms de fichiers. Si un fichier TSV et un fichier MKF partagent le même suffixe/identifiant, ils sont regroupés dans la catégorie "Paires(TSV + MKF)". Les fichiers qui ne trouvent pas leur pair sont classés dans "Fichiers Isolés".

## Interface et Fonctionnement

1. Explorateur de Données (Panneau Gauche):

QTreeWidget : Affiche l'arborescence des fichiers TSV/MKF détectés. Un clic sur une paire charge instantanément les données , calcule les descripteurs et met à jour l'affichage.

2. Viewer 3D et Lecteur (Panneau Central):

- Vue 3D: Affiche les marqueurs et les segments du squelette.
- Boutons Caméra: Face, Dessus, Coté, Iso libre permettent de changer de vue.

- Raccourcis Clavier:
    - M: Affiche/Masque les marqueurs TSV.
    - S: Affiche/Masque le squelette MKF.
    - F: Affiche/Masque un sol rigie.

- Contrôles de Lecture:
    -  ▶ / ⏸: Lance ou met en pause l'animation.
    - Vitesse (x0.16, x1, x2) : Modifie la vitesse de lecture.
    - Boutons de saut (-10, -1, +1, +10) et Slider : Navigation dans la séquence.

3. Panneau d'Analyse (Panneau Droit):

Il y a une partie avec les valeurs numériques et une partie avec les séries temporelles(graphiques)

Analyse Biomécanique : Les Descripteurs
Le calcul des descripteurs utilise scipy.signal pour lisser les données brutes avant de dériver.

```py
def lisser_donnees(self, vecteur, window_length=15, polyorder=3):
    # filtre Savitzky-Golay pour éliminer le bruit haute fréquence
    return savgol_filter(vecteur, window_length, polyorder, axis=0)
```

Les descripteurs sont divisés en deux catégories :

- Les scalaires affichés dans le tableau de valeurs en haut à droite 

    - Spatiotemporel : Cadence (pas/min), Fréquence (Hz), Durée moyenne et Longueur du pas.

    - Stabilité & Symétrie : Oscillation latérale et verticale du centre de masse (COM), index de symétrie (SI) des genoux ou de la vitesse des pieds.

    Exemple:
    ```py
    def compute_symmetry_index(self, val_left, val_right):
        if (val_left + val_right) == 0:
            return 0
        # pourcentage d'assymétrie entre G et D
        return (np.abs(val_left - val_right) / (0.5 * (val_left + val_right))) * 100`
    ```

- Les Séries Temporelles :
Les graphiques avec des calculs sur toute les frames de la séquence.

    - Cinématique : Vitesse et Jerk de chaque pied.

    - Angles : Flexion du genou, de la cheville, et l'angle d'attaque du talon (Heel Strike Angle).

    - Détection automatique des phases de contact au sol (0 = vol, 1 = appui).

Exemple du Jerk:
```py
def calculer_vecteur_jerk_3D_joint(self, nom_joint):
    accel = self.calculer_vecteur_acceleration_3D_joint(nom_joint)
    # Dériv de l'accél.
    jerk = np.vstack([np.zeros((1, 3)), np.diff(accel, axis=0)]) / self.dt
    return jerk / 1000.0 # m/s³
```


## Évolution Future : Classification par Réseau de Neurones
Le menu Réseaux de Neurones est actuellement en préparation pour faire la partie classification dont on avait parlé
= prédire exactement 9 classes pour 9 acteurs...

Pour l'instant J'ai pas encore trop bossé dessus mais je reste sur mon idée de Triplet Loss qui prend 3 fichiers dont 2 proches d'un même acteur et le dernier d'un autre acteur et qui fait en sorte de les représenter dans un espace 3D, en rapprochant au maximum les deux fichiers qui se ressemblent et en éloignant au maximum le dernier qui est différent.