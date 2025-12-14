import cv2

print("Scan des caméras en cours...")

# On teste les index de 0 à 3
for index in range(4):
    print(f"--- Test de l'index {index} ---")
    # On tente d'ouvrir avec DirectShow
    cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
    
    if not cap.isOpened():
        print(f"Index {index} : Impossible d'ouvrir (Port vide)")
    else:
        # On essaie de lire une image
        ret, frame = cap.read()
        if ret:
            # On vérifie si l'image est noire
            if frame.sum() == 0:
                print(f"Index {index} : CONNECTÉ mais IMAGE NOIRE (Problème config ou cache objectif)")
            else:
                print(f"✅ Index {index} : FONCTIONNE ! (Résolution: {frame.shape[1]}x{frame.shape[0]})")
                # Sauvegarde une photo pour preuve
                cv2.imwrite(f"test_cam_{index}.png", frame)
        else:
            print(f"Index {index} : Ouvert, mais impossible de lire une frame (Erreur lecture)")
    
    cap.release()

print("Scan terminé.")
