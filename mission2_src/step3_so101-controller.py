import cv2
import torch
import gc
import time
import subprocess  # Pour lancer les commandes terminal
import shlex       # Pour sécuriser les commandes
from PIL import Image
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, get_model_name_from_path

# --- CONFIGURATION UTILISATEUR ---
MODEL_PATH = "checkpoints/llava-fastvithd_0.5b_stage3"

# ⏱️ INTERVALLE DE TEMPS (15 secondes)
ROBOT_INTERVAL = 15

# 🤖 CONFIGURATION LEROBOT
# Remplacez par vos vrais chemins de modèles entraînés (HuggingFace ou local)
POLICY_MIRROR_PATH = "user/act_yellow_mirror_policy" 
POLICY_TWEEZER_PATH = "user/act_blue_tweezer_policy"

# Arguments génériques pour LeRobot (à adapter selon votre robot, ex: so100_follower)
ROBOT_TYPE = "so100_follower" 
NUM_EPISODES = 1 # On fait 1 action par déclenchement

# ---------------------------------

def run_lerobot_policy(policy_path, tool_name):
    """
    Lance la commande lerobot-record pour exécuter la politique sur le robot réel.
    Cette fonction est 'bloquante' : le script attend que le robot finisse.
    """
    print(f"\n[ROBOT] 🚀 Démarrage de la séquence : {tool_name}")
    print(f"[ROBOT] Chargement de la policy : {policy_path}")

    # Construction de la commande exacte fournie dans votre documentation
    # Note : On utilise 'eval_manual' pour le dataset car on fait de l'inférence
    command = (
        f"lerobot-record "
        f"--robot.type={ROBOT_TYPE} "
        f"--dataset.repo_id=user/eval_{tool_name}_session "
        f"--policy.path={policy_path} "
        f"--episodes={NUM_EPISODES}"
    )

    try:
        # Exécution de la commande
        # shell=True est nécessaire sur Windows pour bien gérer les environnements Conda
        subprocess.run(command, shell=True, check=True)
        print(f"[ROBOT] ✅ Séquence {tool_name} terminée.\n")
    except subprocess.CalledProcessError as e:
        print(f"[ROBOT] ❌ Erreur critique lors de l'exécution du robot : {e}\n")

def get_mouth_state(ai_output):
    """Analyse la sortie texte de l'IA"""
    text = ai_output.upper()
    if "OPEN" in text:
        return "OPEN"
    elif "CLOSED" in text:
        return "CLOSED"
    return "UNKNOWN"

def main():
    disable_torch_init()
    print(f"Chargement du modèle Vision depuis {MODEL_PATH}...")
    
    model_name = get_model_name_from_path(MODEL_PATH)
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        MODEL_PATH, None, model_name, device="cuda"
    )
    print("Système prêt ! Timer de 15s enclenché.")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERREUR: Caméra non détectée !")
        return

    # Initialisation du Timer
    last_trigger_time = time.time()
    
    # Prompt simple
    PROMPT_TEXT = "Is the mouth in this image OPEN or CLOSED? Answer with a single word."

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # --- LOGIQUE TEMPORELLE ---
        current_time = time.time()
        time_elapsed = current_time - last_trigger_time
        time_remaining = max(0, ROBOT_INTERVAL - time_elapsed)

        # On affiche le compte à rebours sur l'image
        cv2.rectangle(frame, (0, 0), (frame.shape[1], 80), (30, 30, 30), -1)
        
        # Si le temps est écoulé, on lance l'analyse ET l'action robot
        if time_elapsed > ROBOT_INTERVAL:
            cv2.putText(frame, "ANALYZING & TRIGGERING ROBOT...", (10, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            cv2.imshow('Surgical Controller', frame)
            cv2.waitKey(1) # Force la mise à jour de l'écran avant le gel du robot

            try:
                # 1. Vision (FastVLM)
                image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(image_rgb)
                image_tensor = image_processor.preprocess(pil_image, return_tensors='pt')['pixel_values'].half().cuda()
                
                raw_prompt = f"USER: {DEFAULT_IMAGE_TOKEN}\n{PROMPT_TEXT}\nASSISTANT:"
                input_ids = tokenizer_image_token(raw_prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
                attention_mask = torch.ones_like(input_ids, device="cuda")

                with torch.inference_mode():
                    output_ids = model.generate(
                        input_ids,
                        images=image_tensor,
                        attention_mask=attention_mask,
                        pad_token_id=tokenizer.eos_token_id,
                        do_sample=False, 
                        num_beams=1,
                        max_new_tokens=10,
                        use_cache=True
                    )
                
                result = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
                state = get_mouth_state(result)
                print(f"[VISION] État détecté : {state}")

                # 2. Action Robot (LeRobot)
                if state == "CLOSED":
                    # Lancer Policy YELLOW-MIRROR
                    run_lerobot_policy(POLICY_MIRROR_PATH, "YELLOW-MIRROR")
                
                elif state == "OPEN":
                    # Lancer Policy BLUE-TWEEZER
                    run_lerobot_policy(POLICY_TWEEZER_PATH, "BLUE-TWEEZER")
                
                else:
                    print("[VISION] État incertain, pas d'action robot.")

                # Nettoyage
                del image_tensor, input_ids, output_ids, attention_mask
                torch.cuda.empty_cache()

                # Reset du timer
                last_trigger_time = time.time()

            except Exception as e:
                print(f"Erreur cycle : {e}")

        else:
            # Mode attente (Affichage du chrono)
            cv2.putText(frame, f"NEXT SCAN IN: {int(time_remaining)}s", (10, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow('Surgical Controller', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()