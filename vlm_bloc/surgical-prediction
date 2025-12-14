import cv2
import torch
import gc
import json
from PIL import Image
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, get_model_name_from_path

# --- CONFIGURATION ---
MODEL_PATH = "checkpoints/llava-fastvithd_0.5b_stage3"
PROCESS_EVERY_N_FRAMES = 15

# Le Prompt est crucial ici. On impose un format strict.
# On donne des exemples pour guider le petit modèle.
SYSTEM_PROMPT = """You are a surgical AI assistant. Analyze the image to identify the current surgical step.
Based on the current action, predict the most likely NEXT surgical tool needed.
Return the answer in this exact format:
ACTION: [current action description] | NEXT_TOOL: [name of the tool]"""
# ---------------------

def parse_model_output(output_text):
    """
    Fonction pour nettoyer et formater la réponse brute du modèle.
    Transforme "ACTION: Incision | NEXT_TOOL: Scalpel" en dictionnaire Python.
    """
    data = {"action": "Analyzing...", "next_tool": "Wait..."}
    
    # Nettoyage basique
    text = output_text.replace("USER:", "").replace("ASSISTANT:", "").strip()
    
    try:
        if "|" in text:
            parts = text.split("|")
            # On cherche la partie ACTION
            for part in parts:
                if "ACTION:" in part:
                    data["action"] = part.replace("ACTION:", "").strip()
                if "NEXT_TOOL:" in part:
                    data["next_tool"] = part.replace("NEXT_TOOL:", "").strip()
        else:
            # Fallback si le modèle oublie le séparateur
            data["action"] = text
            data["next_tool"] = "Unknown"
            
    except Exception as e:
        print(f"Parsing error: {e}")
        
    return data

def main():
    disable_torch_init()

    print(f"Chargement du modèle chirurgical depuis {MODEL_PATH}...")
    model_name = get_model_name_from_path(MODEL_PATH)
    
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        MODEL_PATH, None, model_name, device="cuda"
    )
    print("Modèle prêt ! Simulation chirurgicale active.")

    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("ERREUR: Caméra non détectée !")
        return

    frame_count = 0
    # Valeurs par défaut
    current_data = {"action": "Initializing", "next_tool": "..."}

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % PROCESS_EVERY_N_FRAMES == 0:
            try:
                # Préparation Image
                image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(image_rgb)
                image_tensor = image_processor.preprocess(pil_image, return_tensors='pt')['pixel_values'].half().cuda()

                # Construction du Prompt
                # On combine le Token Image + Notre consigne stricte
                raw_prompt = f"USER: {DEFAULT_IMAGE_TOKEN}\n{SYSTEM_PROMPT}\nASSISTANT:"

                input_ids = tokenizer_image_token(raw_prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
                attention_mask = torch.ones_like(input_ids, device="cuda")

                with torch.inference_mode():
                    output_ids = model.generate(
                        input_ids,
                        images=image_tensor,
                        attention_mask=attention_mask,
                        pad_token_id=tokenizer.eos_token_id,
                        do_sample=False,    # Déterministe pour respecter le format
                        num_beams=1,
                        max_new_tokens=40,  # Court pour forcer la concision
                        use_cache=True
                    )

                raw_output = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
                print(f"RAW AI: {raw_output}") # Pour débugger dans la console
                
                # Parsing vers format fixe
                current_data = parse_model_output(raw_output)

                # Nettoyage VRAM
                del image_tensor, input_ids, output_ids, attention_mask
                torch.cuda.empty_cache()

            except Exception as e:
                print(f"Erreur inférence: {e}")

        # --- Affichage Interface Chirurgicale ---
        height, width, _ = frame.shape
        
        # Bandeau Haut (Action en cours)
        cv2.rectangle(frame, (0, 0), (width, 60), (50, 50, 50), -1)
        cv2.putText(frame, f"CURRENT: {current_data['action']}", (20, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)

        # Bandeau Bas (NEXT TOOL - En évidence)
        cv2.rectangle(frame, (0, height-80), (width, height), (0, 0, 150), -1) # Fond rouge sombre
        cv2.putText(frame, f"PREDICTION: {current_data['next_tool'].upper()}", (20, height-30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 3)

        cv2.imshow('Surgical Assistant AI', frame)
        frame_count += 1

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()