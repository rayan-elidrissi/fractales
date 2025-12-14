import cv2
import torch
import gc
from pathlib import Path
import sys
from PIL import Image

# Ensure llava package is importable when running directly from this script
PROJECT_ROOT = Path(__file__).resolve().parents[2]
LLAVA_PACKAGE_DIR = PROJECT_ROOT / "ml-fastvlm"
if LLAVA_PACKAGE_DIR.exists():
    sys.path.insert(0, str(LLAVA_PACKAGE_DIR))

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, get_model_name_from_path

# --- CONFIGURATION ---
MODEL_PATH = "checkpoints/llava-fastvithd_0.5b_stage3"
# On change le prompt pour être très directif
PROMPT_TEXT = "Describe this image details."
PROCESS_EVERY_N_FRAMES = 120
# ---------------------

def main():
    disable_torch_init()

    print(f"Chargement du modèle depuis {MODEL_PATH}...")
    model_name = get_model_name_from_path(MODEL_PATH)
    
    # Chargement
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        MODEL_PATH, None, model_name, device="cuda"
    )
    print("Modèle chargé ! Démarrage...")

    cap = cv2.VideoCapture(0)
    
    # Petite vérification que la caméra marche
    if not cap.isOpened():
        print("ERREUR: Caméra non détectée !")
        return

    frame_count = 0
    current_description = "Waiting..."

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % PROCESS_EVERY_N_FRAMES == 0:
            try:
                # 1. Conversion propre BGR (OpenCV) -> RGB (PIL)
                image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(image_rgb)
                
                # 2. Prétraitement
                image_tensor = image_processor.preprocess(pil_image, return_tensors='pt')['pixel_values'].half().cuda()

                # 3. CONSTRUCTION MANUELLE DU PROMPT (Le secret est ici)
                # On évite 'conv_templates' qui ajoute trop de blabla système.
                # Format standard LLaVA : USER: <image>\nQuestion ASSISTANT:
                raw_prompt = f"USER: {DEFAULT_IMAGE_TOKEN}\n{PROMPT_TEXT}\nASSISTANT:"

                # 4. Tokenization
                input_ids = tokenizer_image_token(raw_prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
                
                # Création manuelle du masque d'attention pour éviter les bugs
                attention_mask = torch.ones_like(input_ids, device="cuda")

                # 5. Génération
                with torch.inference_mode():
                    output_ids = model.generate(
                        input_ids,
                        images=image_tensor,
                        attention_mask=attention_mask,
                        pad_token_id=tokenizer.eos_token_id,
                        do_sample=False,    # False = plus déterministe/moins d'hallucination
                        num_beams=1,        # Greedy search (plus rapide)
                        max_new_tokens=40,
                        use_cache=True
                    )

                # 6. Décodage
                outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
                
                # Si l'IA hallucine encore des balises, on nettoie
                if "USER:" in outputs: outputs = outputs.split("USER:")[0]
                
                current_description = outputs
                print(f"IA: {current_description}")

                # 7. Ménage mémoire
                del image_tensor, input_ids, output_ids, attention_mask
                torch.cuda.empty_cache()

            except Exception as e:
                print(f"Erreur: {e}")

        # --- Affichage Visuel ---
        # Bandeau noir
        cv2.rectangle(frame, (0, 0), (frame.shape[1], 60), (0, 0, 0), -1)
        # Texte blanc
        cv2.putText(frame, current_description, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        cv2.imshow('FastVLM Live', frame)
        frame_count += 1

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()