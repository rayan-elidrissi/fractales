import cv2
import torch
import gc
from PIL import Image
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, get_model_name_from_path

# --- CONFIGURATION ---
MODEL_PATH = "checkpoints/llava-fastvithd_0.5b_stage3"
PROCESS_EVERY_N_FRAMES = 15

# Le Prompt est simplifié à l'extrême pour forcer une réponse binaire.
PROMPT_TEXT = "Is the mouth in this image OPEN or CLOSED? Answer with a single word."
# ---------------------

def get_tool_logic(ai_output):
    """
    Applique la règle métier stricte demandée :
    - Fermée -> YELLOW-MIRROR
    - Ouverte -> BLUE-TWEEZER
    """
    text = ai_output.upper()
    
    if "OPEN" in text:
        return {
            "state": "BOUCHE OUVERTE",
            "tool": "BLUE-TWEEZER",
            "color": (255, 0, 0) # Bleu en BGR pour OpenCV
        }
    elif "CLOSED" in text:
        return {
            "state": "BOUCHE FERMEE",
            "tool": "YELLOW-MIRROR",
            "color": (0, 255, 255) # Jaune en BGR (Cyan/Jaune selon mix) -> Jaune = (0, 255, 255)
        }
    else:
        return {
            "state": "ANALYSE...",
            "tool": "WAITING...",
            "color": (200, 200, 200) # Gris
        }

def main():
    disable_torch_init()

    print(f"Chargement du modèle depuis {MODEL_PATH}...")
    model_name = get_model_name_from_path(MODEL_PATH)
    
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        MODEL_PATH, None, model_name, device="cuda"
    )
    print("Modèle prêt ! Approchez votre visage de la caméra.")

    # Essayez l'index qui a répondu "Connecté" à l'étape 1 (souvent 0 ou 1)
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    # 1. On force une résolution standard IMMÉDIATEMENT après l'ouverture
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    # 2. On laisse la caméra "chauffer" (les 1ères frames sont souvent noires)
    for i in range(10):
        cap.read()

    if not cap.isOpened():
        print("Erreur fatale : Caméra non accessible.")
        exit()
    
    if not cap.isOpened():
        print("ERREUR: Caméra non détectée !")
        return

    frame_count = 0
    current_data = {"state": "INIT", "tool": "...", "color": (200, 200, 200)}

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

                # Construction du Prompt Direct
                raw_prompt = f"USER: {DEFAULT_IMAGE_TOKEN}\n{PROMPT_TEXT}\nASSISTANT:"

                input_ids = tokenizer_image_token(raw_prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
                attention_mask = torch.ones_like(input_ids, device="cuda")

                with torch.inference_mode():
                    output_ids = model.generate(
                        input_ids,
                        images=image_tensor,
                        attention_mask=attention_mask,
                        pad_token_id=tokenizer.eos_token_id,
                        do_sample=False,    # Déterministe
                        num_beams=1,
                        max_new_tokens=10,  # Très court (juste un mot)
                        use_cache=True
                    )

                raw_output = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
                print(f"IA voit : {raw_output}")
                
                # Application de la logique métier
                current_data = get_tool_logic(raw_output)

                # Nettoyage
                del image_tensor, input_ids, output_ids, attention_mask
                torch.cuda.empty_cache()

            except Exception as e:
                print(f"Erreur: {e}")

        # --- Affichage Visuel ---
        height, width, _ = frame.shape
        
        # 1. État de la bouche (Haut)
        cv2.rectangle(frame, (0, 0), (width, 50), (30, 30, 30), -1)
        cv2.putText(frame, f"ETAT: {current_data['state']}", (20, 35), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        # 2. Outil Recommandé (Bas - En gros et en couleur)
        # Fond sombre pour contraste
        cv2.rectangle(frame, (0, height-80), (width, height), (20, 20, 20), -1) 
        
        # Texte avec la couleur dynamique (Jaune ou Bleu)
        tool_text = f"OUTIL: {current_data['tool']}"
        text_size = cv2.getTextSize(tool_text, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 3)[0]
        text_x = (width - text_size[0]) // 2 # Centrer le texte
        
        cv2.putText(frame, tool_text, (text_x, height-25), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, current_data['color'], 3)

        cv2.imshow('Dental Assistant AI', frame)
        frame_count += 1

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()