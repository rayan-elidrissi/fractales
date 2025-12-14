# AMD_Robotics_Hackathon_2025_Fractales

## Team Information
- **Team:** 9
- **Team Name:** Fractales
- **Team Members:** Abdalraouf BENYZID, Ryu OSADA, Rayan EL IDRISSI DAFALI

## Summary
We demonstrate a robotic dental assistant using an S0-101 arm and vision-based reasoning to retrieve and hand over medical tools. The system picks up a mirror or tweezers from both organized and unorganized tables and performs a safe handover only when a gloved dentist’s hand is detected.

---

## Submission Details

### 1. Mission Description
The mission of this project is to explore how a robotic assistant can support a dentist during routine check-up procedures by autonomously understanding requests, perceiving the environment, and safely handing over medical tools.

In a real-world dental setting, dentists frequently need to switch tools while maintaining focus on the patient and maintaining sterile conditions. Our robotic assistant acts as a proof-of-concept dental assistant that can respond to direct voice requests, observe the clinical workspace, and deliver the appropriate instrument in a controlled and safe manner. The dentist can explicitly ask for a specific tool (example : “give me the mirror”), request a tool to be returned (“I don’t need this anymore”), or remain in a standby state while the assistant holds an instrument.

Beyond explicit commands, the assistant is designed to reason about context and workflow. By observing the ongoing procedure and the dentist’s actions, the system can potentially suggest or predict the next tool that may be needed, enabling a more proactive form of assistance rather than purely reactive behavior. 

The robot assistant uses vision to monitor the patient area, the tool table, and the dentist’s hand. Once a tool is retrieved, the robot waits in a safe standby position and only performs a handover when the dentist’s gloved hand is detected in a predefined region. If safety conditions are not met, the robot will not proceed, prioritizing hygiene and controlled interaction.

The system is also designed with emergency awareness in mind. In unexpected situations such as sudden movements, loss of visual tracking, or ambiguous handover signals. The robot can pause, hold position, or abort the handover entirely, allowing the dentist to retain full control of the situation.

While the current implementation focuses on a dental assistant scenario, the underlying logic is intentionally general. The same perception, reasoning, and manipulation framework can be transferred to other domains that involve human robot collaboration, such as medical assistance in other specialties, mechanical workshops, laboratory environments, or office settings. This project therefore demonstrates a broader approach to intelligent, collaborative robotic assistance through a focused dental proof of concept.

### 2. Creativity
"At its heart, this project reimagines the robotic arm not as a programmed machine, but as an intuitive partner. By integrating multimodal perception with the Hugging Face LeRobot framework, we moved away from the rigid, jerky trajectories of traditional automation in favor of fluid, human-like motion. Because the system is trained directly on human demonstrations, it captures the subtle nuances of dexterity learning not just how to move a tool, but how to hand it over gently.

The true innovation, however, is in the choreography of collaboration. The robot doesn’t just blindly execute tasks; it understands the rhythm of the procedure. It uses Vision-Language Models to read the room and explicitly models 'readiness,' waiting patiently in a standby pose until it detects a gloved hand. This creates a safety layer that feels instinctive rather than forced.

Finally, placing this technology in a dentist’s office grounds high-level robotics in a scenario everyone understands. Seeing a robot calmly waiting to hand a mirror to a dentist transforms complex AI concepts into a relatable, lighthearted interaction, proving that the future of robotics is as much about trust and timing as it is about code."

### 3. Technical Implementations

- **Teleoperation / Dataset capture**
Demonstrations were collected using teleoperation of the S0-101 robotic arm to capture task-relevant trajectories for tool pickup and handover. The dataset includes variations in tool placement, including both organized and unorganized table layouts, to encourage generalization. Visual data from the robot’s camera captures the workspace, tools, and dentist interaction cues in our case the gloved hand during demonstrations.

**Dataset URLs :** 
Pick and Hand Yellow Mirror : https://huggingface.co/datasets/Zer0Lander/PickHandYellowMirrorv1
Pick and Hand Blue Tweezers : https://huggingface.co/datasets/Zer0Lander/PickHandBlueTweezersv1

**Visualization URLs :** 
Yellow Mirror : https://huggingface.co/spaces/lerobot/visualize_dataset?path=%2FZer0Lander%2FPickHandYellowMirrorv1%2Fepisode_0
Blue Tweezer : https://huggingface.co/spaces/lerobot/visualize_dataset?path=%2FZer0Lander%2FPickHandBlueTweezersv1%2Fepisode_0

- **Training**
We trained two separate action policies:
 - One policy for picking up and handing a yellow dental mirror
 - One policy for picking up and handing blue  tweezers

Each policy learns to identify the target object, approach it, and grasp it robustly across different table configurations. After pickup it stays in standby until the dentist’s gloved hand is detected. Training focused on consistency, safe motion, and repeatability rather than scaling to a large number of tools, aligning with the project’s proof-of-concept scope.

Since it is our first time ever working with such a robot and lerobot framework. We tried to make our dataset as clean as possible and easy to replicate the scenario and add more episodes to it.

**Model URLs :**
Yellow Mirror Policy : https://huggingface.co/Zer0Lander/act_policy_yellow_mirror_vf
Blue Tweezer Policy : https://huggingface.co/Zer0Lander/act_policy_blue_tweezers_vf 

- **Inference**
  
At inference time, a Vision-Language Model observes the scene and generates a semantic description of the environment, including the patient area, the dentist’s position, and the robot’s current state. The VLM also predicts the next tool based on the mouth of the patient. If it is closed, it will recommend a mirror if it is opened it will recommend tweezers.

After grasping the tool, the robot transitions into a standby handover mode. The system continuously monitors for the dentist’s gloved hand appearing on the correct side of the robot. Only when this condition is satisfied does the robot perform a controlled and gentle handover motion. If the glove is not detected, the robot will not proceed, ensuring safety and adherence to the intended interaction protocol.

### 4. Ease of Use
- How generalizable is your implementation across tasks or environments?
The robot reliably performs tool pickup and handover across variations of the same environment. In particular, it operates when the tool table is either neatly organized or unorganized (fails sometimes).

- Flexibility and adaptability of the solution
Once the required tool is selected by the system, the robot retrieves it and enters a standby state while holding the object. The robot continuously monitors the scene and only initiates the handover when the dentist’s gloved hand is detected on the correct side of the robot. If the hand is not detected or the conditions are ambiguous, the robot remains in standby, allowing the dentist to control the timing of the interaction implicitly.

- Types of commands or interfaces needed to control the robot
No explicit commands, controllers, or wearable devices are required from the dentist during operation. Interaction is entirely based on visual cues and scene understanding, reducing cognitive load and allowing the dentist to remain focused on the patient. The absence of manual input or configuration steps contributes to a smooth and non-intrusive workflow.

Overall, the demonstrated system provides a clear and low-effort interaction loop within its current scope, emphasizing safety, clarity, and ease of integration into the dental assistant scenario.
