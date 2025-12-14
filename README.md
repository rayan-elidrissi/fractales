# AMD_Robotics_Hackathon_2025_Fractales

## Team Information
- **Team:** Your team number, name of your team, and members

## Summary
A brief description of your work

< Images or video demonstrating your project >

---

## Submission Details

### 1. Mission Description
The mission of this project is to explore how a robotic assistant can support a dentist during routine check-up procedures by autonomously understanding requests, perceiving the environment, and safely handing over medical tools.

In a real-world dental setting, dentists frequently need to switch tools while maintaining focus on the patient and maintaining sterile conditions. Our robotic assistant acts as a proof-of-concept dental assistant that can respond to direct voice requests, observe the clinical workspace, and deliver the appropriate instrument in a controlled and safe manner. The dentist can explicitly ask for a specific tool (e.g., “give me the mirror”), request a tool to be returned (“I don’t need this anymore”), or remain in a standby state while the assistant holds an instrument.

Beyond explicit commands, the assistant is designed to reason about context and workflow. By observing the ongoing procedure and the dentist’s actions, the system can potentially suggest or predict the next tool that may be needed, enabling a more proactive form of assistance rather than purely reactive behavior.

The robot assistant uses vision to monitor the patient area, the tool table, and the dentist’s hand. Once a tool is retrieved, the robot waits in a safe standby position and only performs a handover when the dentist’s gloved hand is detected in a predefined region. If safety conditions are not met, the robot will not proceed, prioritizing hygiene and controlled interaction.

The system is also designed with emergency awareness in mind. In unexpected situations—such as sudden movements, loss of visual tracking, or ambiguous handover signals—the robot can pause, hold position, or abort the handover entirely, allowing the dentist to retain full control of the situation.

While the current implementation focuses on a dental assistant scenario, the underlying logic is intentionally general. The same perception, reasoning, and manipulation framework can be transferred to other domains that involve human–robot collaboration, such as medical assistance in other specialties, mechanical workshops, laboratory environments, or office settings. This project therefore demonstrates a broader approach to intelligent, collaborative robotic assistance through a focused dental proof of concept.

### 2. Creativity
- What is novel or unique in your approach?
- Innovation in design, methodology, or application

### 3. Technical Implementations
- **Teleoperation / Dataset capture**
  <Image/video of teleoperation or dataset capture>
- **Training**
- **Inference**
  <Image/video of inference eval>

### 4. Ease of Use
- How generalizable is your implementation across tasks or environments?
- Flexibility and adaptability of the solution
- Types of commands or interfaces needed to control the robot

### Additional Links
For example, you can provide links to:
- Link to a video of your robot performing the task
- URL of your dataset in Hugging Face
- URL of your model in Hugging Face
- Link to a blog post describing your work

---

## Code Submission
This is the directory tree of this repo. Fill in the `mission` directory with your submission details.

```
AMD_Robotics_Hackathon_2025_ProjectTemplate-main/
├── README.md
└── mission
    ├── code
    │   └── <code and script>
    └── wandb
        └── <latest run directory copied from wandb of your training job>
```

The `latest-run` directory structure should look like this:

```
outputs/train/smolvla_so101_2cube_30k_steps/wandb/
├── debug-internal.log -> run-20251029_063411-tz1cpo59/logs/debug-internal.log
├── debug.log -> run-20251029_063411-tz1cpo59/logs/debug.log
├── latest-run -> run-20251029_063411-tz1cpo59
└── run-20251029_063411-tz1cpo59
    ├── files
    │   ├── config.yaml
    │   ├── output.log
    │   ├── requirements.txt
    │   ├── wandb-metadata.json
    │   └── wandb-summary.json
    ├── logs
    │   ├── debug-core.log -> /dataset/.cache/wandb/logs/core-debug-20251029_063411.log
    │   ├── debug-internal.log
    │   └── debug.log
    ├── run-tz1cpo59.wandb
    └── tmp
        └── code
```

### NOTES
- The `latest-run` is a soft link. Make sure to copy the real target directory it links to, including all subdirectories and files.
- Only provide (upload) the `wandb` directory of your last successful pre-trained model for the Mission.
