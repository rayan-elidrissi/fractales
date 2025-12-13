# AMD_Robotics_Hackathon_2025_Fractales

## Team Information
- **Team:** Your team number, name of your team, and members

## Summary
A brief description of your work

< Images or video demonstrating your project >

---

## Submission Details

### 1. Mission Description
- Real world application of your mission

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
