# Beyond-GOTEX-Texture-Synthesis-Project

This repository houses the implementation and results of our study "Beyond GOTEX: Using Multiple Feature Detectors for Better Texture Synthesis".

## Overview
Our work delves into enhancing texture synthesis by:
- Employing the GOTEX framework with innovative feature detectors.
- Utilizing Gaussian pyramid patches and deep VGG features.
- Tackling issues like contrast loss and checkerboard artifacts in synthesized textures.
- Experimenting with various parameter tunings for optimal results.

## Repository Contents
- `run_inception_synthesis.py`: Script for texture synthesis using Inception features.
- `run_optim_synthesis.py`: Script for optimization-based texture synthesis.
- `run_cnn_synthesis.py`: Script for texture synthesis using CNN.
- `run_cnn_synthesis_incept.py`: Script combining CNN and Inception features for texture synthesis.
- `run_gotex_vgg.py`: Script implementing GOTEX with VGG features.
- `experiment_results/`: Folder containing results from our various experiments.
- `Report.pdf`: Comprehensive report detailing our methods, experiments, and findings.

## Getting Started
To replicate our experiments:
1. Clone the repository.
2. Install necessary dependencies.
3. Run any of the above scripts to see the texture synthesis in action.
4. Explore `experiment_results/` to view our findings.

## Authors
- [Author's Name 1]
- [Author's Name 2]
- [Author's Name 3]
- [Additional Authors]

## Contributions and Feedback
We welcome contributions and feedback. Feel free to fork the project, submit pull requests, or open issues for discussion.

## License
This project is under the MIT License - see the LICENSE file for details.
