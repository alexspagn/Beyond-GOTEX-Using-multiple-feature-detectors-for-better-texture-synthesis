# Beyond-GOTEX-Texture-Synthesis

This repository houses the implementation and results of our study "Beyond GOTEX: Using Multiple Feature Detectors for Better Texture Synthesis", an extended and improved version of the paper "A Generative Model for Texture Synthesis based on Optimal Transport between Feature Distributions" that can be found at: https://arxiv.org/abs/2007.03408.

## Overview
Our work delves into enhancing texture synthesis by:
- Employing the GOTEX framework with innovative feature detectors like the InceptionV3 NN.
- Trying the model's limits on more complicated textures
- Tackling issues like contrast loss and checkerboard artifacts in synthesized textures.
- Experimenting with various parameter tunings for optimal results.

## Repository Contents
- `run_vgg_synthesis.py`: Script for single image texture synthesis using both Gaussian patches and VGG-19 features.
- `run_inception_synthesis.py`: Script for single image texture synthesis using both Gaussian patches and InceptionV3 features.
- `run_cnn_synthesis_vgg.py`: Script for texture synthesis using a generative CNN model and both Gaussian patches and VGG-19 features.
- `run_cnn_synthesis_incept.py`: Script for texture synthesis using a generative CNN model and both Gaussian patches and InceptionV3 features.
- `test/`: Folder containing results from our various experiments.
- `CT scans/` : Folder contining tests made on CT scans for two different pathologies.
- `Report.pdf`: Comprehensive report detailing our methods, experiments, and findings.

## Getting Started
To replicate our experiments:
1. Clone the repository.
2. Install necessary dependencies.
3. Run any of the above scripts to see the texture synthesis in action.
4. Explore `test/` to view some of our reproduced textures, some of them have the checkpoint file .pt saved. It can be used for sampling.

## Example use
In order to train and sample a single texture using both Gaussian patches and InceptionV3 features:

`python run_inception_synthesis.py texture_images\demo_texture_9.png --save`

In order to train a Generative CNN using both Gaussian patches and InceptionV3 features:

`python run_cnn_synthesis_inception.py texture_images\demo_texture_9.png --save`

## Authors
- Alessio Spagnoletti

## Contributions and Feedback
We welcome contributions and feedback. Feel free to fork the project, submit pull requests, or open issues for discussion.

## License
This project is under the MIT License - see the LICENSE file for details.
