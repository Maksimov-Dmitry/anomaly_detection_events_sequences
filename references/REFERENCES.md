# References

This project has been significantly influenced by ideas and code from external sources. Detailed acknowledgements and credits are given below:

## Primary Inspiration and Idea Source
The concept of employing context and the approach to synthesizing data were primarily inspired by the following work:

**Title**: Event Outlier Detection in Continuous Time
**Authors**: S. Liu and M. Hauskrecht
**Publication Details**: Presented in the Proceedings of the 38th International Conference on Machine Learning, PMLR, Jul. 2021, pp. 6793–6803.
**Accessed on**: Nov. 16, 2022.
**URL**: [Event Outlier Detection in Continuous Time](https://proceedings.mlr.press/v139/liu21g.html)

## Code Adaptations
Several components of the project's codebase were adapted from an external repository, with modifications and enhancements to suit the project's requirements. The original source is acknowledged below:

**Repository**: CPPOD
**URL**: [CPPOD GitHub Repository](https://github.com/siqil/CPPOD)
**Summary of Adaptations**:
1. `src/data/make_dataset.py`: Incorporated personalized sequences generation.
2. `src/data/dataloader.py`: Retained only the convert functions from the original source.
3. `src/models/baselines.py`: Implemented several improvements to enhance functionality.
4. `src/models/cppod.py`: Integrated personalization features to augment the model's capabilities.

Special thanks to the authors and contributors of the above-mentioned resources for their valuable work, which substantially contributed to the realization of this project.
