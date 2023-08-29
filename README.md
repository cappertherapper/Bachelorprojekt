# Relational shape measures of coagulating milk proteins
This repository contains the work of a Bachelor of Science (BSc) thesis, which introduces a novel approach to the study of casein micelle coagulation during milk curdling through the application of spatial analysis methods. The project involves developing a quantitative method for spatial analysis, based on Ripley's K-function and computational imaging, to understand the structure and behavior of coagulating casein proteins in milk. The method computes relational shape measures to analyze the size and spatial distribution of protein clusters at different stages of coagulation and emulates real observations by generating synthetic data.

## Setup
1. Install dependencies
2. Import microscopy video of coagulating cheese to path `\images\`
3. Run notebooks

## Spacial image analysis
Run `demo.ipynb` with path to video.

## Optimize parameters for synthetic images
Run `gradient_descent.ipynb` with path to video.

## About the Data
The provided data consists of a sample time-lapse video of microscopic imagery capturing the aggregation of casein micelles in milk treated with a rennet enzyme and placed in an incubator at 37°C. The proteins are visualized using a fluorescent dye that binds to all the different types of proteins present in milk. The videos are treated as a sequence of grayscale images capturing the transition from milk to cheese.

## Conclusion of report
The developed method allows for easy understanding of the average spatial distribution in an image by computing the mean measures, hence interpretability is predominantly limited to general trends within an image. Although there are areas in the study yet to be explored that could improve the model, the method provides a solid foundation for food scientists and other specialists to investigate further the possible links between our model and the corresponding biological processes.



