# Feature Clock

It is difficult for humans to perceive high-dimensional data. Therefore, high-dimensional data is projected into lower dimensions to visualize it. 
Many applications benefit from complex nonlinear dimensionality reduction techniques (e.g., UMAP, t-SNE, PHATE, and autoencoders), but the effects of individual high-dimensional features are hard to explain in the latent spaces. 

Most solutions use multiple two-dimensional plots to analyze the effect of every variable in the embedded space, but this is not scalable, leading to k plots for k different variables. 

Our solution, Feature Clock, provides novel visualizations that eliminate the need for multiple plots to inspect the influence of original variables in the latent space. Feature Clock enhances the explainability and compactness of visualizations of embedded data and is available in an open-source Python library.
