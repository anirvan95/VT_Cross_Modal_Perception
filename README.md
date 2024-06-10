Visuo-Tactile Cross Modal prediction

This is the repository for the cross-modal experiments and implementation

Current Datasets - 
1) Pendulum: Please download the dataset from the Google Drive [link](https://drive.google.com/drive/folders/1-fOrB2GfavJFHxQfdyj4codQHhGTvqmJ?usp=drive_link).
2) ..

Folder Structure - 
1) datasets/ - Contains the dataset files
2) debuggers/ - Test scripts
3) documentation/ - Details about implemenation, network model, PGMs
4) dump/ - Training plots and videos for evaluation
5) legacy/ - Previous implemenation and codes
6) results/ - Saves the training files
7) utils/ - Helper files for training the vae and dvbf

Run & Train - 
1) In a new terminal type and run `visdom`. Select the link highlighted in the 
terminal to open the visualisation tab in the default browser
2) Use `train_pendulum_vae.py` to pre-train the VAE model
3) Use `train_pendulum_dvbf.py` to train the DVBF filtering model
4) Use `test_pendulum_dvbf.py` to evaluate and test the trained model (lets call it basic one)
OR
5) Use `test_pendulum_dvbf_fusion.py` to evaluate and test the latent filter with Bayesian Integration model.

Current DVBF (Latent Filter model)
The current DVBF network model for the basic version is - 
![Alt text](documentation/latent_filter.png)

The current DVBF network model for the basic version is - 
![Alt text](documentation/latent_filter_bi.jpg)

The Probabilistic Graphical Model for the basic latent filter is presented as - 
![Alt text](documentation/latent_model.png)

The Probabilistic Graphical Model for the basic latent filter is presented as - 
![Alt text](documentation/latent_model_bi.jpg)