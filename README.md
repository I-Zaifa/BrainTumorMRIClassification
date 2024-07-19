## MRI Scan Based Tumor Detection
##### This project is about building an image classification CNN using PyTorch to seperate MRI scans of types of Brain Tumors into the following categories i.e, Glioma tumor, Meningioma tumor, Pituitary tumor, _and Normal_.

### Dataset
 The dataset was taken from kaggle: https://www.kaggle.com/datasets/thomasdubail/brain-tumors-256x256. 

 It has 3 types of brain tumors in 3 different folders along with a seperate folder for a _normal_ brain. The data was preprocessed which included normalizing it, resizing it and introducing randomness into it to enchance the models performance. Total Images are ~3500. It was divided into Training (70%), Validation (15%) and Testing (15%) parts.

### Requirements
  >Pip install requirements.txt

### Results
 The model was trained for 1000 epochs but as you can see in the jupyter notebook that 300-400 would have been enough as the results tend to stay the same after that. It took 6 hours to train the model on an _RTX_ gpu. I could not upload the pth file for the saved model as it exceeded github's allowed storage limit per file (The saved model is not based on the last run epoch but on the highest validation result). The final test data prediction accuracy is 91%. It is very good in predicting and differentiating between Glioma, Pituitary and Meningioma tumors but it has issues with normal brain scans. The project only needs to be updated with a larger dataset to provide further better performance. 


### License
###### This project is licensed under the MIT License - see the LICENSE file for details.
