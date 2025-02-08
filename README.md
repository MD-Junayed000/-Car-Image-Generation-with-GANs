# -Car-Image-Generation-with-GANs
![output](https://github.com/user-attachments/assets/f3eb2543-819f-4424-a4ee-2b0a0e554fa4)

This repository contains my deep learning project where I built a Generative Adversarial Network (GAN) to generate realistic car images. The project demonstrates my journey through data preprocessing, GAN architecture development, training, and saving the generated images.
________________________________________
📂 Project Overview
Objectives:
•	Build a GAN model to generate high-resolution car images.
•	Preprocess and merge real-world car datasets for model training.
•	Save and visualize generated images dynamically during training.
Features:
•	Fully functional Generator and Discriminator architectures.
•	Dataset processing: Image resizing, normalization, and visualization.
•	Training setup with loss tracking and saving outputs for analysis.
________________________________________
🛠 Tools and Technologies
•	Python: Core programming language.
•	TensorFlow/Keras: Used to build and train the GAN model.
•	OpenCV & PIL: For image preprocessing and visualization.
•	Matplotlib: For plotting generated image results.
________________________________________
📁 Dataset
The dataset consists of over 4,000 car images merged from multiple folders, resized to 128x128x3 resolution, and normalized for training.
•	Kaggle Dataset [Link (https://www.kaggle.com/datasets/kshitij192/cars-image-dataset/data)](https://www.kaggle.com/datasets/kshitij192/cars-image-dataset/data)
________________________________________
🧱 GAN Architecture
Generator:
•	Converts random noise (latent vector) into high-quality 128x128x3 car images.
•	Includes Dense layers, LeakyReLU activations, BatchNormalization, and Conv2DTranspose layers for upsampling.
Discriminator:
•	Distinguishes between real and fake images with a convolutional architecture.
•	Uses Conv2D layers, Dropout, and LeakyReLU for downsampling.
Combined GAN:
•	The Generator and Discriminator are combined for end-to-end adversarial training.
•	Loss Function: Binary Cross-Entropy Loss
________________________________________
🚀 Training Process
1.	Dataset preprocessed and split into training and testing sets.
2.	Trained the GAN for 3000 epochs using a batch size of 64.
3.	Monitored training stability through Generator Loss and Discriminator Loss.
4.	Saved generated images dynamically during training for analysis.
Results:
•	Successfully generated high-resolution car images within limited epochs.
•	Loss and accuracy metrics show promising stability for further training.
________________________________________
📊 Results
Generated images are saved in the generated_car_images folder. Here's an example:
(Add a sample image here or link a preview image.)
🌟 Future Improvements
•	Fine-tune model hyperparameters for better image realism.
•	Experiment with advanced GAN architectures like DCGAN, StyleGAN, or CycleGAN.
•	Train the model on a larger and more diverse dataset.
📢 Acknowledgments
This project was built as part of my deep learning journey. Feel free to explore, use, and suggest improvements!
________________________________________
📥 Contribute
Feel free to open issues or submit pull requests if you’d like to improve this project.
________________________________________
🔗 Links
•	GitHub Repository: [Link](https://github.com/MD-Junayed000/-Car-Image-Generation-with-GANs)
•	Kaggle Dataset: [Link](https://www.kaggle.com/datasets/kshitij192/cars-image-dataset/data)
>> Contact Me: [www.linkedin.com/in/mohammad-junayed-ete20](https://www.linkedin.com/in/mohammad-junayed-ete20/)

