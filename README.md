# General

- This repository is for building the ViT Image Similarity Model.
- It requires the use of Google Cloud Integration with the MAR file generated.


# Training the Models

To train the ViT Model, follow these steps:

1. **Set desired model hyperparameters**

   - Adjust desired training hyperparameters in 'config/train_config.yml', or leave as default.

2. **Run the train_model.py scrpt in a CMD window:**

     ```
     python train_model.py
     ```

# Generating the MAR file

To generate the mar file, follow these steps:

1. **Train model using steps above**

2. **Download the required dependencies to prevent version mismatches**
   - Use the following command to generate the mar file in a CMD window:
     ```
     cd vit-similarity-model
     python download_dependencies.py download
     ```

3. **Run the manage.py command**

   - Use the following command to generate the mar file in a CMD window:
     ```
     cd vit-similarity-model
     python manage.py build
     ```

# Integrating MAR file with Virtual Machine

1. **Run inference using the MAR file**

   - Ensure that the config.properties file in "supplemental/config.properties" is included in the torchserve folder when running the model.

   - The MAR file should contain all files needed for the model to run inference. Use the torchserve command to start inference on any virtual machine:
     ```
     torchserve --start --model-store "name of model store folder with MAR file" --disable-token-auth
     ```
