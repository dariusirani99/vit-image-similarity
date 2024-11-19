# General

- This repository is for building the ViT Image Similarity Model.
- It requires the use of Google Cloud Integration with the MAR file generated.
- Please see the "console_scripts" for the scripts to run on the VM and on the raspberry pi.


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

1. Transfer over the "fetch_compute_images.py" script to a VM instance.

   - Ensure that python is running on a virtual environment inside the machine

2. Pip download all requirements listed in requirements.txt file

3. Place fetch_compute_images.py script in a folder known to you on your Virtual Machine.

4. Place take_and_upload.py script in a folder known to you on your Raspberry Pi.

5. After running the manage.py command, place the "vitsimilaritymodel-0.1.0" in a folder named "torchserve/model-store", with the "torchserve" folder being your main folder for torchserve.

6. Place the "config.properties" file located in vit-similarity-model/supplemental in the torchserve folder (NOT the torchserve/model-store folder)

7. Download your Google Cloud Application credentials for your entire project as a json, and place in a known path on your Virtual Machine AND your Raspberry pi.

8. Go into fetch_compute_images.py and take_and_upload.py and add your google cloud data where the #TODO lines of text are.

9. Ensure your ~/.bashrc file on your VM has the following lines of code:
    '''
    # adding paths
    export GOOGLE_APPLICATION_CREDENTIALS="path to your google application credentials json"
    export JAVA_HOME=path to your java bin
    export PATH=$JAVA_HOME/bin:$PATH

    # Starting Python script
    python3 path to fetch_compute_images.py &

    # Starting TorchServe
    (
      cd /home/ubuntu/google-cloud/torchserve &&
      torchserve --start --model-store model-store --disable-token-auth
    ) &
'''

10. After all these steps are taken, you should be able to run the command on your Raspberry pi, connected to a camera:
    '''
    python take_and_upload.py
    '''
