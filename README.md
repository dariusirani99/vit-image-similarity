# Copyright Notice

This repository and its contents are the intellectual property of **Darius Irani**, protected under applicable copyright laws. As stated in GitHub’s [Terms of Service](https://docs.github.com/en/github/site-policy/github-terms-of-service#6-repository-contents), I retain ownership of all content I post in this repository.

Anyone attempting to claim credit for this code or integration using this code without proper attribution will be considered in violation of these terms. Such violations must be reported directly to **Darius Irani** and may result in legal action.

Unauthorized claiming of credit for this work is strictly prohibited and will result in legal action. However, this repository is open for use, and individuals are permitted to use it for their own purposes as long as proper credit is given, and ownership is not claimed. Any fork or reuse must include clear attribution to the original author, Darius Irani.

By using this repository, you agree to these terms.

---

# General

- This repository is for building the ViT Image Similarity Model.
- It requires the use of Google Cloud Integration with the MAR file generated.
- Please see the "console_scripts" for the scripts to run on the VM and on the Raspberry Pi.

# Training the Models

To train the ViT Model, follow these steps:

1. **Set desired model hyperparameters**

   - Adjust desired training hyperparameters in 'config/train_config.yml', or leave as default.

2. **Run the train_model.py script in a CMD window:**

     ```plaintext
     python train_model.py
     ```

# Generating the MAR file

To generate the MAR file, follow these steps:

1. **Train the model using steps above**

2. **Download the required dependencies to prevent version mismatches**

   - Use the following command to generate the MAR file in a CMD window:
     ```plaintext
     cd vit-similarity-model
     python download_dependencies.py download
     ```

3. **Run the manage.py command**

   - Use the following command to generate the MAR file in a CMD window:
     ```plaintext
     cd vit-similarity-model
     python manage.py build
     ```

# Predicting a Single Image

1. **Train the model using steps above**

2. **Use the predict_model.py script to generate a prediction**

   - Use the following command to predict the output given a single image:
     ```plaintext
     python predict_model.py --model-file {path to model file} --device {cuda or cpu} --input-file {path to input image}
     ```
3. **See the output image and json in the ./output folder**


# Integrating MAR file with Virtual Machine

1. **Transfer the "fetch_compute_images.py" script to a VM instance.**

   - Ensure that Python is running on a virtual environment inside the machine.

2. **Pip download all requirements listed in the requirements.txt file.**

3. **Place fetch_compute_images.py script in a folder known to you on your Virtual Machine.**

4. **Place take_and_upload.py script in a folder known to you on your Raspberry Pi.**

5. **After running the manage.py command, place the "vitsimilaritymodel-0.1.0" in a folder named "torchserve/model-store", with the "torchserve" folder being your main folder for TorchServe.**

6. **Place the "config.properties" file located in `vit-similarity-model/supplemental` in the `torchserve` folder (NOT the `torchserve/model-store` folder).**

7. **Download your Google Cloud Application credentials for your entire project as a JSON file, and place it in a known path on your Virtual Machine AND your Raspberry Pi.**

8. **Edit `fetch_compute_images.py` and `take_and_upload.py` to add your Google Cloud data where the `#TODO` lines are located.**

9. **Ensure your `~/.bashrc` file on your VM has the following lines of code:**

    ```plaintext
    export GOOGLE_APPLICATION_CREDENTIALS="path to your google application credentials json"
    export JAVA_HOME=path to your java bin
    export PATH=$JAVA_HOME/bin:$PATH

    python3 path to fetch_compute_images.py &

    (
      cd /home/ubuntu/google-cloud/torchserve &&
      torchserve --start --model-store model-store --disable-token-auth
    ) &
    ```

10. **After completing all these steps, run the following command on your Raspberry Pi, connected to a camera:**

    ```plaintext
    python take_and_upload.py
    ```
