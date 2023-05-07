# Drill bit identification task

### **Windows executable** - How to run model inference with inference.exe
- Please place all unseen images in the inference_images directory and execute the inference.exe file

### **Python** - How to run model inference with inference.py

- Within a conda virtual environment please install 
    - pytorch 1.13
    - python 3.9
    - opencv-python
        ```
        conda create -n env_setup python=3.9
        pip install opencv-python
        conda install pytorch torchvision torchaudio -c pytorch (For MAC only)
        ```
- Once done, please place all unseen images in the **inference_images** directory and run

    ``` 
    python inference.py
    ```

- Since the images would be placed in the inference_images folder, the program automatically picks up the images and generates predictions. These predictions will be written to a text file - "output.txt"

- A sample output.txt (generated on the images present inside inference_images) has been provided to see what the final output would look like.


### Note - Python only.
One can also provide another location to the unseen images by executing

```
python inference.py --inferenceDataset './inference_images/'
```

 

