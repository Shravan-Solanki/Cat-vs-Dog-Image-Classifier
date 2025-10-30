# Cat vs. Dog Image Classifier

This project is a Convolutional Neural Network (CNN) built with TensorFlow and Keras to classify images as either cats or dogs. The model is trained on the "Dogs vs. Cats" dataset from Kaggle.


## Model Architecture

The model is a `Sequential` Keras model with the following architecture:

1.  **Input:** 50x50 grayscale images (normalized by dividing by 255.0)
2.  **Layer 1:** `Conv2D` (64 filters, 3x3 kernel) -> `ReLU` -> `MaxPooling2D`
3.  **Layer 2:** `Conv2D` (64 filters, 3x3 kernel) -> `ReLU` -> `MaxPooling2D`
4.  **Layer 3:** `Conv2D` (64 filters, 3x3 kernel) -> `ReLU` -> `MaxPooling2D`
5.  **Flatten:** Flattens the 3D feature map into a 1D vector.
6.  **Output Layer:** `Dense` (1 neuron) with a `sigmoid` activation function for binary classification.

The model is compiled using the `adam` optimizer and `binary_crossentropy` loss, which is ideal for a two-class problem.

## Dataset

This model is trained on the [Kaggle "Dogs vs. Cats" dataset](https://www.kaggle.com/competitions/dogs-vs-cats/data).

To use this project, you must download the `train.zip` file from Kaggle and extract its contents into a folder named `train/` in the root of this repository. The notebook also uses a `test1/` folder for prediction examples.

## How to Use

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/Shravan-Solanki/Cat-vs-Dog-Image-Classifier.git
    cd Cat-vs-Dog-Image-Classifier
    ```

2.  **Set up the environment:**
    It's recommended to use a virtual environment:
    ```bash
    python -m venv env
    source env/bin/activate  # On Windows, use `env\Scripts\activate`
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Get the data:**
    * Download `train.zip` and `test1.zip` from the [Kaggle competition page](https://www.kaggle.com/competitions/dogs-vs-cats/data).
    * Create a `train/` folder and extract the training images into it.
    * Create a `test1/` folder and extract the test images into it.

5.  **Train the model:**
    * Run the `catvsdog.ipynb` notebook.

      or 
    * Run the ` catvsdog.py` script
    * The notebook will process the data, build the CNN, and train it for 10 epochs.
    * After training, it will save the model to a file named `64x3-CNN.keras`.

6.  **Make predictions:**
    The notebook includes a section at the end that shows how to:
    * Load the saved `64x3-CNN.keras` model.
    * Load and preprocess a new image from the `test1/` folder.
    * Predict whether the image is a "cat" or a "dog".

## Technologies Used
* TensorFlow / Keras
* NumPy
* OpenCV (`opencv-python`)
* Matplotlib
* Jupyter Notebook