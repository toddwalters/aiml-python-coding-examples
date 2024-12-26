# Setup Instructions

## Creating a Conda Environment

To create a new conda environment that leverages Python 3.10.14 and installs all of the Python modules listed in the `requirements.txt` file, follow these steps:

1. **Install Conda:**
   If you haven't already installed Conda, you can download and install it from the [official Conda website](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html).

2. **Create a New Conda Environment:**
   Open a terminal or command prompt and run the following command to create a new conda environment:

    ```zsh
    conda env create -f conda-env.yaml
    ```

3. **Activate the Conda Environment:** Activate the newly created environment using the following command:

    ```zsh
    conda activate ggl-colab-genai-[macos|winos]-20241208
    ```
