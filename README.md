# Text Detoxification Project
Welcome to the Text Detoxification project! This project focuses on creating a powerful model to transform toxic
sentences into neutral forms. Whether you're interested in learning about text detoxification or want to apply it to
your own text data, this project has you covered.

---

## How to Use
This project serves both as a learning resource and a practical tool for text detoxification. Follow the steps below to
make the most out of it:

### Detoxification
To detoxify text using the model, simply use the `src/models.predict` function. Check out the example in the code to see
how it works.

### Training
If you want to train your own model, you can use the `src/models.train` function. The training process is explained in
detail in the corresponding section. The model was trained using an initial dataset, but you can customize it by adding
new examples and adapting the training and evaluation processes.

---

## Project Structure
Explore the repository's structure to find what you need:
```
text-detoxification
├── README.md
│
├── data 
│   └── raw      # The original, immutable data
│
├── models       # Trained and serialized models, final checkpoints
│
├── notebooks    # Jupyter notebooks.
│ 
├── references   # Manuals and all other explanatory materials.
│
├── reports      # Generated analysis as PDF
│
│
├── requirements.txt
│
└── src                 # Source code for use in this assignment
    │                 
    ├── data            # Scripts to download or generate data
    │
    ├── models          # Scripts to train models and then use trained models to make predictions
```

Please note that the `data` folder is not included as GitHub restricts pushing data larger than 100 MB. You may need to
add it manually.

---

## Models ##

### BART-based Detox Model (bart_base_model.py) ###
This model utilizes BART for text detoxification. The `bart_detox` function takes a text as input and provides a
detoxified version.

### GPT-3.5 Turbo Model (gpt35_turbo_model.py) ###
This model uses [OpenAI's GPT-3.5 Turbo](https://openai.com/blog/gpt-3-5-turbo-fine-tuning-and-api-updates) for text
detoxification. The `gpt_detox` function takes a toxic text and generates a detoxified version using OpenAI's API.

### T5 Model (t5_model.py) ###
This model employs the T5 transformer for text detoxification. The `t5_detox` function takes a text and returns a
detoxified version.

---

## Project Author ##
This project was created by [Zener085](https://github.com/Zener085).
Feel free to explore, learn, and contribute to this project! If you encounter any issues or have suggestions, don't
hesitate to reach out. Happy learning!