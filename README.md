# text detoxification #
This is my little project about text detoxification. I tried to find some good solutions for this task and
compare them to each other.

---

## How to use ##
This project is more about my learning and understanding how to detoxify, but you may use it for yourself

### Detoxification ###
To use the model for detoxification, just use `src/models.predict` function. You may find the example of its work here

### Training ###
To train your own model you may use `src/models.train` function. You may find how I did it [here](notebooks/3.0-train-model.ipynb).
The whole process was made by using initial dataset for this problem. If you want to add new examples, you have to
create your own training and evaluation processes then, or add samples to the dataset.

---

## Structure ##
Here's a structure of the repo:
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
├── references   # manuals and all other explanatory materials.
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
    │
    └── visualization   # Scripts to create exploratory and results oriented visualizations (Now it's empty)
```
`data` folder is missed, because GitHub does not allow to push data more than 100 mb here.
I still did not solve this problem, so I hope you will add it by yourself:)
The problem with the best model storing.

---

## References ##
Check `references` dictionary to find some materials I use for this project.

---

## Author ##
This project was made by [me](https://github.com/Zener085) - Didenko Timofey, t.didenko@innopolis.university, BS21-DS02