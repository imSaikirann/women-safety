
# Women Safety Tweet Analyzer using Machine Learning

## Project Overview

This project analyzes social media tweets to determine whether a location or situation mentioned in the tweet indicates a **Safe** or **Unsafe** environment for women.
The system uses **Natural Language Processing (NLP)** and a **Machine Learning classification model** to analyze tweet text and predict safety status.

The application is built using:

* Python
* Scikit-Learn
* Streamlit
* NLP (TF-IDF text processing)

---

## How the System Works

1. A small labeled dataset of tweets is used for training.
2. Tweets are converted into numerical features using **TF-IDF Vectorization**.
3. A **Logistic Regression model** is trained on the dataset.
4. Users enter a tweet in the web interface.
5. The model predicts whether the tweet indicates:

   * Safe Area
   * Unsafe Area

---

## Requirements

Install required packages:

```bash
pip install streamlit pandas scikit-learn
```

---

## Running the Application (Without Docker)

### Step-1

Navigate to the project folder:

```bash
cd project_folder
```

### Step-2

Run the application:

```bash
streamlit run app.py
```

### Step-3

Open browser:

```
http://localhost:8501
```

Enter a tweet and click **Analyze** to see prediction.

---

## Running the Application Using Docker

### Step-1 Start Docker container

```bash
docker run -p 8888:8888 -p 8501:8501 -v C:\Users\neela:/workspace -it jupyter/scipy-notebook
```

### Step-2 Install dependencies inside container

```bash
pip install streamlit pandas scikit-learn
```

### Step-3 Run Streamlit app

```bash
streamlit run app.py --server.port 8501 --server.address 0.0.0.0
```

### Step-4 Open browser

```
http://localhost:8501
```

---

## Example Inputs

| Tweet                                      | Result |
| ------------------------------------------ | ------ |
| "Street is dark and no police patrol"      | Unsafe |
| "CCTV cameras and police security present" | Safe   |

---

## Project Workflow

* Tweet Collection
* Text Preprocessing
* Feature Extraction (TF-IDF)
* Model Training
* Prediction through Web Interface

---

## Purpose of Project

This project demonstrates how **machine learning and NLP techniques** can be used to analyze social media data to understand women safety conditions in cities and identify potential risk areas.

---
