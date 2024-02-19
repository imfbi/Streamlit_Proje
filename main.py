import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.preprocessing import StandardScaler


def load_data(dataset_name):
    if dataset_name == 'Diabetes':
        data = pd.read_csv('C:/Users/PC/Desktop/diabetes.csv')
    elif dataset_name == 'Breast Cancer':
        data = pd.read_csv('C:/Users/PC/Desktop/Breast Cancer Wisconsin Data.csv')
    return data

def clean_data(data):
    if 'Unnamed: 32' in data.columns:
        data = data.drop(['Unnamed: 32'], axis=1)
    return data

def convert_outcome(data):
    if 'diagnosis' in data.columns:
        data['diagnosis'] = data['diagnosis'].map({'M': 1, 'B': 0})
    elif 'Outcome' in data.columns:
        data['Outcome'] = data['Outcome'].map({1: 1, 0: 0})
    return data

def train_model_with_grid_search(model, param_grid, X_train, y_train):
    grid_search = GridSearchCV(model, param_grid, cv=5)
    grid_search.fit(X_train, y_train)
    best_params = grid_search.best_params_
    model.set_params(**best_params)
    model.fit(X_train, y_train)
    return model


def main():

    st.title('Veri Tabanlı Model Geliştirme ve Analizi: Ön İşlemlerden Model Gerçeklemesine')
    st.sidebar.title('Veri Seti ve Model Seçimi')

    dataset_name = st.sidebar.selectbox('Veri Seti', ('Breast Cancer','Diabetes'))

    data = load_data(dataset_name)
    st.subheader(dataset_name+' Veri Seti')
    st.subheader('Veri Setinin İlk 10 Satırı')
    st.write(data.head(10))

    st.subheader('Sütunlar')
    st.write(data.columns.tolist())
    if dataset_name=='Breast Cancer':
        data = clean_data(data)
    data = convert_outcome(data)

    st.subheader('Veri Setinin Son 10 Satırı')
    st.write(data.tail(10))

    st.subheader('Korelasyon Matrisi')

    if dataset_name == 'Breast Cancer':
        st.write("Malignant (M) ve Benign (B) olarak ayrılmış verilerin korelasyon matrisi:")
        plt.figure(figsize=(10, 8))
        fig, ax = plt.subplots()
        sns.scatterplot(x='radius_mean', y='texture_mean', hue='diagnosis', data=data, palette='coolwarm', ax=ax)
        ax.set_title('Malignant (Kötü Huylu) ve Benign (İyi Huylu)')
        st.pyplot(fig)
    elif dataset_name == 'Diabetes':
        plt.figure(figsize=(10, 8))
        sns.scatterplot(x='Glucose', y='BloodPressure', hue='Outcome', data=data, palette='coolwarm')
        plt.xlabel('Glucose')
        plt.ylabel('Blood Pressure')
        plt.title('Glucose vs. Blood Pressure')
        st.pyplot(plt.gcf())
    else:
        corr = data.corr()
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr, annot=True, cmap='coolwarm')
        st.pyplot()

    X = data.drop('diagnosis', axis=1) if 'diagnosis' in data.columns else data.drop('Outcome', axis=1)
    y = data['diagnosis'] if 'diagnosis' in data.columns else data['Outcome']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    model_name = st.sidebar.selectbox('Model Seçimi', ('KNN', 'SVM', 'Naïve Bayes'))

    if model_name == 'KNN':
        model = KNeighborsClassifier()
        param_grid = {'n_neighbors': [3, 5]}
    elif model_name == 'SVM':
        model = SVC()
        param_grid = {'C': [0.1, 1], 'gamma': [0.1, 0.01], 'kernel': ['rbf', 'linear']}
    else:  # Naïve Bayes
        model = GaussianNB()
        param_grid = {}

    model = train_model_with_grid_search(model, param_grid, X_train, y_train)

    st.write(f"Seçilen Model: {model_name}")
    st.write(f"En iyi parametreler: {model.get_params()}")

    y_pred = model.predict(X_test)

    classification_rep = classification_report(y_test, y_pred, output_dict=True)

    # Classification report'u pandas DataFrame'e dönüştürme !
    df_classification_rep = pd.DataFrame(classification_rep).transpose()

    st.subheader('Classification Report')

    st.write(df_classification_rep.style.set_table_styles([{
        'selector': 'td',
        'props': [('padding', '20px')]
    }]), unsafe_allow_html=True)

    conf_matrix = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, cmap='coolwarm', fmt='d')
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix')
    st.pyplot(plt.gcf())

    # F1 skorlarını hesaplayalım
    f1_scores = {}

    for model_name in ['KNN', 'SVM', 'Naïve Bayes']:
        if model_name == 'KNN':
            model = KNeighborsClassifier()
            param_grid = {'n_neighbors': [3, 5]}
        elif model_name == 'SVM':
            model = SVC()
            param_grid = {'C': [0.1, 1], 'gamma': [0.1, 0.01], 'kernel': ['rbf', 'linear']}
        else:  # Naive Bayes
            model = GaussianNB()
            param_grid = {}

        model = train_model_with_grid_search(model, param_grid, X_train, y_train)
        y_pred = model.predict(X_test)
        f1 = f1_score(y_test, y_pred)
        f1_scores[model_name] = f1

    # En yüksek F1 skoruna sahip modeli bulmak için
    best_model = max(f1_scores, key=f1_scores.get)
    best_score = f1_scores[best_model]
    st.subheader('EN İYİ MODEL')
    st.write(f"F1 skoruna göre en iyi model: {best_model} (F1 skoru: {best_score})")

if __name__ == '__main__':
    main()
