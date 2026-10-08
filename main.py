import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder,OneHotEncoder,StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report

#data preview
df=pd.read_csv("adult.csv")
df.columns = df.columns.str.strip()
df["income"]=df["income"].str.strip()
print("\nFirst 5 rows of the dataset:\n")
print(df.head())


X=df.drop("income",axis=1)
Y=df["income"]

#Splitting dataset
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=42)
print("\nData types of X_train:")
print(X_train.dtypes)
print("\nData types of Y_train:")
print(Y_train.dtypes)
print("\nFirst 5 values of Y_train:")
print(Y_train[:5])

numerical_cols=X_train.select_dtypes(include=["int64","float64"]).columns
categorical_cols=X_train.select_dtypes(include=["str"]).columns
print("\n columns classification:\n")
print("Numerical columns: ",numerical_cols)
print("\n")
print("Categorical columns: ",categorical_cols)

le=LabelEncoder()

Y_train=le.fit_transform(Y_train)
Y_test=le.transform(Y_test)
print("Classes in Y_train:", le.classes_)
print("First 5 values of Y_train:", Y_train[:5])

encoder=OneHotEncoder(handle_unknown="ignore")
preprocessor=ColumnTransformer(transformers=[("num", StandardScaler(), numerical_cols), ("cat", encoder, categorical_cols)])
X_train_transformed=preprocessor.fit_transform(X_train)
X_test_transformed=preprocessor.transform(X_test)
print("\nFeature names after transformation:\n")
print(preprocessor.get_feature_names_out())
print("\nX_train shape:", X_train_transformed.shape)
print("X_test shape:", X_test_transformed.shape)

knn=KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train_transformed,Y_train)
Y_pred=knn.predict(X_test_transformed)
print("\n model evaluation:\n")
print("Accuracy: ",accuracy_score(Y_test,Y_pred))
cm=confusion_matrix(Y_test,Y_pred)
print("Confusion Matrix:\n",cm)
print("Classification Report:\n",classification_report(Y_test,Y_pred))

X_train_knn, X_val_knn, Y_train_knn, Y_val_knn = train_test_split(X_train_transformed, Y_train, test_size=0.2, random_state=42)

k_values=[1,3,5,7,9]
print("\nValidation Accuracy for different k values:\n")
for k in k_values:
    knn=KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train_knn, Y_train_knn)
    y_val_pred=knn.predict(X_val_knn)
    print(f"Validation Accuracy for k={k}: ", accuracy_score(Y_val_knn, y_val_pred))

best_knn=KNeighborsClassifier(n_neighbors=7)
best_knn.fit(X_train_transformed,Y_train)
Y_pred_best=best_knn.predict(X_test_transformed)
print("\nBest Model Accuracy: ",accuracy_score(Y_test,Y_pred_best))

new_person = pd.DataFrame([{
    "age": 30,
    "workclass": "Private",
    "fnlwgt": 200000,
    "education": "Bachelors",
    "education-num": 13,
    "marital-status": "Never-married",
    "occupation": "Tech-support",
    "relationship": "Not-in-family",
    "race": "White",
    "sex": "Female",
    "capital-gain": 0,
    "capital-loss": 0,
    "hours-per-week": 40,
    "native-country": "United-States"
}])
new_person_transformed = preprocessor.transform(new_person)
new_person_pred = best_knn.predict(new_person_transformed)
new_person_pred_label = le.inverse_transform(new_person_pred)
print("\nPredicted income for the new person: ", new_person_pred_label[0])
0


