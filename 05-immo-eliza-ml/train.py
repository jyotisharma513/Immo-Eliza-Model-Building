import joblib
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder


def train():
    """Trains a linear regression model on the full dataset and stores output."""
    # Load the data
    data = pd.read_csv("C:\\Users\\jyoti\\Immo-Eliza-Model-Building\\05-immo-eliza-ml\\data\\properties.csv")

    # Define features to use

    num_features = ["id","zip_code","latitude","longitude","construction_year","total_area_sqm","surface_land_sqm","nbr_frontages","nbr_bedrooms","terrace_sqm","garden_sqm","primary_energy_consumption_sqm","cadastral_income"]
    fl_features = ["fl_furnished","fl_open_fire","fl_terrace","fl_garden","fl_swimming_pool","fl_floodzone","fl_double_glazing"]
    cat_features = ["property_type","subproperty_type","region","province","locality","equipped_kitchen","state_building","epc","heating_type"]

    # Split the data into features and target
    X = data[num_features + fl_features + cat_features]
    y = data["price"]

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=505
    )

    # Impute missing values using SimpleImputer
    


    # Separate features for mean and median imputation
    mean_impute_features = ["latitude", "longitude","nbr_frontages","nbr_bedrooms","construction_year"]  # Add the feature names you want to impute using mean
    median_impute_features = ["total_area_sqm", "surface_land_sqm","terrace_sqm","garden_sqm","primary_energy_consumption_sqm","cadastral_income"]  # Add the feature names you want to impute using median

    # Initialize imputers
    mean_imputer = SimpleImputer(strategy="mean")
    median_imputer = SimpleImputer(strategy="median")

    # Fit and transform mean imputation
    X_train[mean_impute_features] = mean_imputer.fit_transform(X_train[mean_impute_features])
    X_test[mean_impute_features] = mean_imputer.transform(X_test[mean_impute_features])

    # Fit and transform median imputation
    X_train[median_impute_features] = median_imputer.fit_transform(X_train[median_impute_features])
    X_test[median_impute_features] = median_imputer.transform(X_test[median_impute_features])


    # Convert categorical columns with one-hot encoding using OneHotEncoder
    enc = OneHotEncoder()
    enc.fit(X_train[cat_features])
    X_train_cat = enc.transform(X_train[cat_features]).toarray()
    X_test_cat = enc.transform(X_test[cat_features]).toarray()

    # Combine the numerical and one-hot encoded categorical columns
    X_train = pd.concat(
        [
            X_train[num_features + fl_features].reset_index(drop=True),
            pd.DataFrame(X_train_cat, columns=enc.get_feature_names_out()),
        ],
        axis=1,
    )

    X_test = pd.concat(
        [
            X_test[num_features + fl_features].reset_index(drop=True),
            pd.DataFrame(X_test_cat, columns=enc.get_feature_names_out()),
        ],
        axis=1,
    )

    print(f"Features: \n {X_train.columns.tolist()}")

    # Train the model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Evaluate the model
    train_score = r2_score(y_train, model.predict(X_train))
    test_score = r2_score(y_test, model.predict(X_test))
    print(f"Train R² score: {train_score}")
    print(f"Test R² score: {test_score}")

    # Save the model
    artifacts = {
        "features": {
            "num_features": num_features,
            "fl_features": fl_features,
            "cat_features": cat_features,
        },
        "mean_imputer": mean_imputer,
        "median_imputer":median_imputer,
        "enc": enc,
        "model": model,
    }
    joblib.dump(artifacts, "C:\\Users\\jyoti\\Immo-Eliza-Model-Building\\05-immo-eliza-ml\\models\\artifacts.joblib")


if __name__ == "__main__":
    train()

