import pandas as pd
import numpy as np
import re
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

class DataPreprocessor:
    """
    DataPreprocessor Class

    This class is responsible for loading and preprocessing the supply chain dataset.
    It performs multiple preprocessing steps such as:
    - Removing unnecessary columns
    - Handling missing values
    - Cleaning text data
    - Extracting useful features from date columns
    - Encoding categorical variables
    - Normalizing numerical features

    Attributes
    ----------
    data : pandas.DataFrame
        Original dataset loaded from the CSV file.

    processed_data : pandas.DataFrame
        The cleaned and transformed dataset after preprocessing.
    """

    def __init__(self, filepath):
        """
        Constructor of the DataPreprocessor class.

        Parameters
        ----------
        filepath : str
            Path to the CSV dataset file.

        This method loads the dataset and automatically starts
        the preprocessing pipeline.
        """

        # Load the dataset with a specific encoding
        self.data = pd.read_csv(filepath, encoding='ISO-8859-1')

        # Placeholder for the processed dataset
        self.processed_data = None

        # Run preprocessing automatically when object is created
        self._preprocess()


    def _preprocess(self):
        """
        Private preprocessing method.

        This method performs all data cleaning and transformation steps:
        1. Removes irrelevant columns
        2. Handles missing values
        3. Cleans text columns
        4. Converts date columns and extracts new time features
        5. Creates a new feature (shipping delay)
        6. Encodes categorical variables using Label Encoding
        7. Normalizes numerical features using MinMaxScaler
        """

        # Create a copy of the original dataset to avoid modifying it directly
        data = self.data.copy()

        # ----------------------------------------------------
        # Step 1: Remove unnecessary columns
        # These columns are not useful for analysis or modeling
        # ----------------------------------------------------
        cols_to_drop = [
            'Product Image', 'Product Card Id', 'Product Category Id',
            'Order Zipcode', 'Department Id', 'Customer Zipcode',
            'Customer Lname', 'Product Description'
        ]

        # Drop only columns that actually exist in the dataset
        data = data.drop(columns=[c for c in cols_to_drop if c in data.columns])

        # Remove rows with missing critical values
        data = data.dropna(subset=['Sales per customer', 'Product Price'])


        # ----------------------------------------------------
        # Step 2: Clean text columns
        # Convert text to lowercase, remove spaces and special characters
        # ----------------------------------------------------
        text_cols = ['Product Name', 'Delivery Status', 'Category Name', 'Shipping Mode', 'Order Status']

        for col in text_cols:
            data[col] = data[col].astype(str).str.lower().str.strip()
            data[col] = data[col].apply(lambda x: re.sub(r'[^a-z0-9\s]', '', x))


        # ----------------------------------------------------
        # Step 3: Convert date columns and extract features
        # ----------------------------------------------------
        data['order date (DateOrders)'] = pd.to_datetime(
            data['order date (DateOrders)'], errors='coerce'
        )

        data['shipping date (DateOrders)'] = pd.to_datetime(
            data['shipping date (DateOrders)'], errors='coerce'
        )

        # Extract useful time-based features
        data['Month'] = data['order date (DateOrders)'].dt.strftime('%b')
        data['Month_num'] = data['order date (DateOrders)'].dt.month
        data['Year'] = data['order date (DateOrders)'].dt.year


        # ----------------------------------------------------
        # Step 4: Create a derived feature
        # Shipping delay = actual shipping days - scheduled shipping days
        # ----------------------------------------------------
        data['shipping_delay'] = (
            data['Days for shipping (real)'] - data['Days for shipment (scheduled)']
        )


        # ----------------------------------------------------
        # Step 5: Encode categorical variables
        # Convert text categories into numerical values
        # ----------------------------------------------------
        categorical_cols = ['Delivery Status', 'Category Name', 'Shipping Mode', 'Order Status']

        for col in categorical_cols:
            le = LabelEncoder()
            data[col + '_enc'] = le.fit_transform(data[col])


        # ----------------------------------------------------
        # Step 6: Normalize numerical features
        # Scale numeric data to the range [0,1]
        # ----------------------------------------------------
        numeric_cols = [
            'Days for shipping (real)', 'Days for shipment (scheduled)',
            'Benefit per order', 'Sales per customer',
            'Product Price', 'Late_delivery_risk'
        ]

        scaler = MinMaxScaler()
        data[numeric_cols] = scaler.fit_transform(data[numeric_cols])


        # Save the processed dataset
        self.processed_data = data


# ── Usage Example ──────────────────────────────────────────
if __name__ == "__main__":
    """
    Example usage of the DataPreprocessor class.

    This block runs only when the script is executed directly.
    It loads the dataset, preprocesses it, and prints basic information.
    """

    preprocessor = DataPreprocessor(
        "C:\\Users\\Test\\Desktop\\final_project_DEPI\\final-project-DEPI\\data\\DataCoSupplyChain.csv"
    )

    # Get the processed dataset
    df = preprocessor.processed_data

    # Display dataset shape and first rows
    print(df.shape)
    print(df.head())