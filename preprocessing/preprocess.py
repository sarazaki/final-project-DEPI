import pandas as pd
import numpy as np
import re
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import plotly.graph_objects as go
from plotly.subplots import make_subplots

class DataPreprocessor:
    def __init__(self, filepath):
        self.data = pd.read_csv(filepath, encoding='ISO-8859-1')
        self.processed_data = None
        self.PALETTE  = ['#4f8ef7','#7c5cbf','#2ec4b6','#f7c94f','#f76d6d','#e884f7','#56d364','#ff9f43']
        self.LAYOUT = dict(
            paper_bgcolor='#0b0e1a', plot_bgcolor='#131728',
            font=dict(family='Courier New, monospace', color='#e4e8f5', size=12),
            title_font=dict(size=15, color='white'),
            legend=dict(bgcolor='rgba(0,0,0,0)', bordercolor='#1e2540'),
            margin=dict(t=90, b=60, l=70, r=50)
        )
        self.AXIS = dict(gridcolor='#1e2540', gridwidth=1, zerolinecolor='#1e2540',
                         linecolor='#252d45', tickfont=dict(color='#7a87ab', size=10),
                         title_font=dict(color='#e4e8f5'))
        self._preprocess()

    def _preprocess(self):
        data = self.data.copy()

        # Drop unnecessary columns
        cols_to_drop = ['Product Image', 'Product Card Id', 'Product Category Id', 'Order Zipcode',
                        'Department Id','Customer Zipcode','Customer Lname','Product Description']
        data = data.drop(columns=[c for c in cols_to_drop if c in data.columns])
        data = data.dropna(subset=['Sales per customer', 'Product Price'])

        # Clean text
        text_cols = ['Product Name', 'Delivery Status', 'Category Name', 'Shipping Mode', 'Order Status']
        for col in text_cols:
            data[col] = data[col].astype(str).str.lower().str.strip()
            data[col] = data[col].apply(lambda x: re.sub(r'[^a-z0-9\s]', '', x))

        # Dates
        data['order date (DateOrders)'] = pd.to_datetime(data['order date (DateOrders)'], errors='coerce')
        data['shipping date (DateOrders)'] = pd.to_datetime(data['shipping date (DateOrders)'], errors='coerce')
        data['Month'] = data['order date (DateOrders)'].dt.strftime('%b')
        data['Month_num'] = data['order date (DateOrders)'].dt.month
        data['Year'] = data['order date (DateOrders)'].dt.year

        # Derived
        data['shipping_delay'] = data['Days for shipping (real)'] - data['Days for shipment (scheduled)']

        # Encode categorical
        categorical_cols = ['Delivery Status', 'Category Name', 'Shipping Mode', 'Order Status']
        for col in categorical_cols:
            le = LabelEncoder()
            data[col + '_enc'] = le.fit_transform(data[col])

        # Scale numeric
        numeric_cols = ['Days for shipping (real)', 'Days for shipment (scheduled)',
                        'Benefit per order', 'Sales per customer', 'Product Price', 'Late_delivery_risk']
        scaler = MinMaxScaler()
        data[numeric_cols] = scaler.fit_transform(data[numeric_cols])

        self.processed_data = data

    # ────────────── FIGURE FUNCTIONS ──────────────
    def get_shipping_figure(self):
        data = self.processed_data
        fig1 = make_subplots(rows=2, cols=2, subplot_titles=(
            'Real vs Scheduled Shipping Days',
            'Delivery Status Distribution',
            'Late Delivery Risk by Shipping Mode (%)',
            'Shipping Delay Distribution (Real − Scheduled)',
        ))
        # Plot logic here (use your previous fig1 code, remove invalid 'titlefont')
        # Example: replace colorbar with correct Plotly 6+
        # ...
        return fig1

    def get_sales_figure(self):
        data = self.processed_data
        fig2 = make_subplots(rows=2, cols=2, subplot_titles=(
            'Sales Distribution (per order)',
            'Avg Profit Ratio by Category',
            'Total Sales by Market',
            'Order Profit per Order — by Customer Segment',
        ))
        # Plot logic here
        return fig2

    def get_order_figure(self):
        data = self.processed_data
        fig3 = make_subplots(rows=2, cols=2, specs=[[{'type':'xy'},{'type':'domain'}],
                                                    [{'type':'xy'},{'type':'xy'}]],
                             subplot_titles=(
                                'Monthly Order Volume',
                                'Order Status Distribution',
                                'Discount Rate vs Profit Ratio',
                                'Top 10 Categories by Total Sales',
                             ))
        # Plot logic here
        return fig3