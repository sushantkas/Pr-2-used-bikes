import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import joblib
import seaborn as sns
import warnings
import re
import pickle
warnings.filterwarnings("ignore")
from hdbscan import approximate_predict , HDBSCAN



class BikePreprocess:
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.df.drop(self.df[self.df["price"]==0].index, inplace=True)
    
    def bike_preprocess_mileage(self):
        self.df["mileage"]= self.df["mileage"].str.strip()
        self.df["mileage"] = self.df["mileage"].str.replace(" kmpl", "")
        self.df["mileage"]=self.df["mileage"].apply(lambda x: x[:2] if isinstance(x, str) else x)
        self.df["mileage"] = self.df["mileage"].astype(int, errors='ignore')
        # Mileage colmn values reflected as string, converting to numeric
        self.df["mileage"]=pd.to_numeric(self.df["mileage"].astype(int, errors='ignore').values, errors='coerce')
        # Engineered a new column for brand name from model_name
        self.df["brand"] = self.df["model_name"].apply(lambda x: x.split(" ")[0])
        # Calculate mean mileage per brand
        brand_means = self.df.groupby('brand')['mileage'].mean()
        # Fillna using map
        self.df['mileage'].fillna(self.df['brand'].map(brand_means), inplace=True)
        self.df["mileage"].fillna(30, inplace=True)
        self.df[["model_name", "brand", "location"]]=self.df[["model_name", "brand", "location"]].apply(lambda x: x.str.title() if x.dtype == "object" else x)
        return self.df
    
    def power_column(self):
        # Ensure model_name, brand, and location are in title case
        self.df[["model_name", "brand", "location"]]=self.df[["model_name", "brand", "location"]].apply(lambda x: x.str.title() if x.dtype == "object" else x)
        power_bhp=lambda x: re.sub(r'[a-zA-Z]', '',  x.split(" ")[0])[:3] if isinstance(x, str) else x
        self.df["power"]=self.df["power"].apply(power_bhp)
        self.df["power"]=pd.to_numeric(self.df["power"].astype(float).values, errors='coerce')
        brand_means_power = self.df.groupby('brand')['power'].mean()
        self.df['power'].fillna(self.df['brand'].map(brand_means_power), inplace=True)
        self.df["power"].fillna(self.df["power"].mean(), inplace=True)
        return self.df
    
    def extract_cc_column(self):
        def extract_cc(value_list):
            """
            Extracts CC value from a list of tokens.
            Returns integer CC value if found, else None.
            """
            if not isinstance(value_list, list):
                return None
            
            for token in value_list:
                match = re.search(r'(\d{2,4})\s*Cc', token, re.IGNORECASE)
                if match:
                    return int(match.group(1))
                
                # Some entries may just have '220' without 'Cc'
                if token.isdigit() and 50 < int(token) < 1000:  # simple sanity check
                    return int(token)
            
            return None
        cc=lambda x: extract_cc(x.split(" "))
        self.df["cc"] = self.df["model_name"].apply(cc)
        brand_means_cc = self.df.groupby('brand')['cc'].mean()
        self.df['cc'].fillna(self.df['brand'].map(brand_means_cc), inplace=True)
        self.df["cc"].fillna(self.df["cc"].mean(), inplace=True)
        return self.df
    
    def kms_driven_clean(self):
        self.df["kms_driven"].str.title()
        self.df["kms_driven"].str.strip()
        kms_value=lambda x: re.findall(r'-?\d+\.?\d*', x)
        self.df["kms_driven"].apply(kms_value)
        self.df["kms_driven"]=self.df["kms_driven"].apply(kms_value)
        def clean_kms(x):
            if isinstance(x, list) and len(x) == 1:
                try:
                    return float(x[0])
                except ValueError:
                    return np.nan
            elif isinstance(x, str):
                try:
                    return float(x)
                except ValueError:
                    return np.nan
            else:
                return np.nan

        self.df["kms_driven"] = self.df["kms_driven"].apply(clean_kms)
        brand_means_kms = self.df.groupby('brand')['kms_driven'].mean()
        self.df['kms_driven'].fillna(self.df['brand'].map(brand_means_kms), inplace=True)
        return self.df
    
    def textformat(self):
        self.df[["model_name", "brand", "location"]]=self.df[["model_name", "brand", "location"]].apply(lambda x: x.str.title() if x.dtype == "object" else x)
        return self.df
    
    def owner_prperocess(self):
        self.df["owner"] = self.df["owner"].str.strip()
        #self.df["owner"] = self.df["owner"].apply(lambda x: x.str.title() if isinstance(x, str) else x)
        self.df["owner"].replace({"first owner": 1, "second owner": 2, "third owner": 3, "fourth owner or more": 4}, inplace=True)
        self.df["owner"] = self.df["owner"].replace({"First Owner": 1, "Second Owner": 2, "Third Owner": 3, "Fourth & Above Owner": 4})
        #self.df["owner"] = pd.to_numeric(self.df["owner"], errors='coerce')
        self.df["owner"].fillna(1, inplace=True)
        return self.df
    
    def location_clean(self):
        self.df["location"].str.title()
        self.df["location"].str.strip()
        self.df["location"].fillna("Not Disclosed", inplace=True)
        return self.df



# Processing functions for the dataset
    def preprocess_dataset(self):
        try:
            self.bike_preprocess_mileage()
        except Exception as e:
            print(f"Error in bike_preprocess_mileage: {e}")
        try:
            self.power_column()
        except Exception as e:
            print(f"Error in power_column: {e}")
        try:
            self.extract_cc_column()
        except Exception as e:
            print(f"Error in extract_cc_column: {e}")
        try:
            self.kms_driven_clean()
        except Exception as e:
            print(f"Error in kms_driven_clean: {e}")
        try:
            self.owner_prperocess()
        except Exception as e:
            print(f"Error in owner_prperocess: {e}")
        try:
            self.textformat()
        except Exception as e:
            print(f"Error in textformat: {e}")
        try:
            self.location_clean()
        except Exception as e:
            print(f"Error in location_clean: {e}")
        return self.df

    


class ML_scale_tranfsormed:
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
    
        self.categorical_cols = ['model_name', 'location', 'brand']
        self.numerical_cols_x= ["kms_driven", "mileage", "power", "cc"]
        self.y_col = "price"


    def scaler_tranformer_load(self):
        with open("categorical_encoders.pkl", "rb") as f:
            categorical_scaler = joblib.load(f)
        with open("x_num.pkl", "rb") as f:
            num_scaler = joblib.load(f)
        with open("y_num.pkl", "rb") as f:
            y_scaler = joblib.load(f)
        return categorical_scaler, num_scaler, y_scaler
    
    def transform_data_x(self):
        new_df=pd.DataFrame()
        # loading all the scalers
        categorical_scaler, num_scaler, _ = self.scaler_tranformer_load()
        for i in self.categorical_cols:
            new_df[i] = categorical_scaler[i].transform(self.df[i])
        new_df[self.numerical_cols_x] = num_scaler.transform(self.df[self.numerical_cols_x])
        new_df[["model_year","owner"]] = self.df[["model_year","owner"]]
        return new_df
    


    def inverse_transform_data_y(self):
        # Load only the y scaler
        _, _, self.y_scaler = self.scaler_tranformer_load()
        
        # Transform y and flatten to 1D
        transformed_y = self.y_scaler.inverse_transform(
            self.df[self.y_col].values.reshape(-1, 1)
        ).ravel()
        original_price = np.expm1(transformed_y)

        
        # Assign into a new DataFrame
        new_df = pd.DataFrame({self.y_col: original_price})
        return new_df
    

    def transform_data_freq(self):
        with open("categorical_encoders_freq.pkl", "rb") as f:
            freq = pickle.load(f)
            for i in self.categorical_cols:
                self.df[i]=self.df[i].map(freq[i])

        return self.df




class pipelines_ml:
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        def open_files(path):
            with open(path, "rb") as f:
                return pickle.load(f)

        self.cluster = open_files("saved_models/Cluster_predictor.pkl")
        self.model_outlier = open_files("saved_models/Catboost_model_outlier.pkl")
        self.model = open_files("saved_models/Catboost_model.pkl")
        self.df_trans = ML_scale_tranfsormed(self.df).transform_data_freq()
        if "Unnamed: 0" in self.df_trans.columns:
            self.df_trans.drop("Unnamed: 0", axis=1, inplace=True)
        #self.categorical_cols = ['model_name', 'location', 'brand']
        #self.numerical_cols_x = ["kms_driven", "mileage", "power", "cc"]
        self.df_trans["Cluster"]=self.cluster.predict(self.df_trans)
        self.final_cluster = self.df_trans[self.df_trans["Cluster"]!=-1][self.df_trans.columns]
        self.final_cluster_outlier = self.df_trans[self.df_trans["Cluster"]==-1][self.df_trans.columns]

    def predict(self):
        if self.final_cluster_outlier.shape[0]==0:
            self.final_cluster["price"] = np.expm1(self.model.predict(self.final_cluster))
            return self.final_cluster
        else:
            #np.expm1(y_bi)
            self.final_cluster["price"] = np.expm1(self.model.predict(self.final_cluster))
            self.final_cluster_outlier["price"] = np.expm1(self.model_outlier.predict(self.final_cluster_outlier))
            self.final_output_x = pd.concat([self.final_cluster, self.final_cluster_outlier], axis=0)
            self.df["price"] = self.final_output_x["price"]
            if "Unnamed: 0" in self.df.columns:
                self.df.drop("Unnamed: 0", axis=1, inplace=True)
            self.final_output = self.df
            return self.final_output