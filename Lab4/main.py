from sklearn.model_selection import train_test_split
import pandas as pd
from utils import *

target_label = "cardio"

data = pd.read_csv("./data/cardio_train.csv", sep=';')
data.head()

data.drop(["id"], axis=1, inplace=True)

data["age"] = data["age"].apply(lambda x: "young" if x < 20 * 365 else "middle" if x < 60 * 365 else "old")
data["height"] = data["height"].apply(lambda x: "low" if x < 165 else "middle" if x < 190 else "tall")
data["weight"] = data["weight"].apply(lambda x: "skinny" if x < 60 else "middle" if x < 100 else "fat")
data["ap_hi"] = data["ap_hi"].apply(lambda x: "low" if x < 110 else "normal" if x < 130 else "high")
data["ap_lo"] = data["ap_lo"].apply(lambda x: "low" if x < 75 else "normal" if x < 85 else "high")

data.dropna(inplace=True)

y = data[target_label]
X = data.drop([target_label], axis=1)

X_train, X_val_test, y_train, y_val_test = train_test_split(X, y, train_size=0.75, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_val_test, y_val_test, test_size=0.5, random_state=42)

determine_the_best_model(target_label=target_label,
                         X_train=X_train,
                         y_train=y_train,
                         X_val=X_val,
                         y_val=y_val,
                         X_test=X_test,
                         y_test=y_test,
                         depth_range=10,
                         main_metric='f1_score',
                         print_model=False
                         )
