import sys
import time
from functools import wraps
from collections import Counter
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix,
    precision_score,
    recall_score,
    accuracy_score,
    roc_auc_score,
    roc_curve,
)
from eqc_models.ml.classifierqboost import QBoostClassifier
from xgboost import XGBClassifier

# Some parameters
INP_FILE = "trn_dfOfCreditCard.csv"
TEST_SIZE = 0.2
TRAIN_DATA_COUNT_LIST = [
    1000,
    2000,
    5000,
    10000,
    20000,
    50000,
    100000,
    150000,
]
NUM_FEA_LIST = [5, 10, 20, 30, 38]
MINOR_CLASS_COUNT = 568
NUM_ITERS = 10

OUT_FILE1 = "kaggle_hamiltonian_times_data_count.csv"
OUT_FILE2 = "kaggle_hamiltonian_times_num_features.csv"


def timer(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        beg_time = time.time()
        val = func(*args, **kwargs)
        end_time = time.time()
        tot_time = end_time - beg_time

        print(
            "Runtime of %s: %0.2f seconds!"
            % (
                func.__name__,
                tot_time,
            )
        )

        return val

    return wrapper


def prep_data(df, train_data_count, num_features):
    data_count = int(train_data_count / (1.0 - TEST_SIZE)) + 1

    assert data_count > MINOR_CLASS_COUNT

    major_class_count = data_count - MINOR_CLASS_COUNT

    df0 = df[df["Class"] == 0].sample(major_class_count)
    df1 = df[df["Class"] == 1]

    df = pd.concat([df0, df1]).sample(frac=1.0)

    df["Class"] = df["Class"].apply(lambda x: -1 if x == 0 else 1)

    fea_names = [
        "V1",
        "V2",
        "V3",
        "V4",
        "V5",
        "V6",
        "V7",
        "V8",
        "V9",
        "V10",
        "V11",
        "V12",
        "V13",
        "V14",
        "V15",
        "V16",
        "V17",
        "V18",
        "V19",
        "V20",
        "V21",
        "V22",
        "V23",
        "V24",
        "V25",
        "V26",
        "V27",
        "V28",
        "V_Sum",
        "V_Min",
        "V_Max",
        "V_Avg",
        "V_Std",
        "V_Pos",
        "V_Neg",
        "V_Var",
        "Amount",
        "Time",
    ]

    if num_features is not None:
        fea_names = fea_names[:num_features]

    X = np.array(df[fea_names])
    y = np.array(df["Class"])

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=42,
    )

    return X_train, y_train, X_test, y_test


def calc_precision_recall(
    y_train, y_train_prd, y_test, y_test_prd, model_name
):
    train_precision = precision_score(
        y_train, y_train_prd, labels=[-1, 1], pos_label=1
    )
    train_recall = recall_score(
        y_train, y_train_prd, labels=[-1, 1], pos_label=1
    )
    train_f1 = (
        2.0
        * (train_precision * train_recall)
        / (train_precision + train_recall)
    )
    train_accuracy = accuracy_score(y_train, y_train_prd)

    print("%s: Train precision: %0.4f" % (model_name, train_precision))
    print("%s: Train recall: %0.4f" % (model_name, train_recall))
    print("%s: Train F1 score: %0.4f" % (model_name, train_f1))
    print("%s: Train accuracy: %0.4f" % (model_name, train_accuracy))

    test_precision = precision_score(
        y_test, y_test_prd, labels=[-1, 1], pos_label=1
    )
    test_recall = recall_score(
        y_test, y_test_prd, labels=[-1, 1], pos_label=1
    )
    test_f1 = (
        2.0
        * (test_precision * test_recall)
        / (test_precision + test_recall)
    )
    test_accuracy = accuracy_score(y_test, y_test_prd)

    print("%s: Test precision: %0.4f" % (model_name, test_precision))
    print("%s: Test recall: %0.4f" % (model_name, test_recall))
    print("%s: Test F1 score: %0.4f" % (model_name, test_f1))
    print("%s: Test accuracy %0.4f" % (model_name, test_accuracy))

    return test_precision, test_recall, test_f1, test_accuracy


@timer
def run_qboost(X_train, y_train, X_test, y_test):
    obj = QBoostClassifier(
        relaxation_schedule=1,
        num_samples=1,
        lambda_coef=0,
        weak_cls_schedule=1,
        weak_cls_type="lg",
        weak_max_depth=5,
        weak_min_samples_split=50,
        weak_cls_strategy="sequential",
    )

    # Train
    beg_time = time.time()
    resp = obj.fit(X_train, y_train)
    end_time = time.time()

    total_train_time = end_time - beg_time

    y_train_prd = obj.predict(X_train)
    y_test_prd = obj.predict(X_test)

    (
        test_precision,
        test_recall,
        test_f1,
        test_accuracy,
    ) = calc_precision_recall(
        y_train,
        y_train_prd,
        y_test,
        y_test_prd,
        "CVQBoost",
    )

    # Get dirac3 and queue times
    job_resp = resp["job_info"]["job_status"]
    submit1 = job_resp["submitted_at_rfc3339nano"]
    queue1 = job_resp["queued_at_rfc3339nano"]
    run1 = job_resp["running_at_rfc3339nano"]
    comp1 = job_resp["completed_at_rfc3339nano"]

    #dirac3_time = (pd.to_datetime(comp1) - pd.to_datetime(run1)).seconds
    queue_time = (pd.to_datetime(run1) - pd.to_datetime(queue1)).seconds
    dirac3_time = (pd.to_datetime(comp1) - pd.to_datetime(submit1)).seconds
    dirac3_time = dirac3_time - queue_time

    # Adjust total train time
    total_train_time = total_train_time - queue_time

    # Calculate the AUC-ROC score
    y_test_probs = obj.predict_raw(X_test)

    for i in range(len(y_test_probs)):
        y_test_probs[i] = 0.5 * (y_test_probs[i] + 1.0)

    test_roc_auc = roc_auc_score(y_test, y_test_probs)

    print("Test AUC: %0.6f" % test_roc_auc)

    return (
        test_precision,
        test_recall,
        test_f1,
        test_accuracy,
        test_roc_auc,
        total_train_time,
        dirac3_time,
    )


@timer
def run_xgboost(X_train, y_train, X_test, y_test):
    xgb_params = {
        "n_estimators": 3093,
        "min_child_weight": 96,
        "max_depth": 12,
        "learning_rate": 0.07516,
        "subsample": 0.95,
        "colsample_bytree": 0.95,
        "reg_lambda": 1.50,
        "reg_alpha": 1.50,
        "gamma": 1.50,
        "max_bin": 512,
        "random_state": 228,
        "objective": "binary:logistic",
        "tree_method": "auto",
        "eval_metric": "auc",
    }
    xgb_model = XGBClassifier(**xgb_params)

    beg_time = time.time()
    xgb_model.fit(X_train, y_train)
    end_time = time.time()

    total_train_time = end_time - beg_time

    y_train_prd = xgb_model.predict(X_train)
    y_test_prd = xgb_model.predict(X_test)

    (
        test_precision,
        test_recall,
        test_f1,
        test_accuracy,
    ) = calc_precision_recall(
        y_train,
        y_train_prd,
        y_test,
        y_test_prd,
        "XGBoost",
    )

    # Calculate the AUC-ROC score
    y_test_probs = xgb_model.predict_proba(X_test)[:, 1]

    test_roc_auc = roc_auc_score(y_test, y_test_probs)

    print("Test AUC: %0.6f" % test_roc_auc)

    return (
        test_precision,
        test_recall,
        test_f1,
        test_accuracy,
        test_roc_auc,
        total_train_time,
    )


def run(train_data_count, num_features, iteration):
    stats_hash = {}

    # Read data
    df = pd.read_csv(INP_FILE)

    # Prep data
    X_train, y_train, X_test, y_test = prep_data(
        df, train_data_count, num_features
    )

    # Train and test QBoost
    (
        test_precision,
        test_recall,
        test_f1,
        test_accuracy,
        test_roc_auc,
        total_train_time,
        dirac3_time,
    ) = run_qboost(X_train, y_train, X_test, y_test)

    stats_hash["cvqboost precision"] = [test_precision]
    stats_hash["cvqboost recall"] = [test_recall]
    stats_hash["cvqboost f1"] = [test_f1]
    stats_hash["cvqboost accuracy"] = [test_accuracy]
    stats_hash["cvqboost auc"] = [test_roc_auc]
    stats_hash["cvqboost total train time"] = [total_train_time]
    stats_hash["cvqboost dirac3 time"] = [dirac3_time]

    # CVQBoost on classcial solvers
    for solver in ["scipy", "hexaly"]:
        (
            test_precision,
            test_recall,
            test_f1,
            test_accuracy,
            test_roc_auc,
            total_train_time,
        ) = run_qboost_classical(X_train, y_train, X_test, y_test, solver)

        stats_hash["cvqboost-%s precision" % solver] = [test_precision]
        stats_hash["cvqboost-%s recall" % solver] = [test_recall]
        stats_hash["cvqboost-%s f1" % solver] = [test_f1]
        stats_hash["cvqboost-%s accuracy" % solver] = [test_accuracy]
        stats_hash["cvqboost-%s auc" % solver] = [test_roc_auc]
        stats_hash["cvqboost-%s total train time" % solver] = [
            total_train_time
        ]
        
    # Train and test XGBoost
    for i in range(len(y_train)):
        y_train[i] = 0.5 * (y_train[i] + 1)

    for i in range(len(y_test)):
        y_test[i] = 0.5 * (y_test[i] + 1)

    (
        test_precision,
        test_recall,
        test_f1,
        test_accuracy,
        test_roc_auc,
        total_train_time,
    ) = run_xgboost(X_train, y_train, X_test, y_test)

    stats_hash["xgboost precision"] = [test_precision]
    stats_hash["xgboost recall"] = [test_recall]
    stats_hash["xgboost f1"] = [test_f1]
    stats_hash["xgboost accuracy"] = [test_accuracy]
    stats_hash["xgboost auc"] = [test_roc_auc]
    stats_hash["xgboost total train time"] = [total_train_time]

    stats_df = pd.DataFrame(stats_hash)

    stats_df["train data count"] = train_data_count
    stats_df["num features"] = num_features

    return stats_df

if __name__ == "__main__":
    beg_flag = True
    for train_data_count in TRAIN_DATA_COUNT_LIST:
        for itr in range(NUM_ITERS):
            iteration = itr + 1
            stats_df = run(
                train_data_count, 38, iteration
            )
            stats_df["iteration"] = iteration

            if beg_flag:
                stats_df.to_csv(OUT_FILE1, index=False)
            else:
                stats_df.to_csv(
                    OUT_FILE1, index=False, mode="a", header=False
                )

            beg_flag = False

    beg_flag = True
    for num_features in NUM_FEA_LIST:
        for itr in range(NUM_ITERS):
            iteration = itr + 1
            stats_df = run(
                50000, num_features, iteration
            )
            stats_df["iteration"] = iteration

            if beg_flag:
                stats_df.to_csv(OUT_FILE2, index=False)
            else:
                stats_df.to_csv(
                    OUT_FILE2, index=False, mode="a", header=False
                )

            beg_flag = False
