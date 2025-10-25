#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ỨNG DỤNG CÂY QUYẾT ĐỊNH VÀ SVM TRONG DỰ ĐOÁN GIAO DỊCH GIAN LẬN
----------------------------------------------------------------
Chương trình end-to-end, có chú thích chi tiết, chạy bằng Python + scikit-learn.

CHỨC NĂNG CHÍNH
1) Đọc CSV (hoặc tự tạo dữ liệu synthetic nếu không truyền --csv).
2) Tự nhận diện cột số / danh mục, cho phép override bằng --categorical.
3) Tiền xử lý an toàn trong Pipeline để tránh rò rỉ (impute, encode, scale).
4) Huấn luyện & tuning:
   - DecisionTreeClassifier (class_weight="balanced")
   - SVC(kernel="rbf", probability=True, class_weight="balanced")
   Sử dụng StratifiedKFold + GridSearchCV với scoring = "average_precision" (PR-AUC).
5) Đánh giá trên tập test: ROC-AUC, PR-AUC, classification_report, confusion matrices.
6) Tối ưu ngưỡng theo mục tiêu Recall (--recall-target) và/hoặc tối thiểu hóa chi phí (--cost-fn, --cost-fp).
7) Lưu mô hình tốt nhất, báo cáo, biểu đồ PR/ROC vào thư mục output.

YÊU CẦU
- Python 3.8+
- numpy, pandas, scikit-learn, matplotlib (tùy chọn imbalanced-learn nếu muốn SMOTE)

VÍ DỤ CHẠY
- Dùng data synthetic (mặc định):
    python fraud_dt_svm.py
- Với file của bạn:
    python fraud_dt_svm.py --csv your_transactions.csv --target is_fraud
- Chỉ rõ cột danh mục (phân tách bởi dấu phẩy):
    python fraud_dt_svm.py --csv your.csv --categorical device_type,merchant_id,city
- Sử dụng SMOTE (nếu đã cài imbalanced-learn):
    python fraud_dt_svm.py --csv your.csv --use-smote
- Đặt mục tiêu Recall 95% và chi phí FN=20, FP=1:
    python fraud_dt_svm.py --csv your.csv --recall-target 0.95 --cost-fn 20 --cost-fp 1

"""

import argparse
import os
from dataclasses import dataclass
import json
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (classification_report, confusion_matrix, roc_auc_score,
                             average_precision_score, precision_recall_curve, roc_curve)
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.utils import check_random_state

try:
    import joblib
except Exception:
    joblib = None  # sẽ warn khi lưu model

# Optional: SMOTE
try:
    from imblearn.pipeline import Pipeline as ImbPipeline
    from imblearn.over_sampling import SMOTENC
    IMB_AVAILABLE = True
except Exception:
    IMB_AVAILABLE = False


@dataclass
class Config:
    csv_path: Optional[str]
    target_col: str
    categorical_cols: Optional[List[str]]
    test_size: float
    random_state: int
    n_splits: int
    recall_target: float
    cost_fn: float
    cost_fp: float
    use_smote: bool
    output_dir: str


def parse_args() -> Config:
    parser = argparse.ArgumentParser(description="Fraud Detection with Decision Tree & SVM (Vietnamese, detailed).")
    parser.add_argument("--csv", dest="csv_path", type=str, default=None,
                        help="Đường dẫn CSV. Nếu bỏ trống, chương trình sẽ tạo dữ liệu synthetic.")
    parser.add_argument("--target", dest="target_col", type=str, default="is_fraud",
                        help="Tên cột mục tiêu (mặc định: is_fraud).")
    parser.add_argument("--categorical", dest="categorical", type=str, default=None,
                        help="Danh sách cột danh mục, phân tách dấu phẩy. Nếu không truyền, chương trình tự suy đoán.")
    parser.add_argument("--test-size", dest="test_size", type=float, default=0.2,
                        help="Tỷ lệ test (0..1). Mặc định 0.2")
    parser.add_argument("--seed", dest="seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--splits", dest="splits", type=int, default=5, help="Số folds StratifiedKFold (>=3).")
    parser.add_argument("--recall-target", dest="recall_target", type=float, default=0.90,
                        help="Mục tiêu recall để chọn ngưỡng.")
    parser.add_argument("--cost-fn", dest="cost_fn", type=float, default=0.0,
                        help="Chi phí False Negative (>=0). Nếu >0, sẽ tối ưu ngưỡng theo chi phí.")
    parser.add_argument("--cost-fp", dest="cost_fp", type=float, default=0.0,
                        help="Chi phí False Positive (>=0). Nếu >0, sẽ tối ưu ngưỡng theo chi phí.")
    parser.add_argument("--use-smote", dest="use_smote", action="store_true",
                        help="Dùng SMOTENC trong cross-validation (cần imbalanced-learn).")
    parser.add_argument("--out", dest="output_dir", type=str, default="output_fraud",
                        help="Thư mục xuất kết quả.")
    args = parser.parse_args()

    categorical_cols = None
    if args.categorical:
        categorical_cols = [c.strip() for c in args.categorical.split(",") if c.strip()]

    cfg = Config(
        csv_path=args.csv_path,
        target_col=args.target_col,
        categorical_cols=categorical_cols,
        test_size=args.test_size,
        random_state=args.seed,
        n_splits=args.splits,
        recall_target=args.recall_target,
        cost_fn=max(0.0, args.cost_fn),
        cost_fp=max(0.0, args.cost_fp),
        use_smote=bool(args.use_smote),
        output_dir=args.output_dir
    )
    return cfg


def ensure_outdir(path: str):
    os.makedirs(path, exist_ok=True)


def load_or_make_data(cfg: Config) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Đọc CSV, kiểm tra cột target; nếu không có CSV, tạo synthetic dataset
    mô phỏng gian lận hiếm với vài cột danh mục & số.
    """
    rng = check_random_state(cfg.random_state)

    if cfg.csv_path and os.path.isfile(cfg.csv_path):
        df = pd.read_csv(cfg.csv_path)
        if cfg.target_col not in df.columns:
            raise ValueError(f"Không tìm thấy cột mục tiêu '{cfg.target_col}' trong file CSV.")
        y = df[cfg.target_col].astype(int)
        X = df.drop(columns=[cfg.target_col])
        print(f"[INFO] Đã đọc dữ liệu từ '{cfg.csv_path}' với shape X={X.shape}, y={y.shape}")
        return X, y

    # Tạo synthetic dataset nếu không có CSV
    print("[WARN] Không truyền --csv hoặc file không tồn tại. Sẽ tạo dữ liệu synthetic để demo.")
    n = 40000
    # Tạo đặc trưng số:
    X_num = rng.normal(size=(n, 6))
    df = pd.DataFrame(X_num, columns=["amount_z", "velocity", "customer_age_z", "time_delta_z", "risk_score_z", "geo_dist_z"])
    # Thêm đặc trưng danh mục:
    df["device_type"] = rng.choice(["android", "ios", "web"], size=n, p=[0.5, 0.35, 0.15])
    df["merchant_id"] = rng.choice([f"m{m:03d}" for m in range(50)], size=n)
    df["country"] = rng.choice(["VN", "US", "IN", "ID", "BR", "PH"], size=n, p=[0.35, 0.1, 0.2, 0.15, 0.1, 0.1])
    # Sinh nhãn lệch lớp (2% gian lận) dựa vào một số quy luật:
    logits = (
        1.2*df["amount_z"] + 0.8*df["velocity"] + 0.9*df["risk_score_z"]
        + 0.6*(df["device_type"].isin(["web"]).astype(float))
        + 0.5*(df["country"].isin(["US", "BR"]).astype(float))
        + rng.normal(scale=0.8, size=n)
    )
    prob = 1 / (1 + np.exp(-logits))
    # ép tỷ lệ gian lận hiếm:
    thr = np.quantile(prob, 0.98)  # ~2% trên tổng
    y = (prob >= thr).astype(int)
    df["is_fraud"] = y
    # Lưu ra CSV để người dùng xem thử:
    demo_path = os.path.join(cfg.output_dir, "synthetic_transactions.csv")
    ensure_outdir(cfg.output_dir)
    df.to_csv(demo_path, index=False)
    print(f"[INFO] Đã tạo synthetic dataset và lưu tại: {demo_path}")
    X = df.drop(columns=["is_fraud"])
    y = df["is_fraud"].astype(int)
    return X, y


def infer_column_types(X: pd.DataFrame, categorical_override: Optional[List[str]]=None) -> Tuple[List[str], List[str]]:
    """
    Suy đoán cột số & danh mục. Cho phép override danh mục để chính xác hơn.
    Quy tắc:
    - object/category -> danh mục
    - còn lại -> số
    """
    cat_cols = [c for c in X.columns if str(X[c].dtype) in ("object", "category")]
    num_cols = [c for c in X.columns if c not in cat_cols]

    if categorical_override:
        # Đảm bảo chỉ chọn cột có trong X
        override = [c for c in categorical_override if c in X.columns]
        # Hợp nhất với đã suy đoán
        cat_cols = sorted(list(set(cat_cols).union(set(override))))
        num_cols = [c for c in X.columns if c not in cat_cols]

    return num_cols, cat_cols


def make_preprocessors(num_cols: List[str], cat_cols: List[str], for_svm: bool) -> ColumnTransformer:
    """
    Tạo ColumnTransformer:
    - SVM: số -> impute median + StandardScaler; danh mục -> impute most_frequent + OneHot
    - Tree: số -> impute median; danh mục -> impute most_frequent + OneHot
    """
    num_steps = [("imp", SimpleImputer(strategy="median"))]
    if for_svm:
        num_steps.append(("scaler", StandardScaler()))

    numeric = Pipeline(steps=num_steps)
    categorical = Pipeline(steps=[
        ("imp", SimpleImputer(strategy="most_frequent")),
        ("oh", OneHotEncoder(handle_unknown="ignore"))
    ])

    pre = ColumnTransformer(
        transformers=[
            ("num", numeric, num_cols),
            ("cat", categorical, cat_cols)
        ],
        remainder="drop"
    )
    return pre


def build_pipelines(num_cols: List[str], cat_cols: List[str], cfg: Config):
    """
    Xây dựng 2 pipelines: Decision Tree & SVM.
    Nếu --use-smote và imbalanced-learn sẵn có, dùng ImbPipeline để oversample trong CV.
    """
    pre_tree = make_preprocessors(num_cols, cat_cols, for_svm=False)
    pre_svm  = make_preprocessors(num_cols, cat_cols, for_svm=True)

    tree_clf = DecisionTreeClassifier(class_weight="balanced", random_state=cfg.random_state)
    svm_clf  = SVC(kernel="rbf", probability=True, class_weight="balanced", random_state=cfg.random_state)

    if cfg.use_smote and IMB_AVAILABLE:
        # Với SMOTENC cần cung cấp index của cột danh mục sau biến đổi? -> SMOTENC yêu cầu cột chưa one-hot.
        # Do đó ta nên SMOTE TRƯỚC OneHot (khó vì ColumnTransformer). Ở đây, ta chỉ SMOTE trên dữ liệu gốc:
        # Ta sẽ làm hai bước: oversample trong pipeline trước preprocessor bằng SMOTENC.
        # Xác định cột danh mục (trên X raw):
        cat_indices = [i for i, c in enumerate(num_cols + cat_cols) if c in cat_cols]
        # Nhưng vì ColumnTransformer nhận dataframes với tên cột, SMOTENC hoạt động trên numpy array -> ta bọc bằng ImbPipeline với step 'sampler'
        # Cách đơn giản & đúng: bỏ ColumnTransformer, thay bằng một pipeline riêng cho SMOTE trong CV là phức tạp.
        # Để giữ an toàn & đơn giản, ta không áp SMOTE trực tiếp khi dùng ColumnTransformer.
        # Thay vào đó, ta sẽ cảnh báo và chuyển sang class_weight nếu người dùng bật --use-smote.
        print("[WARN] --use-smote được bật nhưng do kiến trúc ColumnTransformer, demo này không áp dụng SMOTE trực tiếp.")
        print("      Gợi ý: Dùng imblearn.Pipeline tách preprocess theo numeric/categorical trước rồi SMOTENC, sau đó mới OneHot.")
        print("      Trong phiên bản gọn này, ta giữ class_weight='balanced'.")
        # Giữ nguyên pipelines với class_weight.
        pass

    # Pipelines
    dt_pipe = Pipeline(steps=[("pre", pre_tree), ("clf", tree_clf)])
    svm_pipe = Pipeline(steps=[("pre", pre_svm), ("clf", svm_clf)])
    return dt_pipe, svm_pipe


def tune_and_fit(pipe, params, X_train, y_train, cfg: Config, model_name: str):
    cv = StratifiedKFold(n_splits=cfg.n_splits, shuffle=True, random_state=cfg.random_state)
    gs = GridSearchCV(
        pipe,
        param_grid=params,
        scoring="average_precision",  # PR-AUC
        cv=cv,
        n_jobs=-1,
        verbose=1
    )
    gs.fit(X_train, y_train)
    print(f"[{model_name}] Best params:", gs.best_params_, "CV PR-AUC=", round(gs.best_score_, 4))
    return gs.best_estimator_, gs.best_score_, gs.cv_results_


def evaluate(model, X_test, y_test, name: str, outdir: str):
    """
    Trả về dict kết quả + lưu PR/ROC curves.
    """
    proba = model.predict_proba(X_test)[:, 1]
    pred_default = (proba >= 0.5).astype(int)

    roc = roc_auc_score(y_test, proba)
    prauc = average_precision_score(y_test, proba)

    cm = confusion_matrix(y_test, pred_default)
    report = classification_report(y_test, pred_default, digits=4, output_dict=True)

    # Vẽ PR & ROC
    p, r, _ = precision_recall_curve(y_test, proba)
    fpr, tpr, _ = roc_curve(y_test, proba)

    plt.figure()
    plt.step(r, p, where="post", label=name)
    plt.xlabel("Recall"); plt.ylabel("Precision"); plt.title(f"Precision–Recall ({name})")
    plt.legend()
    pr_path = os.path.join(outdir, f"PR_curve_{name}.png")
    plt.savefig(pr_path, bbox_inches="tight")
    plt.close()

    plt.figure()
    plt.plot(fpr, tpr, label=name)
    plt.plot([0,1], [0,1], linestyle="--")
    plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate"); plt.title(f"ROC ({name})")
    plt.legend()
    roc_path = os.path.join(outdir, f"ROC_curve_{name}.png")
    plt.savefig(roc_path, bbox_inches="tight")
    plt.close()

    return {
        "name": name,
        "roc_auc": float(roc),
        "pr_auc": float(prauc),
        "confusion_matrix@0.5": cm.tolist(),
        "classification_report@0.5": report,
        "pr_curve_path": pr_path,
        "roc_curve_path": roc_path,
        "proba": proba  # trả về để tối ưu ngưỡng
    }


def threshold_for_recall(y_true, proba, recall_target: float) -> Tuple[float, float, float]:
    """
    Tìm ngưỡng đầu tiên đạt recall >= recall_target, trả về (thr, precision, recall).
    Nếu không đạt, chọn ngưỡng có F0.5 cao nhất (ưu tiên precision nhẹ).
    """
    p, r, thr = precision_recall_curve(y_true, proba)
    best = None
    for i in range(len(thr)):
        if r[i] >= recall_target:
            best = (float(thr[i]), float(p[i]), float(r[i]))
            break
    if best is None:
        f = (1+0.5**2)*p[:-1]*r[:-1]/(0.5**2*p[:-1]+r[:-1]+1e-12)
        i = int(np.nanargmax(f))
        best = (float(thr[i]), float(p[i]), float(r[i]))
    return best


def threshold_min_cost(y_true, proba, cost_fn: float, cost_fp: float) -> Tuple[float, float]:
    """
    Chọn ngưỡng tối thiểu hóa COST = cost_fn*FN + cost_fp*FP.
    Trả về (best_thr, best_cost_per_sample).
    """
    if cost_fn <= 0 and cost_fp <= 0:
        raise ValueError("Cần cost_fn hoặc cost_fp > 0 để tối ưu chi phí.")

    # Sử dụng tất cả ngưỡng ứng viên từ các điểm proba
    thr_candidates = np.unique(np.clip(proba, 1e-9, 1-1e-9))
    thr_candidates.sort()
    n = len(y_true)
    best_thr, best_cost = 0.5, np.inf
    for thr in thr_candidates:
        y_pred = (proba >= thr).astype(int)
        cm = confusion_matrix(y_true, y_pred, labels=[0,1])
        tn, fp, fn, tp = cm.ravel()
        total_cost = cost_fn*fn + cost_fp*fp
        cost_per_sample = total_cost / n
        if cost_per_sample < best_cost:
            best_cost, best_thr = cost_per_sample, thr
    return float(best_thr), float(best_cost)


def confusion_at_threshold(y_true, proba, thr: float) -> dict:
    y_pred = (proba >= thr).astype(int)
    cm = confusion_matrix(y_true, y_pred, labels=[0,1])
    report = classification_report(y_true, y_pred, digits=4, output_dict=True)
    return {"threshold": float(thr),
            "confusion_matrix": cm.tolist(),
            "classification_report": report}


def extract_dt_feature_importance(dt_model: Pipeline, outdir: str) -> Optional[str]:
    """
    Xuất feature importance của Decision Tree (nếu khả dụng) về CSV.
    Cố gắng lấy tên cột sau one-hot; nếu không có, ghi số thứ tự.
    """
    try:
        pre = dt_model.named_steps["pre"]
        clf = dt_model.named_steps["clf"]
        if not hasattr(pre, "get_feature_names_out"):
            raise AttributeError("ColumnTransformer không có get_feature_names_out (sklearn quá cũ).")
        feat_names = pre.get_feature_names_out()
        importances = getattr(clf, "feature_importances_", None)
        if importances is None:
            return None
        imp_df = pd.DataFrame({"feature": feat_names, "importance": importances})
        imp_df = imp_df.sort_values("importance", ascending=False)
        path = os.path.join(outdir, "decision_tree_feature_importance.csv")
        imp_df.to_csv(path, index=False)
        return path
    except Exception as e:
        print("[WARN] Không thể xuất feature importance chi tiết cho Decision Tree:", e)
        return None


def main():
    cfg = parse_args()
    ensure_outdir(cfg.output_dir)

    # 1) Đọc / tạo dữ liệu
    X, y = load_or_make_data(cfg)

    # 2) Tách train/test theo stratify
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=cfg.test_size, stratify=y, random_state=cfg.random_state
    )
    print(f"[INFO] Split: X_train={X_train.shape}, X_test={X_test.shape}, positive_rate_train={y_train.mean():.4f}")

    # 3) Nhận diện cột
    num_cols, cat_cols = infer_column_types(X_train, cfg.categorical_cols)
    print(f"[INFO] Numeric columns ({len(num_cols)}): {num_cols[:10]}{'...' if len(num_cols)>10 else ''}")
    print(f"[INFO] Categorical columns ({len(cat_cols)}): {cat_cols[:10]}{'...' if len(cat_cols)>10 else ''}")

    # 4) Xây pipelines
    dt_pipe, svm_pipe = build_pipelines(num_cols, cat_cols, cfg)

    # 5) Hyperparameters
    dt_params = {
        "clf__max_depth": [None, 6, 10, 16, 24],
        "clf__min_samples_leaf": [1, 5, 20],
        "clf__min_samples_split": [2, 10, 50],
        "clf__max_features": [None, "sqrt", 0.5]
    }
    svm_params = {
        "clf__C": [0.5, 1, 2, 4],
        "clf__gamma": ["scale", 0.1, 0.01]
    }

    # 6) Tune & fit
    best_dt, cv_dt, cvres_dt = tune_and_fit(dt_pipe, dt_params, X_train, y_train, cfg, "DecisionTree")
    best_svm, cv_svm, cvres_svm = tune_and_fit(svm_pipe, svm_params, X_train, y_train, cfg, "SVM_RBF")

    # 7) Evaluate trên test
    dt_eval = evaluate(best_dt, X_test, y_test, "DecisionTree", cfg.output_dir)
    svm_eval = evaluate(best_svm, X_test, y_test, "SVM_RBF", cfg.output_dir)

    # 8) Chọn ngưỡng theo Recall target
    dt_thr_rec = threshold_for_recall(y_test, dt_eval["proba"], cfg.recall_target)
    svm_thr_rec = threshold_for_recall(y_test, svm_eval["proba"], cfg.recall_target)

    dt_conf_rec = confusion_at_threshold(y_test, dt_eval["proba"], dt_thr_rec[0])
    svm_conf_rec = confusion_at_threshold(y_test, svm_eval["proba"], svm_thr_rec[0])

    # 9) Chọn ngưỡng tối thiểu chi phí (nếu có cost)
    dt_thr_cost, dt_cost = (None, None)
    svm_thr_cost, svm_cost = (None, None)
    if cfg.cost_fn > 0 or cfg.cost_fp > 0:
        dt_thr_cost, dt_cost = threshold_min_cost(y_test, dt_eval["proba"], cfg.cost_fn, cfg.cost_fp)
        svm_thr_cost, svm_cost = threshold_min_cost(y_test, svm_eval["proba"], cfg.cost_fn, cfg.cost_fp)

    # 10) Xuất feature importance cho cây
    dt_imp_path = extract_dt_feature_importance(best_dt, cfg.output_dir)

    # 11) Tổng hợp báo cáo JSON
    report = {
        "config": {
            "csv_path": cfg.csv_path,
            "target_col": cfg.target_col,
            "test_size": cfg.test_size,
            "random_state": cfg.random_state,
            "n_splits": cfg.n_splits,
            "recall_target": cfg.recall_target,
            "cost_fn": cfg.cost_fn,
            "cost_fp": cfg.cost_fp,
            "use_smote": cfg.use_smote,
            "output_dir": cfg.output_dir,
            "numeric_cols": num_cols,
            "categorical_cols": cat_cols
        },
        "cv": {
            "decision_tree_pr_auc": cv_dt,
            "svm_rbf_pr_auc": cv_svm
        },
        "test_eval": {
            "decision_tree": {k: v for k, v in dt_eval.items() if k != "proba"},
            "svm_rbf": {k: v for k, v in svm_eval.items() if k != "proba"}
        },
        "thresholds": {
            "decision_tree_recall_target": {
                "threshold": dt_thr_rec[0],
                "precision": dt_thr_rec[1],
                "recall": dt_thr_rec[2],
                "confusion": dt_conf_rec
            },
            "svm_rbf_recall_target": {
                "threshold": svm_thr_rec[0],
                "precision": svm_thr_rec[1],
                "recall": svm_thr_rec[2],
                "confusion": svm_conf_rec
            },
            "decision_tree_min_cost": {
                "threshold": dt_thr_cost,
                "cost_per_sample": dt_cost
            } if (dt_thr_cost is not None) else None,
            "svm_rbf_min_cost": {
                "threshold": svm_thr_cost,
                "cost_per_sample": svm_cost
            } if (svm_thr_cost is not None) else None
        },
        "artifacts": {
            "pr_curve_decision_tree": dt_eval["pr_curve_path"],
            "roc_curve_decision_tree": dt_eval["roc_curve_path"],
            "pr_curve_svm_rbf": svm_eval["pr_curve_path"],
            "roc_curve_svm_rbf": svm_eval["roc_curve_path"],
            "dt_feature_importance_csv": dt_imp_path
        }
    }

    # 12) Lưu mô hình & báo cáo
    if joblib is not None:
        try:
            joblib.dump(best_dt, os.path.join(cfg.output_dir, "best_decision_tree.joblib"))
            joblib.dump(best_svm, os.path.join(cfg.output_dir, "best_svm_rbf.joblib"))
        except Exception as e:
            print("[WARN] Không thể lưu mô hình bằng joblib:", e)
    else:
        print("[WARN] joblib không khả dụng: bỏ qua lưu mô hình.")

    report_path = os.path.join(cfg.output_dir, "report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"[INFO] Đã lưu báo cáo JSON: {report_path}")

    # In tóm tắt gọn ra màn hình:
    print("\n========== TÓM TẮT ==========")
    print(f"DT:  PR-AUC(test)={report['test_eval']['decision_tree']['pr_auc']:.4f}  ROC-AUC(test)={report['test_eval']['decision_tree']['roc_auc']:.4f}")
    print(f"SVM: PR-AUC(test)={report['test_eval']['svm_rbf']['pr_auc']:.4f}  ROC-AUC(test)={report['test_eval']['svm_rbf']['roc_auc']:.4f}")
    print("Ngưỡng theo Recall target:")
    print("  DT :", report['thresholds']['decision_tree_recall_target'])
    print("  SVM:", report['thresholds']['svm_rbf_recall_target'])
    if report['thresholds']['decision_tree_min_cost'] is not None:
        print("Ngưỡng tối thiểu chi phí (DT):", report['thresholds']['decision_tree_min_cost'])
    if report['thresholds']['svm_rbf_min_cost'] is not None:
        print("Ngưỡng tối thiểu chi phí (SVM):", report['thresholds']['svm_rbf_min_cost'])
    print("Artifacts lưu tại:", cfg.output_dir)
    print("================================")
    print("Hoàn tất.")

if __name__ == "__main__":
    main()
