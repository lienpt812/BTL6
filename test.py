#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ỨNG DỤNG CÂY QUYẾT ĐỊNH VÀ SVM TRONG DỰ ĐOÁN GIAO DỊCH GIAN LẬN
(Chỉnh sửa để dùng CSV từ người dùng + SMOTE)
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

from sklearn import tree
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D

try:
    import joblib
except Exception:
    joblib = None  # sẽ warn khi lưu model

# Optional: SMOTE
try:
    from imblearn.pipeline import Pipeline as ImbPipeline
    from imblearn.over_sampling import SMOTE, SMOTENC
    IMB_AVAILABLE = True
except Exception:
    IMB_AVAILABLE = False


@dataclass
class Config:
    csv_path: str
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
    parser = argparse.ArgumentParser(description="Fraud Detection with Decision Tree & SVM.")
    parser.add_argument("--csv", dest="csv_path", type=str, required=True,
                        help="Đường dẫn CSV. Bắt buộc.")
    parser.add_argument("--target", dest="target_col", type=str, required=True,
                        help="Tên cột mục tiêu (ví dụ: Class).")
    parser.add_argument("--categorical", dest="categorical", type=str, default=None,
                        help="Danh sách cột danh mục, phân tách dấu phẩy.")
    parser.add_argument("--test-size", dest="test_size", type=float, default=0.2)
    parser.add_argument("--seed", dest="seed", type=int, default=42)
    parser.add_argument("--splits", dest="splits", type=int, default=5)
    parser.add_argument("--recall-target", dest="recall_target", type=float, default=0.90)
    parser.add_argument("--cost-fn", dest="cost_fn", type=float, default=0.0)
    parser.add_argument("--cost-fp", dest="cost_fp", type=float, default=0.0)
    parser.add_argument("--use-smote", dest="use_smote", action="store_true",
                        help="Dùng SMOTE trên train set (cần imbalanced-learn).")
    parser.add_argument("--out", dest="output_dir", type=str, default="output_fraud")
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


def load_data_from_csv(cfg: Config) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Đọc CSV từ người dùng, kiểm tra cột target.
    """
    if not os.path.isfile(cfg.csv_path):
        raise FileNotFoundError(f"File CSV không tồn tại: {cfg.csv_path}")

    df_full = pd.read_csv(cfg.csv_path)
    if cfg.target_col not in df_full.columns:
        raise ValueError(f"Không tìm thấy cột mục tiêu '{cfg.target_col}' trong file CSV.")

    # Lấy sample ~40k, fraud ~2%
    df = sample_data_for_training(df_full, target_col=cfg.target_col, n_total=40000, target_fraud_ratio=0.02, random_state=cfg.random_state)

    y = df[cfg.target_col].astype(int)
    X = df.drop(columns=[cfg.target_col])
    print(f"[INFO] X shape={X.shape}, y shape={y.shape}")
    return X, y


def sample_data_for_training(df: pd.DataFrame, target_col: str, n_total: int = 40000, target_fraud_ratio: float = 0.02, random_state: int = 42) -> pd.DataFrame:
    """
    Lấy sample từ dataset gốc:
    - Tổng số n_total row (hoặc ít hơn nếu gốc không đủ)
    - Tỷ lệ fraud ~ target_fraud_ratio
    """
    from sklearn.utils import shuffle

    df_pos = df[df[target_col] == 1]
    df_neg = df[df[target_col] == 0]

    n_pos = len(df_pos)
    n_neg_needed = int(n_pos * (1 - target_fraud_ratio) / target_fraud_ratio)

    n_neg_needed = min(n_neg_needed, len(df_neg))
    sampled_neg = df_neg.sample(n=n_neg_needed, random_state=random_state)

    sampled_df = pd.concat([df_pos, sampled_neg], axis=0)
    sampled_df = shuffle(sampled_df, random_state=random_state)

    print(f"[INFO] Sampled data: {len(sampled_df)} rows, fraud ratio={sampled_df[target_col].mean():.4f}")
    return sampled_df


def infer_column_types(X: pd.DataFrame, categorical_override: Optional[List[str]]=None) -> Tuple[List[str], List[str]]:
    cat_cols = [c for c in X.columns if str(X[c].dtype) in ("object", "category")]
    num_cols = [c for c in X.columns if c not in cat_cols]
    if categorical_override:
        override = [c for c in categorical_override if c in X.columns]
        cat_cols = sorted(list(set(cat_cols).union(set(override))))
        num_cols = [c for c in X.columns if c not in cat_cols]
    return num_cols, cat_cols


def make_preprocessors(num_cols: List[str], cat_cols: List[str], for_svm: bool) -> ColumnTransformer:
    num_steps = [("imp", SimpleImputer(strategy="median"))]
    if for_svm:
        num_steps.append(("scaler", StandardScaler()))
    numeric = Pipeline(steps=num_steps)
    categorical = Pipeline(steps=[
        ("imp", SimpleImputer(strategy="most_frequent")),
        ("oh", OneHotEncoder(handle_unknown="ignore"))
    ])
    pre = ColumnTransformer(
        transformers=[("num", numeric, num_cols), ("cat", categorical, cat_cols)],
        remainder="drop"
    )
    return pre


def build_pipelines(num_cols: list, cat_cols: list, cfg: Config):
    pre_tree = make_preprocessors(num_cols, cat_cols, for_svm=False)
    pre_svm = make_preprocessors(num_cols, cat_cols, for_svm=True)
    tree_clf = DecisionTreeClassifier(class_weight="balanced", random_state=cfg.random_state)
    svm_clf = SVC(kernel="rbf", probability=True, class_weight="balanced", random_state=cfg.random_state)

    if cfg.use_smote and IMB_AVAILABLE:
        smote_sampler = SMOTE(random_state=cfg.random_state)
        dt_pipe = ImbPipeline(steps=[("smote", smote_sampler), ("pre", pre_tree), ("clf", tree_clf)])
        svm_pipe = ImbPipeline(steps=[("smote", smote_sampler), ("pre", pre_svm), ("clf", svm_clf)])
    else:
        dt_pipe = Pipeline(steps=[("pre", pre_tree), ("clf", tree_clf)])
        svm_pipe = Pipeline(steps=[("pre", pre_svm), ("clf", svm_clf)])
        if cfg.use_smote:
            print("[WARN] imbalanced-learn không khả dụng. Bỏ SMOTE.")

    return dt_pipe, svm_pipe


def tune_and_fit(pipe, params, X_train, y_train, cfg: Config, model_name: str):
    cv = StratifiedKFold(n_splits=cfg.n_splits, shuffle=True, random_state=cfg.random_state)
    gs = GridSearchCV(pipe, param_grid=params, scoring="average_precision", cv=cv, n_jobs=-1, verbose=1)
    gs.fit(X_train, y_train)
    print(f"[{model_name}] Best params:", gs.best_params_, "CV PR-AUC=", round(gs.best_score_, 4))
    return gs.best_estimator_, gs.best_score_, gs.cv_results_


def evaluate(model, X_test, y_test, name: str, outdir: str):
    proba = model.predict_proba(X_test)[:, 1]
    pred_default = (proba >= 0.5).astype(int)
    roc = roc_auc_score(y_test, proba)
    prauc = average_precision_score(y_test, proba)
    cm = confusion_matrix(y_test, pred_default)
    report = classification_report(y_test, pred_default, digits=4, output_dict=True)
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
        "proba": proba
    }


def threshold_for_recall(y_true, proba, recall_target: float) -> Tuple[float, float, float]:
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
    try:
        pre = dt_model.named_steps["pre"]
        clf = dt_model.named_steps["clf"]
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
        print("[WARN] Không thể xuất feature importance:", e)
        return None



def plot_decision_tree(dt_model, num_cols, cat_cols, outdir):
    """
    Vẽ sơ đồ Decision Tree bằng matplotlib (không cần pydotplus)
    """
    try:
        clf = dt_model.named_steps["clf"]
        fig, ax = plt.subplots(figsize=(16, 10))
        from sklearn import tree
        tree.plot_tree(
            clf,
            filled=True,
            rounded=True,
            feature_names=getattr(dt_model.named_steps["pre"], "get_feature_names_out", lambda: num_cols + cat_cols)(),
            class_names=["Not Fraud", "Fraud"],
            max_depth=3,
            fontsize=10,
            ax=ax
        )
        path = os.path.join(outdir, "decision_tree_simple.png")
        plt.savefig(path, bbox_inches="tight")
        plt.close()
        print(f"[INFO] Decision Tree (matplotlib) lưu tại: {path}")
        return path
    except Exception as e:
        print("[WARN] Không thể vẽ Decision Tree:", e)
        return None



def plot_svm_rbf_3d(svm_model: Pipeline, X: np.ndarray, y: np.ndarray, outdir: str, n_components: int = 3):
    """
    Vẽ decision boundary SVM (RBF) trên 3D bằng PCA (hoặc 3 feature đầu nếu X có <=3 feature)
    """
    try:
        y = np.array(y)
        if X.shape[1] > n_components:
            pca = PCA(n_components=n_components, random_state=42)
            X_plot = pca.fit_transform(X)
            print(f"[INFO] Dùng PCA giảm chiều từ {X.shape[1]} → {n_components} để vẽ SVM RBF 3D.")
        else:
            X_plot = X[:, :n_components]
            pca = None

        # Grid 3D
        x_min, x_max = X_plot[:,0].min()-0.5, X_plot[:,0].max()+0.5
        y_min, y_max = X_plot[:,1].min()-0.5, X_plot[:,1].max()+0.5
        z_min, z_max = X_plot[:,2].min()-0.5, X_plot[:,2].max()+0.5
        xx, yy, zz = np.meshgrid(
            np.linspace(x_min, x_max, 30),
            np.linspace(y_min, y_max, 30),
            np.linspace(z_min, z_max, 30)
        )
        grid = np.c_[xx.ravel(), yy.ravel(), zz.ravel()]

        if pca is not None:
            grid_orig = pca.inverse_transform(grid)
        else:
            grid_orig = grid

        # Transform dữ liệu
        clf = svm_model.named_steps["clf"]
        pre = svm_model.named_steps["pre"]
        grid_pre = pre.transform(pd.DataFrame(grid_orig, columns=pre.feature_names_in_))

        # Decision function
        Z = clf.decision_function(grid_pre).reshape(xx.shape)

        # Vẽ surface nơi score ≈ 0 (decision boundary)
        fig = plt.figure(figsize=(10,8))
        ax = fig.add_subplot(111, projection='3d')

        # Chỉ vẽ các điểm gần mặt phân cách
        mask = np.abs(Z) < 0.1  # threshold ±0.1 cho surface
        ax.scatter(xx[mask], yy[mask], zz[mask], alpha=0.3, color='blue', label='Decision surface')

        # Scatter dữ liệu
        ax.scatter(X_plot[y==0,0], X_plot[y==0,1], X_plot[y==0,2], c='red', label='Not Fraud', edgecolor='k')
        ax.scatter(X_plot[y==1,0], X_plot[y==1,1], X_plot[y==1,2], c='green', label='Fraud', edgecolor='k')

        ax.set_xlabel('Component 1')
        ax.set_ylabel('Component 2')
        ax.set_zlabel('Component 3')
        ax.set_title('SVM RBF Decision Boundary (3D PCA)')
        ax.legend()
        path = os.path.join(outdir, 'svm_rbf_boundary_3d.png')
        plt.savefig(path, bbox_inches='tight')
        plt.close()
        print(f"[INFO] SVM RBF boundary (3D PCA) lưu tại: {path}")
        return path
    except Exception as e:
        print("[WARN] Không thể vẽ SVM RBF 3D:", e)
        return None


def main():
    cfg = parse_args()
    ensure_outdir(cfg.output_dir)

    # 1) Load CSV
    X, y = load_data_from_csv(cfg)

    # In ra data mẫu
    print("\n========== DỮ LIỆU TRƯỚC TIỀN XỬ LÝ ==========")
    print(X.head(10))  # in 10 dòng đầu
    print(f"Shape: {X.shape}, Fraud ratio: {y.mean():.4f}")

    # 2) Split train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=cfg.test_size, stratify=y, random_state=cfg.random_state
    )
    print(f"[INFO] Split: X_train={X_train.shape}, X_test={X_test.shape}, positive_rate_train={y_train.mean():.4f}")

    # 3) Nhận diện cột
    num_cols, cat_cols = infer_column_types(X_train, cfg.categorical_cols)
    print(f"[INFO] Numeric columns ({len(num_cols)}): {num_cols[:10]}{'...' if len(num_cols)>10 else ''}")
    print(f"[INFO] Categorical columns ({len(cat_cols)}): {cat_cols[:10]}{'...' if len(cat_cols)>10 else ''}")

    # 4) Pipelines
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

    # 8) Threshold theo Recall
    dt_thr_rec = threshold_for_recall(y_test, dt_eval["proba"], cfg.recall_target)
    svm_thr_rec = threshold_for_recall(y_test, svm_eval["proba"], cfg.recall_target)
    dt_conf_rec = confusion_at_threshold(y_test, dt_eval["proba"], dt_thr_rec[0])
    svm_conf_rec = confusion_at_threshold(y_test, svm_eval["proba"], svm_thr_rec[0])

    # 9) Threshold tối thiểu chi phí
    dt_thr_cost, dt_cost = (None, None)
    svm_thr_cost, svm_cost = (None, None)
    if cfg.cost_fn > 0 or cfg.cost_fp > 0:
        dt_thr_cost, dt_cost = threshold_min_cost(y_test, dt_eval["proba"], cfg.cost_fn, cfg.cost_fp)
        svm_thr_cost, svm_cost = threshold_min_cost(y_test, svm_eval["proba"], cfg.cost_fn, cfg.cost_fp)

    # 10) Feature importance
    dt_imp_path = extract_dt_feature_importance(best_dt, cfg.output_dir)

    
    # 11) Báo cáo JSON
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

    dt_tree_path = plot_decision_tree(best_dt, num_cols, cat_cols, cfg.output_dir)
    svm_rbf_3d_pca_path = plot_svm_rbf_3d(best_svm, X_test.to_numpy(), y_test.to_numpy(), cfg.output_dir)

    report["artifacts"]["decision_tree_structure"] = dt_tree_path
    report["artifacts"]["svm_rbf_3d_boundary"] = svm_rbf_3d_pca_path


    # 12) Lưu mô hình & báo cáo
    if joblib is not None:
        try:
            joblib.dump(best_dt, os.path.join(cfg.output_dir, "best_decision_tree.joblib"))
            joblib.dump(best_svm, os.path.join(cfg.output_dir, "best_svm_rbf.joblib"))
        except Exception as e:
            print("[WARN] Không thể lưu mô hình:", e)
    report_path = os.path.join(cfg.output_dir, "report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"[INFO] Đã lưu báo cáo JSON: {report_path}")

    # Tóm tắt
    print("\n========== TÓM TẮT KẾT QUẢ ==========")
    print(f"Data train/test: X_train={X_train.shape}, X_test={X_test.shape}, Fraud ratio train={y_train.mean():.4%}, test={y_test.mean():.4%}\n")

    for model_name, eval_dict, thr_dict in [
        ("DecisionTree", report['test_eval']['decision_tree'], report['thresholds']['decision_tree_recall_target']),
        ("SVM_RBF", report['test_eval']['svm_rbf'], report['thresholds']['svm_rbf_recall_target'])
    ]:
        print(f"--- {model_name} ---")
        print(f"PR-AUC (test):  {eval_dict['pr_auc']:.4f}")
        print(f"ROC-AUC (test): {eval_dict['roc_auc']:.4f}")
        print(f"Ngưỡng theo Recall target {cfg.recall_target:.2%}:")
        print(f"  Threshold: {thr_dict['threshold']:.4f}, Precision: {thr_dict['precision']:.4f}, Recall: {thr_dict['recall']:.4f}")
        print("  Confusion matrix @ threshold:")
        cm = thr_dict['confusion']['confusion_matrix']
        print(f"    TN={cm[0][0]}  FP={cm[0][1]}")
        print(f"    FN={cm[1][0]}  TP={cm[1][1]}")
        print()
        
    # Nếu có tối thiểu chi phí
    for model_name, min_cost_dict in [
        ("DecisionTree", report['thresholds']['decision_tree_min_cost']),
        ("SVM_RBF", report['thresholds']['svm_rbf_min_cost'])
    ]:
        if min_cost_dict:
            print(f"{model_name} - Ngưỡng tối thiểu chi phí:")
            print(f"  Threshold: {min_cost_dict['threshold']:.4f}, Cost per sample: {min_cost_dict['cost_per_sample']:.6f}\n")

    print(f"Artifacts lưu tại: {cfg.output_dir}")
    print("============================================")
    print("Hoàn tất.")


if __name__ == "__main__":
    main()
