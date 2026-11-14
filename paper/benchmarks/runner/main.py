# main.py
import os
import pandas as pd
import numpy as np
import config
from data_loader import load_openml_dataset, load_synthetic_dataset
from feature_selectors import SelectorWrapper
from evaluator import evaluate_downstream, fit_cv_selectors
from sklearn.base import clone

def check_if_done(df, dataset, estimator, alpha, method, run_id):
    """
    Checks if this specific configuration already exists in the DataFrame.
    """
    if df.empty:
        return False
    
    # Vectorized check for matching row
    mask = (
        (df["Dataset"] == dataset) &
        (df["Estimator"] == estimator) &
        (df["Alpha"] == alpha) &
        (df["Method"] == method) &
        (df["Run_ID"] == run_id)
    )
    return mask.any()

def save_result_row(row_dict, filename):
    """
    Appends a single row to the CSV. Creates header if file doesn't exist.
    """
    df = pd.DataFrame([row_dict])
    hdr = not os.path.exists(filename)
    df.to_csv(filename, mode='a', header=hdr, index=False)

def evaluate_and_save_checkpoints(history, X, y, task_type, base_metadata, checkpoint_filename, folds):
    """
    Evaluates model performance at each checkpoint in the history and saves to CSV.

    Parameters
    ----------
    history : list of dict
        The history from wrapper.get_history()
    X : array
        Full feature matrix
    y : array
        Target variable
    task_type : str
        'class' or 'reg'
    base_metadata : dict
        Common metadata for all checkpoints (dataset, estimator, alpha, method, run_id, etc.)
    checkpoint_filename : str
        CSV file to save checkpoint results
    """
    if history is None or len(history) == 0:
        return

    for checkpoint_idx, checkpoint in enumerate(history):
        iteration = checkpoint['iteration']
        confirmed = checkpoint['confirmed']
        tentative = checkpoint['tentative']
        rejected = checkpoint['rejected']
        elapsed_sec = checkpoint['elapsed_sec']

        n_confirmed = len(confirmed)
        n_tentative = len(tentative)
        n_rejected = len(rejected)

        # Counts/times describe the full-data run; scores use training-fold histories.
        score_mean, score_std = evaluate_downstream(
            X, y, folds, task_type=task_type,
            random_state=base_metadata['Run_ID'], iteration=iteration
        )

        # Create checkpoint row
        checkpoint_row = {
            **base_metadata,  # Include all base metadata
            'Checkpoint_ID': checkpoint_idx,
            'Iteration': iteration,
            'Elapsed_Sec': elapsed_sec,
            'N_Confirmed': n_confirmed,
            'N_Tentative': n_tentative,
            'N_Rejected': n_rejected,
            'Score_Mean': score_mean,
            'Score_Std': score_std,
            'Is_Final': checkpoint_idx == len(history) - 1
        }

        # Save checkpoint row
        save_result_row(checkpoint_row, checkpoint_filename)

def compute_gt_metrics(confirmed_set, ground_truth_set, n_features):
    """
    Computes binary classification metrics for feature selection vs. ground truth.
    """
    tp = len(confirmed_set & ground_truth_set)
    fp = len(confirmed_set - ground_truth_set)
    fn = len(ground_truth_set - confirmed_set)
    tn = n_features - tp - fp - fn

    precision   = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall      = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    f1          = (2 * precision * recall / (precision + recall)
                   if (precision + recall) > 0 else 0.0)
    return {
        'TP': tp, 'FP': fp, 'FN': fn, 'TN': tn,
        'Precision': precision, 'Recall': recall,
        'Specificity': specificity, 'F1': f1,
    }


def evaluate_and_save_gt_checkpoints(history, n_features, ground_truth, base_metadata, gt_filename):
    """
    Iterates over Vanilla Boruta history checkpoints and logs ground truth metrics for each.

    Parameters
    ----------
    history : list of dict
        Checkpoint history from wrapper.get_history() (confirmed/tentative/rejected are index sets).
    n_features : int
        Total number of features in X.
    ground_truth : dict
        Dict with 'strict' and 'broad' sets of true feature indices.
    base_metadata : dict
        Common metadata fields (Dataset, Task, n_samples, n_features_orig, Estimator, Method, Alpha, Run_ID).
    gt_filename : str
        Path to the ground truth CSV file.
    """
    if not history:
        return

    for checkpoint_idx, checkpoint in enumerate(history):
        confirmed_set = checkpoint['confirmed']
        tentative_set = checkpoint['tentative']
        rejected_set  = checkpoint['rejected']

        strict_m = compute_gt_metrics(confirmed_set, ground_truth['strict'], n_features)
        broad_m  = compute_gt_metrics(confirmed_set, ground_truth['broad'],  n_features)

        row = {
            **base_metadata,
            'Checkpoint_ID': checkpoint_idx,
            'Iteration':     checkpoint['iteration'],
            'Elapsed_Sec':   checkpoint['elapsed_sec'],
            'N_Confirmed':   len(confirmed_set),
            'N_Tentative':   len(tentative_set),
            'N_Rejected':    len(rejected_set),
            'Is_Final':      checkpoint_idx == len(history) - 1,
            # Strict
            'GT_Strict_Size':     len(ground_truth['strict']),
            'TP_Strict':          strict_m['TP'],
            'FP_Strict':          strict_m['FP'],
            'FN_Strict':          strict_m['FN'],
            'TN_Strict':          strict_m['TN'],
            'Precision_Strict':   strict_m['Precision'],
            'Recall_Strict':      strict_m['Recall'],
            'Specificity_Strict': strict_m['Specificity'],
            'F1_Strict':          strict_m['F1'],
            # Broad
            'GT_Broad_Size':     len(ground_truth['broad']),
            'TP_Broad':          broad_m['TP'],
            'FP_Broad':          broad_m['FP'],
            'FN_Broad':          broad_m['FN'],
            'TN_Broad':          broad_m['TN'],
            'Precision_Broad':   broad_m['Precision'],
            'Recall_Broad':      broad_m['Recall'],
            'Specificity_Broad': broad_m['Specificity'],
            'F1_Broad':          broad_m['F1'],
        }
        save_result_row(row, gt_filename)


def save_gt_final_snapshot(confirmed_set, n_features, n_tentative, n_rejected,
                           ground_truth, base_metadata, gt_filename, iteration, elapsed_sec):
    """
    Saves a single ground truth row for methods without per-checkpoint history
    (Greedy, GreedyParallel).
    """
    strict_m = compute_gt_metrics(confirmed_set, ground_truth['strict'], n_features)
    broad_m  = compute_gt_metrics(confirmed_set, ground_truth['broad'],  n_features)

    row = {
        **base_metadata,
        'Checkpoint_ID': 0,
        'Iteration':     iteration,
        'Elapsed_Sec':   elapsed_sec,
        'N_Confirmed':   len(confirmed_set),
        'N_Tentative':   n_tentative,
        'N_Rejected':    n_rejected,
        'Is_Final':      True,
        # Strict
        'GT_Strict_Size':     len(ground_truth['strict']),
        'TP_Strict':          strict_m['TP'],
        'FP_Strict':          strict_m['FP'],
        'FN_Strict':          strict_m['FN'],
        'TN_Strict':          strict_m['TN'],
        'Precision_Strict':   strict_m['Precision'],
        'Recall_Strict':      strict_m['Recall'],
        'Specificity_Strict': strict_m['Specificity'],
        'F1_Strict':          strict_m['F1'],
        # Broad
        'GT_Broad_Size':     len(ground_truth['broad']),
        'TP_Broad':          broad_m['TP'],
        'FP_Broad':          broad_m['FP'],
        'FN_Broad':          broad_m['FN'],
        'TN_Broad':          broad_m['TN'],
        'Precision_Broad':   broad_m['Precision'],
        'Recall_Broad':      broad_m['Recall'],
        'Specificity_Broad': broad_m['Specificity'],
        'F1_Broad':          broad_m['F1'],
    }
    save_result_row(row, gt_filename)


def run_benchmarks():
    # 1. Load existing results for "Resume" functionality
    if os.path.exists(config.RESULTS_FILE):
        print(f"Found existing results file: {config.RESULTS_FILE}. Resuming...")
        existing_df = pd.read_csv(config.RESULTS_FILE)
    else:
        print("No existing results found. Starting fresh.")
        existing_df = pd.DataFrame()

    # Loop 1: Datasets
    for data_name, data_id in config.DATASETS.items():
        task_type = config.TASK_MAP.get(data_name, "class")
        
        # Lazy Load: Only load dataset if we actually have work to do for it
        # (Optimization: We could check all inner loops first, but that's complex. 
        #  We load once per dataset.)
        X, y = None, None
        ground_truth = None
        dataset_loaded = False

        data_label = "synthetic" if isinstance(data_id, dict) else f"ID: {data_id}"
        print(f"\n==========================================")
        print(f" DATASET: {data_name} ({data_label}) | Task: {task_type}")
        print(f"==========================================")

        # Loop 2: Estimators (RF, XGB, etc.)
        for est_name in config.ESTIMATORS_LIST:
            
            # Loop 3: Alphas
            for alpha in config.ALPHAS:
                
                # Loop 4: Methods (Vanilla, Greedy...)
                # for method in ["GreedyParallel", "Greedy", "Vanilla"]:
                for method in ["GreedyParallel", "Vanilla"]: 
                    
                    # Loop 5: Runs
                    for run_i in range(config.N_RUNS):
                        
                        # --- RESUME CHECK ---
                        if check_if_done(existing_df, data_name, est_name, alpha, method, run_i):
                            # print(f"Skipping {data_name}|{est_name}|{method}|{alpha}|Run{run_i} (Already done)")
                            continue

                        # --- LOAD DATA (If not already loaded) ---
                        if not dataset_loaded:
                            try:
                                if isinstance(data_id, dict):
                                    X, y, ground_truth = load_synthetic_dataset(data_id, random_state=run_i)
                                else:
                                    X, y = load_openml_dataset(data_id, task_type=task_type)
                                    ground_truth = None
                                dataset_loaded = True
                            except Exception as e:
                                print(f"Failed to load {data_name}: {e}")
                                break # Skip this dataset entirely
                        
                        if X is None: break

                        print(f"[{est_name}] [{method}] Alpha:{alpha} Run:{run_i+1} | ", end="", flush=True)

                        # 1. Setup Estimator
                        try:
                            base_est = config.get_base_estimator(
                                task_type=task_type,
                                model_type=est_name,
                                n_jobs=-1
                            )

                            # 2. Full-data fit for timing/counts/ground truth only, not CV masks.
                            wrapper = SelectorWrapper(method, clone(base_est), alpha, random_state=run_i)
                            track_history = (method == "Vanilla")
                            wrapper.fit(X, y, track_history=track_history)
                            
                            # 3. Extract Results
                            mask_confirmed = wrapper.get_confirmed_mask()
                            mask_tentative = wrapper.get_tentative_mask()
                            n_confirmed = np.sum(mask_confirmed)
                            n_tentative = np.sum(mask_tentative)
                            
                            print(f"Iters: {wrapper.n_iterations} | Time: {wrapper.running_time:.2f}s | "
                                  f"Conf: {n_confirmed} | Tent: {n_tentative}", end="", flush=True)

                            # 4. Refit selection inside each CV training fold.
                            folds = fit_cv_selectors(X, y, wrapper, task_type, run_i)
                            score_mean_conf, score_std_conf = evaluate_downstream(
                                X, y, folds,
                                task_type=task_type, 
                                random_state=run_i
                            )
                            
                            # Evaluate with Tentative (if any)
                            score_mean_all = score_mean_conf
                            score_std_all = score_std_conf
                            
                            if any(np.any(fold_selector.get_tentative_mask())
                                   for _, _, fold_selector in folds):
                                score_mean_all, score_std_all = evaluate_downstream(
                                    X, y, folds,
                                    task_type=task_type, 
                                    random_state=run_i, include_weak=True
                                )
                                print(f" | Score: {score_mean_conf:.3f} (All: {score_mean_all:.3f})")
                            else:
                                print(f" | Score: {score_mean_conf:.3f}")

                            # 5. Prepare Data Row
                            row = {
                                "Dataset": data_name,
                                "Task": task_type,
                                "n_samples": X.shape[0],
                                "n_features_orig": X.shape[1],
                                "Estimator": est_name,
                                "Method": method,
                                "Alpha": alpha,
                                "Run_ID": run_i,
                                "Time_Sec": wrapper.running_time,
                                "Iterations": wrapper.n_iterations,
                                "N_Confirmed": n_confirmed,
                                "N_Tentative": n_tentative,
                                "Ratio_Selected": (n_confirmed + n_tentative) / X.shape[1],
                                "Score_Confirmed_Mean": score_mean_conf,
                                "Score_Confirmed_Std": score_std_conf,
                                "Score_All_Mean": score_mean_all,
                                "Score_All_Std": score_std_all
                            }
                            
                            # 6. Save Immediately
                            save_result_row(row, config.RESULTS_FILE)

                            # 7. Process and save checkpoint history (Vanilla only)
                            if method == "Vanilla":
                                history = wrapper.get_history()
                                if history is not None and len(history) > 0:
                                    # Prepare base metadata for checkpoints
                                    checkpoint_metadata = {
                                        "Dataset": data_name,
                                        "Task": task_type,
                                        "n_samples": X.shape[0],
                                        "n_features_orig": X.shape[1],
                                        "Estimator": est_name,
                                        "Method": method,
                                        "Alpha": alpha,
                                        "Run_ID": run_i
                                    }

                                    # Evaluate and save all checkpoints
                                    evaluate_and_save_checkpoints(
                                        history, X, y, task_type,
                                        checkpoint_metadata,
                                        config.CHECKPOINT_RESULTS_FILE, folds
                                    )

                            # 8. Ground truth logging (synthetic datasets only)
                            if ground_truth is not None:
                                gt_metadata = {
                                    "Dataset": data_name,
                                    "Task": task_type,
                                    "n_samples": X.shape[0],
                                    "n_features_orig": X.shape[1],
                                    "Estimator": est_name,
                                    "Method": method,
                                    "Alpha": alpha,
                                    "Run_ID": run_i
                                }
                                if method == "Vanilla":
                                    history = wrapper.get_history()
                                    evaluate_and_save_gt_checkpoints(
                                        history, X.shape[1], ground_truth,
                                        gt_metadata, config.SYNTHETIC_GT_RESULTS_FILE
                                    )
                                else:
                                    confirmed_idx = set(np.where(mask_confirmed)[0].tolist())
                                    save_gt_final_snapshot(
                                        confirmed_set=confirmed_idx,
                                        n_features=X.shape[1],
                                        n_tentative=int(n_tentative),
                                        n_rejected=int(X.shape[1] - n_confirmed - n_tentative),
                                        ground_truth=ground_truth,
                                        base_metadata=gt_metadata,
                                        gt_filename=config.SYNTHETIC_GT_RESULTS_FILE,
                                        iteration=wrapper.n_iterations,
                                        elapsed_sec=wrapper.running_time
                                    )

                        except Exception as e:
                            print(f"\n!!! CRASH on {data_name} {est_name} {method}: {e}")
                            # We continue to next run, leaving this one unsaved so it retries later
                            continue

    print(f"\nBenchmark sweep finished.")
    print(f"Main results: {config.RESULTS_FILE}")
    print(f"Checkpoint history (Vanilla only): {config.CHECKPOINT_RESULTS_FILE}")
    print(f"Ground truth metrics (synthetic datasets): {config.SYNTHETIC_GT_RESULTS_FILE}")

if __name__ == "__main__":
    run_benchmarks()

###########################################################

# import pandas as pd
# import numpy as np
# import config
# from data_loader import load_openml_dataset
# from feature_selectors import SelectorWrapper
# from evaluator import evaluate_downstream
# from sklearn.base import clone

# def run_benchmarks():
#     results = []

#     for data_name, data_id in config.DATASETS.items():
#         # Determine task type from config
#         task_type = config.TASK_MAP.get(data_name, "class")
        
#         print(f"\n==========================================")
#         print(f" PROCESSING: {data_name} (ID: {data_id}) | Task: {task_type}")
#         print(f"==========================================")
        
#         try:
#             X, y = load_openml_dataset(data_id, task_type=task_type)
#         except Exception as e:
#             print(f"Failed to load {data_name}: {e}")
#             continue

#         for alpha in config.ALPHAS:
#             print(f"\n--- Alpha: {alpha} ---")
            
#             for method in ["GreedyParallel", "Vanilla"]:
#             # for method in ["Vanilla", "Greedy", "GreedyParallel"]:
#             # for method in ["GreedyParallel", "Greedy"]: # Skipping Vanilla for speed in testing
#                 for run_i in range(config.N_RUNS):
#                     print(f"[{method}] Run {run_i+1}/{config.N_RUNS} | ", end="", flush=True)
                    
#                     # 1. Get correct estimator (Regressor vs Classifier)
#                     base_est = config.get_base_estimator(
#                         task_type=task_type, 
#                         model_type="extra_trees"
#                     )
                    
#                     wrapper = SelectorWrapper(method, clone(base_est), alpha)
                    
#                     try:
#                         wrapper.fit(X, y)
#                     except Exception as e:
#                         print(f"Error: {e}")
#                         continue
                    
#                     mask_confirmed = wrapper.get_confirmed_mask()
#                     mask_tentative = wrapper.get_tentative_mask()
                    
#                     n_confirmed = np.sum(mask_confirmed)
#                     n_tentative = np.sum(mask_tentative)
                    
#                     print(f"Iters: {wrapper.n_iterations} | Time: {wrapper.running_time:.2f}s | "
#                           f"Conf: {n_confirmed} | Tent: {n_tentative}", end="", flush=True)

#                     # 2. Evaluate (Pass task_type)
#                     score_mean_conf, score_std_conf = evaluate_downstream(
#                         X, y, mask_confirmed, 
#                         task_type=task_type, 
#                         random_state=run_i
#                     )
                    
#                     score_mean_all = score_mean_conf
#                     score_std_all = score_std_conf
                    
#                     if n_tentative > 0:
#                         mask_combined = mask_confirmed | mask_tentative
#                         score_mean_all, score_std_all = evaluate_downstream(
#                             X, y, mask_combined, 
#                             task_type=task_type, 
#                             random_state=run_i
#                         )
#                         print(f" | Score(Conf): {score_mean_conf:.3f} | Score(All): {score_mean_all:.3f}")
#                     else:
#                         print(f" | Score: {score_mean_conf:.3f}")

#                     results.append({
#                         "Dataset": data_name,
#                         "Task": task_type,
#                         "n_samples": X.shape[0],
#                         "n_features_orig": X.shape[1],
#                         "Method": method,
#                         "Alpha": alpha,
#                         "Run_ID": run_i,
#                         "Time_Sec": wrapper.running_time,
#                         "Iterations": wrapper.n_iterations,
#                         "N_Confirmed": n_confirmed,
#                         "N_Tentative": n_tentative,
#                         "Ratio_Selected": (n_confirmed + n_tentative) / X.shape[1],
#                         "Score_Confirmed_Mean": score_mean_conf,
#                         "Score_Confirmed_Std": score_std_conf,
#                         "Score_All_Mean": score_mean_all,
#                         "Score_All_Std": score_std_all
#                     })

#     df_res = pd.DataFrame(results)
#     df_res.to_csv("benchmark_results_v4.csv", index=False)
#     print("\nBenchmark Complete. Results saved to benchmark_results_v4.csv")

# if __name__ == "__main__":
#     run_benchmarks()

# #########################################################

# # main.py
# import pandas as pd
# import numpy as np
# import config
# from data_loader import load_openml_dataset
# from feature_selectors import SelectorWrapper
# from evaluator import evaluate_downstream
# from sklearn.base import clone

# def run_benchmarks():
#     results = []

#     for data_name, data_id in config.DATASETS.items():
#         print(f"\n==========================================")
#         print(f" PROCESSING DATASET: {data_name} (ID: {data_id})")
#         print(f"==========================================")
        
#         try:
#             X, y = load_openml_dataset(data_id)
#         except Exception as e:
#             print(f"Failed to load {data_name}: {e}")
#             continue

#         for alpha in config.ALPHAS:
#             print(f"\n--- Alpha: {alpha} ---")
            
#             # for method in ["Vanilla", "Greedy"]:
#             for method in ["GreedyParallel", "Greedy", 'Vanilla']:
#                 for run_i in range(config.N_RUNS):
#                     print(f"[{method}] Run {run_i+1}/{config.N_RUNS} | ", end="", flush=True)
                    
#                     # 1. Selection Phase
#                     base_est = config.get_base_estimator("extra_trees")
#                     wrapper = SelectorWrapper(method, clone(base_est), alpha)
                    
#                     try:
#                         wrapper.fit(X, y)
#                     except Exception as e:
#                         print(f"Error: {e}")
#                         continue
                    
#                     # 2. Extract Masks
#                     mask_confirmed = wrapper.get_confirmed_mask()
#                     mask_tentative = wrapper.get_tentative_mask()
                    
#                     n_confirmed = np.sum(mask_confirmed)
#                     n_tentative = np.sum(mask_tentative)
                    
#                     print(f"Iters: {wrapper.n_iterations} | Time: {wrapper.running_time:.2f}s | "
#                           f"Conf: {n_confirmed} | Tent: {n_tentative}", end="", flush=True)

#                     # 3. Evaluation Phase (Downstream Model)
                    
#                     # Case A: Evaluate only CONFIRMED features (Both Greedy and Vanilla)
#                     f1_mean_conf, f1_std_conf = evaluate_downstream(X, y, mask_confirmed, random_state=run_i)
                    
#                     # Case B: Evaluate CONFIRMED + TENTATIVE (Mainly for Vanilla fairness)
#                     f1_mean_all = f1_mean_conf
#                     f1_std_all = f1_std_conf
                    
#                     # If there are tentative features, we re-evaluate including them
#                     if n_tentative > 0:
#                         mask_combined = mask_confirmed | mask_tentative
#                         f1_mean_all, f1_std_all = evaluate_downstream(X, y, mask_combined, random_state=run_i)
#                         print(f" | F1(Conf): {f1_mean_conf:.3f} | F1(All): {f1_mean_all:.3f}")
#                     else:
#                         print(f" | F1: {f1_mean_conf:.3f}")

#                     # 4. Log Data
#                     results.append({
#                         "Dataset": data_name,
#                         "n_samples": X.shape[0],
#                         "n_features_orig": X.shape[1],
#                         "Method": method,
#                         "Alpha": alpha,
#                         "Run_ID": run_i,
#                         "Time_Sec": wrapper.running_time,
#                         "Iterations": wrapper.n_iterations,
                        
#                         # Counts
#                         "N_Confirmed": n_confirmed,
#                         "N_Tentative": n_tentative,
#                         "Ratio_Selected": (n_confirmed + n_tentative) / X.shape[1],
                        
#                         # Scores (Confirmed Only)
#                         "F1_Confirmed_Mean": f1_mean_conf,
#                         "F1_Confirmed_Std": f1_std_conf,
                        
#                         # Scores (Confirmed + Tentative)
#                         "F1_All_Mean": f1_mean_all,
#                         "F1_All_Std": f1_std_all
#                     })

#     # Save Results
#     df_res = pd.DataFrame(results)
#     df_res.to_csv("benchmark_results_v3.csv", index=False)
#     print("\nBenchmark Complete. Results saved to benchmark_results_v3.csv")

# if __name__ == "__main__":
#     run_benchmarks()

# # # main.py
# # import pandas as pd
# # import config
# # from data_loader import load_openml_dataset
# # from feature_selectors import SelectorWrapper
# # from evaluator import evaluate_downstream
# # from sklearn.base import clone

# # def run_benchmarks():
# #     results = []

# #     for data_name, data_id in config.DATASETS.items():
# #         print(f"\n=== Processing {data_name} ===")
# #         try:
# #             X, y = load_openml_dataset(data_id)
# #         except Exception as e:
# #             print(f"Failed to load {data_name}: {e}")
# #             continue

# #         for alpha in config.ALPHAS:
# #             print(f"  -- Alpha: {alpha}")
            
# #             # Compare Vanilla vs Greedy
# #             for method in ["Vanilla", "Greedy"]:
                
# #                 # Multiple runs for stability check
# #                 for run_i in range(config.N_RUNS):
# #                     print(f"    -> {method} Run {run_i+1}/{config.N_RUNS}...", end="")
                    
# #                     # 1. Setup Model
# #                     base_est = config.get_base_estimator("extra_trees")
# #                     # IMPORTANT: Clone estimator to ensure fresh start
# #                     wrapper = SelectorWrapper(method, clone(base_est), alpha)
                    
# #                     # 2. Run Selection
# #                     try:
# #                         wrapper.fit(X, y)
# #                     except Exception as e:
# #                         print(f"Error: {e}")
# #                         continue
                    
# #                     mask = wrapper.get_support()
# #                     n_features = sum(mask)
                    
# #                     # 3. Validate Quality (Downstream Model)
# #                     acc, f1 = evaluate_downstream(X, y, mask, random_state=run_i)
                    
# #                     print(f" Done. (Time: {wrapper.running_time:.2f}s, Feats: {n_features}, F1: {f1:.4f})")
                    
# #                     # 4. Log Data
# #                     results.append({
# #                         "Dataset": data_name,
# #                         "n_samples": X.shape[0],
# #                         "n_features_orig": X.shape[1],
# #                         "Method": method,
# #                         "Alpha": alpha,
# #                         "Run_ID": run_i,
# #                         "Time_Sec": wrapper.running_time,
# #                         "Selected_Feats": n_features,
# #                         "Reduction_Ratio": 1 - (n_features / X.shape[1]),
# #                         "Downstream_Acc": acc,
# #                         "Downstream_F1": f1,
# #                         "Iterations": wrapper.n_iterations
# #                     })

# #     # Save Results
# #     df_res = pd.DataFrame(results)
# #     df_res.to_csv("benchmark_results.csv", index=False)
# #     print("\nBenchmark Complete. Results saved to benchmark_results.csv")

# # if __name__ == "__main__":
# #     run_benchmarks()
