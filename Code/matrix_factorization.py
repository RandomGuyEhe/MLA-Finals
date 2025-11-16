# MLA Fall 2025 - Hanoi University
# Academic Integrity Declaration:
# I, [Pham Khanh Son] ([2201140076]), declare that this code is my own original work.
# I have not copied or adapted code from any external repositories or previous years.
# Any sources or libraries used are explicitly cited below.

import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import sqrtm
from utils import (
    load_train_csv,
    load_valid_csv,
    load_public_test_csv,
    load_train_sparse,
    sparse_matrix_evaluate,
)

def svd_reconstruct_hanu(matrix, k):
    """
    Given the matrix, perform SVD to reconstruct the matrix using the top k components.
    - Fill missing values with column mean.
    - Center the matrix.
    - Perform SVD and reconstruct using k components.
    Returns: reconstructed matrix (n_users x n_questions).
    """
    # Make a copy to avoid modifying original
    M = matrix.copy()
    
    # Step 1: Fill missing values (NaN) with column mean
    col_means = np.nanmean(M, axis=0)
    inds = np.where(np.isnan(M))
    M[inds] = np.take(col_means, inds[1])
    
    # Step 2: Center the matrix (subtract column means)
    M_centered = M - col_means
    
    # Step 3: Perform SVD
    U, S, Vt = np.linalg.svd(M_centered, full_matrices=False)
    
    # Step 4: Keep only top k components
    U_k = U[:, :k]
    S_k = S[:k]
    Vt_k = Vt[:k, :]
    
    # Step 5: Reconstruct and add back the mean
    M_reconstructed = U_k @ np.diag(S_k) @ Vt_k + col_means
    
    # Clip to [0, 1] range for probabilities
    M_reconstructed = np.clip(M_reconstructed, 0, 1)
    
    return M_reconstructed

def squared_error_loss(data, u, z, lambda_=0.0):
    """
    Compute squared-error loss WITH L2 regularization given the data.
    Loss = (1/2) * sum_{(n,m) in O} (c_nm - u_n^T z_m)^2 + (lambda_/2)(||U||^2 + ||Z||^2)
    Returns: float (total loss)
    """
    # Extract observed entries
    n_indices = np.array(data["user_id"])
    m_indices = np.array(data["question_id"])
    c_values = np.array(data["is_correct"])
    
    # Compute predictions
    predictions = np.sum(u[n_indices] * z[m_indices], axis=1)
    
    # Compute reconstruction error
    errors = c_values - predictions
    recon_loss = 0.5 * np.sum(errors ** 2)
    
    # Compute L2 regularization
    reg_loss = (lambda_ / 2.0) * (np.sum(u ** 2) + np.sum(z ** 2))
    
    total_loss = recon_loss + reg_loss
    
    return total_loss

def update_u_z(train_data, lr, u, z, lambda_=0.0):
    """
    Perform a SGD update with L2 regularization.
    - Randomly pick an observed (user, question) pair.
    - Compute gradients including L2 terms and update u, z.
    Returns: updated u, z
    """
    # Randomly pick an observed pair
    idx = np.random.randint(len(train_data["user_id"]))
    n = train_data["user_id"][idx]
    m = train_data["question_id"][idx]
    c = train_data["is_correct"][idx]
    
    # Compute prediction
    pred = np.dot(u[n], z[m])
    
    # Compute error
    error = c - pred
    
    # Compute gradients (derivatives of loss w.r.t. u[n] and z[m])
    # dL/du_n = -(c - pred) * z_m + lambda * u_n
    # dL/dz_m = -(c - pred) * u_n + lambda * z_m
    grad_u = -error * z[m] + lambda_ * u[n]
    grad_z = -error * u[n] + lambda_ * z[m]
    
    # Update parameters
    u[n] -= lr * grad_u
    z[m] -= lr * grad_z
    
    return u, z

def als(train_data, valid_data, k, lr, num_iteration, lambda_=0.01, student_id=""):
    """
    ALS with SGD and L2 regularization.
    - Initialize u, z.
    - For each iteration:
        * Call update_u_z (enough times to touch all observed entries).
        * Track and plot both training loss and validation accuracy.
    Returns: predicted matrix (u @ z.T)
    """
    # Get matrix dimensions
    n_users = int(max(max(train_data["user_id"]), max(valid_data["user_id"]))) + 1
    n_questions = int(max(max(train_data["question_id"]), max(valid_data["question_id"]))) + 1
    n_obs = len(train_data["user_id"])
    
    # Initialize u and z randomly
    np.random.seed(42)
    u = np.random.normal(0, 0.1, (n_users, k))
    z = np.random.normal(0, 0.1, (n_questions, k))
    
    # Track losses and accuracies
    train_losses = []
    val_accs = []
    
    print(f"Training ALS with k={k}, lr={lr}, λ={lambda_}")
    
    for iteration in range(num_iteration):
        # Perform multiple SGD updates per iteration (one for each observed entry on average)
        for _ in range(n_obs):
            u, z = update_u_z(train_data, lr, u, z, lambda_=lambda_)
        
        # Compute training loss
        train_loss = squared_error_loss(train_data, u, z, lambda_=lambda_)
        train_losses.append(train_loss)
        
        # Construct predicted matrix for evaluation
        predicted_matrix = u @ z.T
        predicted_matrix = np.clip(predicted_matrix, 0, 1)
        
        # Compute validation accuracy using the full predicted matrix
        val_acc = sparse_matrix_evaluate(valid_data, predicted_matrix)
        val_accs.append(val_acc)
        
        if (iteration + 1) % 10 == 0 or iteration == 0:
            print(f"  Iter {iteration+1:3d}: Loss={train_loss:.4f}, Val Acc={val_acc:.4f}")
    
    # Construct predicted matrix
    predicted_matrix = u @ z.T
    predicted_matrix = np.clip(predicted_matrix, 0, 1)
    
    return predicted_matrix, u, z, train_losses, val_accs

def main():
    train_matrix = load_train_sparse("./data").toarray()
    train_data = load_train_csv("./data")
    val_data = load_valid_csv("./data")
    test_data = load_public_test_csv("./data")

    student_id = "2201140076"  # Replace with your student ID
    
    #####################################################################
    # SVD: Experiment with at least 5 k values
    #####################################################################
    print("\n" + "="*80)
    print("SVD EXPERIMENTS")
    print("="*80)
    
    k_values = [10, 50, 100, 200, 500]
    svd_results = {}
    
    for k in k_values:
        print(f"\nSVD with k={k}")
        reconstructed = svd_reconstruct_hanu(train_matrix, k)
        val_acc = sparse_matrix_evaluate(val_data, reconstructed)
        test_acc = sparse_matrix_evaluate(test_data, reconstructed)
        svd_results[k] = {"val_acc": val_acc, "test_acc": test_acc}
        print(f"  Validation Accuracy: {val_acc:.4f}")
        print(f"  Test Accuracy: {test_acc:.4f}")
    
    # Find best k for SVD
    best_k_svd = max(svd_results, key=lambda x: svd_results[x]["val_acc"])
    print(f"\nBest k for SVD: {best_k_svd} with Val Acc: {svd_results[best_k_svd]['val_acc']:.4f}")

    #####################################################################
    # ALS: Experiment with different k, lr, lambda values
    #####################################################################
    print("\n" + "="*80)
    print("ALS EXPERIMENTS")
    print("="*80)
    
    k_values = [10, 50, 100, 200, 500]
    lrs = [0.01, 0.05, 0.1]
    lambdas = [0.001, 0.01, 0.1]
    num_epoch = 80
    
    best_config = {
        "k": None,
        "lr": None,
        "lambda": None,
        "val_acc": -1.0,
        "test_acc": -1.0
    }
    
    als_results = {}
    
    for k in k_values:
        for lr in lrs:
            for lambda_ in lambdas:
                config_key = (k, lr, lambda_)
                print(f"\nALS: k={k}, lr={lr}, λ={lambda_}")
                
                # Train ALS
                pred_matrix, u, z, train_losses, val_accs = als(
                    train_data, val_data, k, lr, num_epoch, lambda_=lambda_, student_id=student_id
                )
                
                # Evaluate
                val_acc = val_accs[-1]  # Final validation accuracy
                test_acc = sparse_matrix_evaluate(test_data, pred_matrix)
                
                als_results[config_key] = {
                    "val_acc": val_acc,
                    "test_acc": test_acc,
                    "train_losses": train_losses,
                    "val_accs": val_accs,
                    "u": u,
                    "z": z,
                    "pred_matrix": pred_matrix
                }
                
                print(f"  Final Val Acc: {val_acc:.4f}, Test Acc: {test_acc:.4f}")
                
                # Update best configuration
                if val_acc > best_config["val_acc"]:
                    best_config["k"] = k
                    best_config["lr"] = lr
                    best_config["lambda"] = lambda_
                    best_config["val_acc"] = val_acc
                    best_config["test_acc"] = test_acc
                    best_config["results_key"] = config_key
                    print(f"  ✓ New best configuration!")
    
    print("\n" + "="*80)
    print("BEST ALS CONFIGURATION:")
    print(f"  k = {best_config['k']}")
    print(f"  Learning Rate (lr) = {best_config['lr']}")
    print(f"  Regularization (λ) = {best_config['lambda']}")
    print(f"  Validation Accuracy = {best_config['val_acc']:.4f}")
    print(f"  Test Accuracy = {best_config['test_acc']:.4f}")
    print("="*80)
    
    # Get best results
    best_results = als_results[best_config["results_key"]]
    
    # Plot training loss and validation accuracy for best ALS
    plt.figure(figsize=(14, 5))
    
    # Plot 1: Training Loss
    plt.subplot(1, 2, 1)
    plt.plot(range(1, num_epoch + 1), best_results["train_losses"], 
             label='Training Loss', linewidth=2, color='blue')
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title(f'ALS Training Loss (k={best_config["k"]}, lr={best_config["lr"]}, λ={best_config["lambda"]}, num_epoch={num_epoch})', fontsize=13)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    
    # Plot 2: Validation Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(range(1, num_epoch + 1), best_results["val_accs"], 
             label='Validation Accuracy', linewidth=2, color='orange')
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.title(f'ALS Validation Accuracy (k={best_config["k"]}, lr={best_config["lr"]}, λ={best_config["lambda"]}, num_epoch={num_epoch})', fontsize=13)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"mf_results_{student_id}.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nPlot saved as: mf_results_{student_id}.png")
    
    # Save U and Z matrices
    np.save(f"U_matrix_{student_id}.npy", best_results["u"])
    np.save(f"Z_matrix_{student_id}.npy", best_results["z"])
    print(f"U and Z matrices saved.")
    
    # Comparison plot: SVD vs ALS
    plt.figure(figsize=(10, 6))
    
    svd_ks = sorted(svd_results.keys())
    svd_val_accs = [svd_results[k]["val_acc"] for k in svd_ks]
    svd_test_accs = [svd_results[k]["test_acc"] for k in svd_ks]
    
    plt.plot(svd_ks, svd_val_accs, marker='o', linewidth=2, markersize=8, 
             label='SVD Validation', color='blue')
    plt.plot(svd_ks, svd_test_accs, marker='s', linewidth=2, markersize=8, 
             label='SVD Test', color='lightblue')
    
    # Add best ALS result
    plt.axhline(y=best_config["val_acc"], color='orange', linestyle='--', 
                linewidth=2, label=f'ALS Validation (Best)')
    plt.axhline(y=best_config["test_acc"], color='red', linestyle='--', 
                linewidth=2, label=f'ALS Test (Best)')
    
    plt.xlabel('Latent Dimension (k)', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.title('SVD vs ALS Performance Comparison', fontsize=13)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.xticks(svd_ks)
    plt.tight_layout()
    plt.savefig(f"mf_comparison_{student_id}.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Comparison plot saved as: mf_comparison_{student_id}.png")
    
    print(f"\n{'='*80}")
    print(f"SUMMARY:")
    print(f"  SVD best k: {best_k_svd}, Val Acc: {svd_results[best_k_svd]['val_acc']:.4f}")
    print(f"  ALS best k: {best_config['k']}, Val Acc: {best_config['val_acc']:.4f}")
    print(f"  ALS achieves better validation accuracy: {best_config['val_acc'] > svd_results[best_k_svd]['val_acc']}")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
