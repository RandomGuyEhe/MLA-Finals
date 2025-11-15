# MLA Fall 2025 - Hanoi University
# Academic Integrity Declaration:
# I, [Student Name] ([Student ID]), declare that this code is my own original work.
# I have not copied or adapted code from any external repositories or previous years.
# Any sources or libraries used are explicitly cited below.
from utils import (
    load_train_csv,
    load_valid_csv,
    load_public_test_csv,
    load_train_sparse,
)
import numpy as np
import matplotlib.pyplot as plt


def sigmoid(x):
    """Apply sigmoid function."""
    return np.exp(x) / (1 + np.exp(x))


def neg_log_likelihood(data, theta, beta):
    """Compute the negative log-likelihood.

    You may optionally replace the function arguments to receive a matrix.

    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param theta: Vector
    :param beta: Vector
    :return: float
    """
    #####################################################################
    # TODO:                                                             #
    # Implement the function as described in the docstring.             #
    #####################################################################
    u = np.array(data["user_id"])
    q = np.array(data["question_id"])
    c = np.array(data["is_correct"])
    x = theta[u] - beta[q]
    # log-likelihood
    log_lklihood = np.sum(c * x - np.log1p(np.exp(x)))
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return -log_lklihood


def update_theta_beta(data, lr, theta, beta):
    """Update theta and beta using gradient descent.

    You are using alternating gradient descent. Your update should look:
    for i in iterations ...
        theta <- new_theta
        beta <- new_beta

    You may optionally replace the function arguments to receive a matrix.

    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param lr: float
    :param theta: Vector
    :param beta: Vector
    :return: tuple of vectors
    """
    #####################################################################
    # TODO:                                                             #
    # Implement the function as described in the docstring.             #
    #####################################################################
    u = np.array(data["user_id"])
    q = np.array(data["question_id"])
    c = np.array(data["is_correct"])
    x = theta[u] - beta[q]
    p = sigmoid(x)
    # grads of -ll (since we minimize -ll)
    g_theta = np.zeros_like(theta)
    g_beta  = np.zeros_like(beta)
    np.add.at(g_theta, u, (p - c))   # d(-ll)/dθ_i
    np.add.at(g_beta,  q, (c - p))   # d(-ll)/dβ_j  (note sign)
    theta = theta - lr * g_theta
    beta  = beta  - lr * g_beta
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return theta, beta


def irt(data, val_data, lr, iterations):
    """Train IRT model.

    You may optionally replace the function arguments to receive a matrix.

    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param val_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param lr: float
    :param iterations: int
    :return: (theta, beta, val_acc_lst)
    """
    # TODO: Initialize theta and beta.
    n_users = int(max(max(data["user_id"]), max(val_data["user_id"]))) + 1
    n_items = int(max(max(data["question_id"]), max(val_data["question_id"]))) + 1
    theta = np.zeros(n_users)
    beta  = np.zeros(n_items)

    val_acc_lst = []
    train_nll = []
    val_nll = []

    for it in range(iterations):
        train_nll.append(neg_log_likelihood(data, theta, beta))
        val_nll.append(neg_log_likelihood(val_data, theta, beta))
        val_acc_lst.append(evaluate(val_data, theta, beta))
        theta, beta = update_theta_beta(data, lr, theta, beta)
        print(f"Iter {it+1:03d}  TrainNLL={train_nll[-1]:.4f}  ValNLL={val_nll[-1]:.4f}  ValAcc={val_acc_lst[-1]:.4f}")

    # TODO: You may change the return values to achieve what you want.
    return theta, beta, val_acc_lst, train_nll, val_nll


def evaluate(data, theta, beta):
    """Evaluate the model given data and return the accuracy.
    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}

    :param theta: Vector
    :param beta: Vector
    :return: float
    """

    u = np.array(data["user_id"])
    q = np.array(data["question_id"])
    c = np.array(data["is_correct"])
    p = sigmoid(theta[u] - beta[q])
    pred = (p >= 0.5).astype(int)
    return float(np.mean(pred == c))


def plot_icc(theta, beta, questions, student_id="2201140076"):
    """Plot Item Characteristic Curves (ICC) for selected questions.
    
    :param theta: Student ability parameters
    :param beta: Question difficulty parameters
    :param questions: List of question indices to plot
    :param student_id: Student ID for saving the plot
    """
    thetas = np.linspace(-4, 4, 400)
    plt.figure(figsize=(10, 6))
    for j in questions:
        probs = sigmoid(thetas - beta[j])
        plt.plot(thetas, probs, label=f"Question {j} (β={beta[j]:.2f})", linewidth=2)
    plt.xlabel(r"Student Ability ($\theta$)", fontsize=12)
    plt.ylabel(r"Probability of Correct Answer ($p(c=1)$)", fontsize=12)
    plt.title("Item Characteristic Curves (IRT Model)", fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"irt_curves_visualization_{student_id}.png", dpi=150)
    plt.close()


def main():
    train_data = load_train_csv("./data")
    # You may optionally use the sparse matrix.
    # sparse_matrix = load_train_sparse("./data")
    val_data = load_valid_csv("./data")
    test_data = load_public_test_csv("./data")

    #####################################################################
    # TODO:                                                             #
    # Tune learning rate and number of iterations. With the implemented #
    # code, report the validation and test accuracy.                    #
    #####################################################################
    student_id = "2201140076"  # edit to yours
    lr = 0.02
    iterations = 80
    theta, beta, val_acc, train_nll, val_nll = irt(train_data, val_data, lr=lr, iterations=iterations)  

    val_acc_final = evaluate(val_data, theta, beta)
    test_acc_final = evaluate(test_data, theta, beta)
    print(f"Final Val Acc: {val_acc_final:.4f}")
    print(f"Final Test Acc: {test_acc_final:.4f}")

    plt.figure(figsize=(10, 6))
    plt.plot(range(1, iterations+1), [-x for x in train_nll], label="Train Log-Likelihood", linewidth=2)
    plt.plot(range(1, iterations+1), [-x for x in val_nll], label="Valid Log-Likelihood", linewidth=2)
    plt.xlabel("Iteration", fontsize=12)
    plt.ylabel("Log-Likelihood", fontsize=12)
    plt.title("IRT Model: Training and Validation Log-Likelihood", fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"irt_curves_{student_id}.png", dpi=150)
    plt.close()  

    #####################################################################
    # TODO:                                                             #
    # Implement part (d): Visualization of IRT curves                   #
    # Select three questions j1, j2, j3 and plot p(cij=1) as function  #
    # of θi for each question.                                          #
    #####################################################################
    
    # Select three questions with different difficulty levels
    # Sort questions by difficulty (beta) and select ones with low, medium, high difficulty
    sorted_indices = np.argsort(beta)
    q1 = sorted_indices[len(sorted_indices) // 4]      # Easy question
    q2 = sorted_indices[len(sorted_indices) // 2]      # Medium difficulty
    q3 = sorted_indices[3 * len(sorted_indices) // 4]  # Hard question
    
    plot_icc(theta, beta, questions=[q1, q2, q3], student_id=student_id)
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################


if __name__ == "__main__":
    main()
