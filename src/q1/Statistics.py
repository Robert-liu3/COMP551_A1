def compute_statistics(X, y, dataset_name=""):
    print(f"\n===== Statistics for {dataset_name} Dataset =====\n")

    class_distribution = y.value_counts()
    print(f"Class Distribution:\n{class_distribution}")

    # Check if dataset is balanced
    total = class_distribution.sum()
    balance_ratio = class_distribution / total * 100
    print(f"\nClass Balance (%):\n{balance_ratio}\n")

    # Summary
    print("Numerical Feature Statistics:\n")
    print(X.describe())