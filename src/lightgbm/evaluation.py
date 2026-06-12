from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)


def print_binary_results(y_true_multi, y_pred_multi, class_names, normal_label="Normal"):
    normal_idx = list(class_names).index(normal_label)

    y_true_bin = (y_true_multi != normal_idx).astype(int)
    y_pred_bin = (y_pred_multi != normal_idx).astype(int)

    binary_names = ["Normal", "Attack"]

    print("\n" + "=" * 60)
    print("=== BINARY CLASSIFICATION REPORT ===")
    print("=" * 60)

    print(classification_report(
        y_true_bin,
        y_pred_bin,
        target_names=binary_names,
        digits=4
    ))

    print("\n=== BINARY CONFUSION MATRIX ===")
    print(confusion_matrix(y_true_bin, y_pred_bin))

    acc = accuracy_score(y_true_bin, y_pred_bin)
    prec = precision_score(y_true_bin, y_pred_bin, zero_division=0)
    rec = recall_score(y_true_bin, y_pred_bin, zero_division=0)
    f1 = f1_score(y_true_bin, y_pred_bin, zero_division=0)

    print("\n=== BINARY METRICS ===")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1-score:  {f1:.4f}")


def print_multiclass_results(y_test, pred, class_names):
    print("\n" + "=" * 60)
    print("=== BÁO CÁO KẾT QUẢ ===")
    print("=" * 60)

    print(classification_report(
        y_test,
        pred,
        target_names=class_names,
        digits=4
    ))

    print("\n=== MA TRẬN NHẦM LẪN ===")
    cm = confusion_matrix(y_test, pred)
    print(cm)

    print("\n=== PER-CLASS RECALL ===")

    for i, name in enumerate(class_names):
        mask = (y_test == i)

        if mask.sum() > 0:
            rec = (pred[mask] == i).mean()
            n = mask.sum()
            bar = "█" * int(rec * 20)

            print(f"  {name:20s}: {rec:.4f}  {bar}  (n={n})")

    print(f"\nAccuracy:  {accuracy_score(y_test, pred):.4f}")
    print(f"Recall:    {recall_score(y_test, pred, average='macro'):.4f}")
    print(f"Precision: {precision_score(y_test, pred, average='macro'):.4f}")
    print(f"Macro F1:  {f1_score(y_test, pred, average='macro'):.4f}")