from collections import Counter


def report(pairs, classes):
    """pairs: list of (true_label, predicted_label)."""
    n = len(pairs)
    correct = sum(t == p for t, p in pairs)
    print(f"accuracy: {correct}/{n} = {correct / n:.1%}")

    print(f"\n{'class':<15}{'n':>5}{'precision':>11}{'recall':>9}")
    for c in classes:
        tp = sum(t == c and p == c for t, p in pairs)
        predicted = sum(p == c for _, p in pairs)
        actual = sum(t == c for t, _ in pairs)
        precision = tp / predicted if predicted else 0
        recall = tp / actual if actual else 0
        print(f"{c:<15}{actual:>5}{precision:>11.1%}{recall:>9.1%}")

    counts = Counter(pairs)
    print("\nconfusion (rows = true, columns = predicted)")
    print(" " * 15 + "".join(f"{c[:7]:>9}" for c in classes))
    for t in classes:
        print(f"{t:<15}" + "".join(f"{counts[(t, p)]:>9}" for p in classes))