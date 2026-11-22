"""Exact finite-population checks for the cohort referee's identification notes."""

from fractions import Fraction as F


def masses(numerators, denominator):
    return tuple(F(number, denominator) for number in numerators)


def mix(prevalence, anomalies, normals):
    return tuple(
        prevalence * anomaly + (1 - prevalence) * normal
        for anomaly, normal in zip(anomalies, normals, strict=True)
    )


def ap(target, anomalies, prevalence, scores):
    """Step-function AP with all observations of equal score crossing together."""
    total = covered = recall = F(0)
    for score in sorted(set(scores), reverse=True):
        cells = [index for index, value in enumerate(scores) if value == score]
        covered += sum(target[index] for index in cells)
        recall_step = sum(anomalies[index] for index in cells)
        recall += recall_step
        total += prevalence * recall / covered * recall_step
    return total


def ap_coefficients(target, difference, scores):
    linear = quadratic = covered = excess = F(0)
    for score in sorted(set(scores), reverse=True):
        cells = [index for index, value in enumerate(scores) if value == score]
        group_mass = sum(target[index] for index in cells)
        group_excess = sum(difference[index] for index in cells)
        covered += group_mass
        excess += group_excess
        linear += group_mass * excess / covered
        quadratic += group_excess * excess / covered
    return linear, quadratic


def two_rare_groups():
    target = masses((1, 1, 18), 20)
    prevalence = F(1, 20)
    options = ((1, 0, 0), (0, 1, 0))
    outcomes = tuple(
        tuple(ap(target, conditional, prevalence, scores) for scores in options)
        for conditional in options
    )
    assert outcomes == ((1, prevalence), (prevalence, 1))
    print("Single-batch AP by world and detector", outcomes)


def ordered_mixtures():
    target = masses((10, 100, 90), 200)
    high = masses((11, 102, 87), 200)
    prevalence = F(1, 10)
    difference = tuple(h - t for h, t in zip(high, target, strict=True))
    options = ((1, 0, 0), (0, 1, 0), (1, 1, 0))
    coefficients = [ap_coefficients(target, difference, scores) for scores in options]
    max_b = min(t / -d for t, d in zip(target, difference, strict=True) if d < 0)
    (linear_1, quadratic_1), (linear_2, quadratic_2), (linear_3, quadratic_3) = (
        coefficients
    )
    crossing = (linear_2 - linear_1) / (quadratic_1 - quadratic_2)
    assert (max_b, crossing) == (30, F(50, 3))
    for b in (1, max_b):
        winner = linear_3 + b * quadratic_3
        assert all(
            winner > linear + b * quadratic for linear, quadratic in coefficients[:2]
        )
    budget = F(1, 20)
    differences_at_budget = []
    for flags in options:
        selected_mass = sum(t for t, flag in zip(target, flags, strict=True) if flag)
        selected_excess = sum(
            d for d, flag in zip(difference, flags, strict=True) if flag
        )
        differences_at_budget.append(budget * selected_excess / selected_mass)
    assert differences_at_budget == [F(1, 200), F(1, 1000), F(3, 2200)]
    print("Feasible upper bound and AP crossing", max_b, crossing)
    for b in (F(10), F(24)):
        a = prevalence * b / (1 - prevalence)
        anomalies = tuple(t + b * d for t, d in zip(target, difference, strict=True))
        normals = tuple(t - a * d for t, d in zip(target, difference, strict=True))
        high_prevalence = prevalence + (1 - prevalence) / b
        assert min(*anomalies, *normals) >= 0
        assert mix(prevalence, anomalies, normals) == target
        assert mix(high_prevalence, anomalies, normals) == high
        expected = tuple(
            ap(target, anomalies, prevalence, scores) for scores in options
        )
        polynomial = tuple(
            prevalence * (1 + b * linear + b * b * quadratic)
            for linear, quadratic in coefficients
        )
        assert expected == polynomial
        assert (expected[0] > expected[1]) == (b > crossing)
        print("Ordered-mixture AP", b, high_prevalence, tuple(map(float, expected)))


def changed_normal_population():
    target = masses((1, 1, 8), 10)
    anomalies = masses((4, 1, 5), 10)
    target_normals = masses((2, 3, 25), 30)
    high_normals = masses((2, 20, 78), 100)
    target_prevalence = F(1, 10)
    high_prevalence = F(1, 5)
    assert mix(target_prevalence, anomalies, target_normals) == target
    high = mix(high_prevalence, anomalies, high_normals)
    contrasts = tuple(high[i] - target[i] for i in (0, 1))
    precisions = tuple(target_prevalence * anomalies[i] / target[i] for i in (0, 1))
    assert contrasts == (F(-1, 250), F(2, 25))
    assert precisions == (F(2, 5), F(1, 10))
    print("Within-normal drift contrasts and target precision", contrasts, precisions)


if __name__ == "__main__":
    two_rare_groups()
    ordered_mixtures()
    changed_normal_population()
