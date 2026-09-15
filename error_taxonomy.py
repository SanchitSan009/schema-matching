def categorize(case, actual_columns, prediction, retrieved, band=None, same_cluster=False):
    errors = []
    for expected, actual in zip(case["columns"], actual_columns):
        if actual["base"] != expected["base"]:
            errors.append("bad_base_extraction")
        if sorted(actual["qualifiers"]) != sorted(expected["qualifiers"]):
            errors.append("bad_qualifier_extraction")
    expected_match = case["expected_relation"] == "EQUIVALENT"
    if not retrieved:
        if expected_match:
            errors.append("retrieval_miss")
        return sorted(set(errors))
    if prediction.get("relation") is None:
        errors.append("relation_not_evaluated")
    elif prediction["relation"]["relation"] != case["expected_relation"]:
        phenomenon = case["phenomenon"]
        errors.append({"synonym": "synonym_failure", "contrast": "antonym_contrast_failure",
                       "abbreviation": "abbreviation_failure", "typo": "typo_failure",
                       "context": "context_ambiguity"}[phenomenon])
    expected_decision = "ACCEPT" if expected_match else "UNCERTAIN" if case["expected_relation"] == "UNCERTAIN" else "REJECT"
    if prediction["decision"] != expected_decision:
        errors.append(("uncertain_case_forced_accept" if case["expected_relation"] == "UNCERTAIN" else "false_accept") if prediction["decision"] == "ACCEPT" else
                      "abstention" if prediction["decision"] == "UNCERTAIN" else "false_reject")
    if same_cluster and not expected_match:
        errors.append("clustering_error")
    elif expected_match and not same_cluster and band != "HIGH_CONFIDENCE_MATCH":
        errors.append("insufficient_calibration" if prediction["decision"] == "ACCEPT" else "unconfirmed_match")
    return sorted(set(errors))
