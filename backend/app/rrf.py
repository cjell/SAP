from typing import Dict, List, Any, Tuple

def fuse_results_rrf(
    result_sets: Dict[str, List[Dict[str, Any]]],
    k_rrf: int = 10
) -> List[Dict[str, Any]]:
    """Reciprocal rank fusion over the retrieval arms.

    Scores it on rank rather than raw distance, which matters here because DINO
    and the text encoder live in different spaces and their distances aren't
    comparable. k_rrf flattens the curve - higher means less weight on the top hit.
    """

    fused_scores: Dict[str, float] = {}
    payloads: Dict[str, Dict[str, Any]] = {}

    for arm_name, results in result_sets.items():
        for rank, item in enumerate(results):
            # ids are only unique within a store, so namespace them by source
            key = f"{item.get('source', arm_name)}::{item['id']}"

            contribution = 1.0 / (k_rrf + rank + 1)  # +1 since rank is 0-based

            fused_scores[key] = fused_scores.get(key, 0.0) + contribution

            # keep track of which arms found it, handy when debugging a bad ranking
            if key not in payloads:
                payload = dict(item)
                payload["from_arm"] = [arm_name]
                payloads[key] = payload
            else:
                payloads[key]["from_arm"].append(arm_name)

    ranked: List[Tuple[str, float]] = sorted(
        fused_scores.items(),
        key=lambda kv: kv[1],
        reverse=True
    )

    fused_list: List[Dict[str, Any]] = []
    for key, score in ranked:
        item = payloads[key]
        item["rrf_score"] = score
        fused_list.append(item)

    return fused_list
