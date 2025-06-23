import pickle
import json

def dummy_feature_extraction(player):
    x1, y1, x2, y2 = player['bbox']
    area = (x2 - x1) * (y2 - y1)
    return [x1, y1, area]

def extract_features(detections):
    feature_dict = {}
    for frame in detections:
        for player in frame:
            pid = player['id']
            feat = dummy_feature_extraction(player)
            if pid not in feature_dict:
                feature_dict[pid] = []
            feature_dict[pid].append(feat)
    
    # Average feature
    for pid in feature_dict:
        feature_dict[pid] = [sum(x)/len(x) for x in zip(*feature_dict[pid])]
    
    return feature_dict

def match_players(broadcast_feats, tacticam_feats):
    from sklearn.metrics.pairwise import cosine_similarity
    mapping = {}
    for tac_id, tac_feat in tacticam_feats.items():
        best_match = None
        best_sim = -1
        for broad_id, broad_feat in broadcast_feats.items():
            sim = cosine_similarity([tac_feat], [broad_feat])[0][0]
            if sim > best_sim:
                best_sim = sim
                best_match = broad_id
        mapping[tac_id] = best_match
    return mapping

if __name__ == "__main__":
    with open("output/broadcast.pkl", "rb") as f:
        broadcast = pickle.load(f)
    with open("output/tacticam.pkl", "rb") as f:
        tacticam = pickle.load(f)

    broadcast_feats = extract_features(broadcast)
    tacticam_feats = extract_features(tacticam)

    matched = match_players(broadcast_feats, tacticam_feats)

    with open("output/mapping.json", "w") as f:
        json.dump(matched, f, indent=4)

    print("✅ Player mapping completed. Check output/mapping.json")
