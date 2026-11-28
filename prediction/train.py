## train the k means model on featuretable and save the model to a file


from sklearn.cluster import KMeans
import pandas as pd
import joblib
from aggregation.aggregation import BASE_COLUMNS
from pathlib import Path
from sklearn.preprocessing import StandardScaler


CLUSTERS = 3  # Number of clusters for KMeans
LEVELS = [0.0, 0.5, 1.0]  # Low Medium High

# +1: a higher value means MORE attention
# -1: a higher value means LESS attention

# DONT FORGET TO UPDATE THIS WHEN YOU ADD NEW FEATURES
FEATURE_DIRECTIONS = { #for the prototype, we assume all features are equally important with different directions 
    "pitchVariation": +1,          # lively voice
    "pauseRatio": +1,              # pauses give time to process
    "speakingPace": -1,            # very fast speech is harder to follow
    "averageSentenceLength": -1,   # long sentences are harder to follow
    "lexicalDensity": -1,          # dense content is harder to follow
}

def trainModel(tables: list[pd.DataFrame], modelPath: str | Path ) -> None:
    # combine all feature tables to a DataFrame
    combined_table = pd.concat(tables, ignore_index=True)

    # take only fature columns (drop lectureID and windowIndex) 
    feature_columns = [col for col in combined_table.columns if col not in BASE_COLUMNS]
    features = combined_table[feature_columns].dropna()  # Drop rows with any NaN values // for initial pipeline only

    scaler = StandardScaler().fit(features) #fit learns each column's average and spread from the training data. And remembers for test data

    # random_state=0 makes it give the same result every run. k-means starts from random positions, and this fixes the randomness.
    # n_init=10 tries 10 different starting positions and keeps the best result, which is more reliable.
    kmeans = KMeans(n_clusters=CLUSTERS, random_state=0, n_init=10).fit(scaler.transform(features)) 


    # Score each cluster: add up its (scaled) feature averages, flipped by direction.
    # Scaled values are used so every feature counts equally, whatever its units.
    scaledCentres = pd.DataFrame(kmeans.cluster_centers_, columns=feature_columns)

    usedFeatures = [f for f in feature_columns if f in FEATURE_DIRECTIONS]
    attentionScore = sum(scaledCentres[f] * FEATURE_DIRECTIONS[f] for f in usedFeatures)

    order = attentionScore.sort_values().index      # sort by score
    levelByCluster = {int(cluster): level for cluster, level in zip(order, LEVELS)}


    print("Cluster centres:")
    print(scaledCentres.round(3).to_string())
    print(f"Cluster -> attention level: {levelByCluster}")

    #   save results
    joblib.dump({
        "featureColumns": feature_columns,
        "scaler": scaler,
        "kmeans": kmeans,
        "levelByCluster": levelByCluster,
    }, modelPath)
    print(f"Saved model to {modelPath}")