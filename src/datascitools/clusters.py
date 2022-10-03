import matplotlib.pyplot as plt
from kneed import KneeLocator
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


def find_kmeans(data, scaler=StandardScaler(), max_clusters=12, plot=True):
    scaled_features = scaler.fit_transform(data)

    kmeans_kwargs = {"init": "random", "n_init": 10, "max_iter": 300, "random_state": 42}

    sse = []

    for k in range(1, max_clusters):
        kmeans = KMeans(n_clusters=k, **kmeans_kwargs)
        kmeans.fit(scaled_features)
        sse.append(kmeans.inertia_)

    if plot:
        plt.style.use("fivethirtyeight")
        plt.plot(range(1, max_clusters), sse)
        plt.xticks(range(1, max_clusters))
        plt.xlabel("Number of Clusters")
        plt.ylabel("SSE")
        plt.show()

    kl = KneeLocator(range(1, max_clusters), sse, curve="convex", direction="decreasing")

    return kl.elbow
