# guided_kmeans.py
import numpy as np

class GuidedKMeans:
    def __init__(self, n_clusters, max_iter=200, tol=1e-6, random_state=None):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.centers_ = None
        self.labels_ = None
        self.inertia_ = None

    def _l2_distance(self, X, Y):
        return np.sum((X[:, None, :] - Y[None, :, :]) ** 2, axis=2)

    def _initialize_centers(self, X, w):
        """加权 k-means++，L2 距离（修掉原代码 L1 初始化的不一致）"""
        if self.random_state is not None:
            np.random.seed(self.random_state)
        n = X.shape[0]
        w = w / w.sum()
        centers = np.zeros((self.n_clusters, X.shape[1]))

        centers[0] = X[np.random.choice(n, p=w)]
        dist2 = np.full(n, np.inf)
        for i in range(1, self.n_clusters):
            d = np.sum((X - centers[i - 1]) ** 2, axis=1)
            dist2 = np.minimum(dist2, d)
            probs = dist2 * w                       # 加权采样
            probs = probs / max(probs.sum(), 1e-12)
            centers[i] = X[np.random.choice(n, p=probs)]
        return centers

    def fit(self, X, sample_weight=None):
        X = np.asarray(X, dtype=float)
        n = X.shape[0]
        if sample_weight is None:
            sample_weight = np.ones(n)
        sample_weight = np.asarray(sample_weight, dtype=float)

        centers = self._initialize_centers(X, sample_weight)
        prev = None
        for _ in range(self.max_iter):
            dist = self._l2_distance(X, centers)
            labels = np.argmin(dist, axis=1)

            # ★ 加权更新中心（原代码是 np.mean，改为 np.average）
            for j in range(self.n_clusters):
                mask = labels == j
                if np.any(mask):
                    centers[j] = np.average(X[mask], axis=0, weights=sample_weight[mask])

            min_d = dist[np.arange(n), labels]
            inertia = np.sum(min_d * sample_weight)     # ★ 加权 inertia
            if prev is not None and abs(prev - inertia) < self.tol * max(abs(inertia), 1e-12):
                break
            prev = inertia

        self.centers_ = centers
        self.labels_ = labels
        self.inertia_ = prev
        return self

    def fit_predict(self, X, sample_mask=None, sample_weight=None):
        if sample_mask is None:
            return self.fit(X, sample_weight).labels_
        mask = np.asarray(sample_mask, dtype=bool)
        labels = np.full(len(X), -1, dtype=int)
        if np.any(mask):
            w_in = sample_weight[mask] if sample_weight is not None else None
            self.fit(X[mask], w_in)
            labels[mask] = self.labels_
        return labels