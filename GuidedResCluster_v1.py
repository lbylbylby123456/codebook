# guided_res_cluster.py
import numpy as np
from GuidedKMeans_v1 import GuidedKMeans

class GuidedResCluster:
    def __init__(self, n_layers, codebook_size, dim,
                 stat_table,                      # dict: price_int -> {'avg_debias': float, 'std_debias': float, 'cnt': int}
                 weight_alpha=1.0,                # 权重强度：0=不加权，1=标准
                 eps_ratio=0.02,                  # eps = std_debias * eps_ratio
                 eps_min=1e-3,
                 dead_code_ratio=0.005,           # 使用率低于此比例视为死码字
                 reset_dead_codes=True,
                 random_state=42):
        self.n_layers = n_layers
        self.codebook_size = codebook_size
        self.dim = dim
        self.stat_table = stat_table
        self.weight_alpha = weight_alpha
        self.eps_ratio = eps_ratio
        self.eps_min = eps_min
        self.dead_code_ratio = dead_code_ratio
        self.reset_dead_codes = reset_dead_codes
        self.random_state = random_state
        self.codebooks = [None] * n_layers
        self.num_samples = [None] * n_layers
        self.eps_table = {}                      # price_int -> eps
        self.reset_log = []                      # 记录死码字重置次数

    # ---------- 从 stat_table 构建样本权重 ----------
    def build_sample_weight(self, price_bucket):
        """
        price_bucket: array [N] 每个样本的价格桶
        返回: array [N] 权重
        """
        n = len(price_bucket)
        w = np.ones(n, dtype=float)
        if self.weight_alpha == 0:
            return w

        # 全局兜底
        global_avg = np.mean([v['avg_debias'] for v in self.stat_table.values() if v['cnt'] > 0])
        for i, pb in enumerate(price_bucket):
            info = self.stat_table.get(int(pb), None)
            if info is None or info['cnt'] == 0:
                dev = abs(global_avg - 1.0)
            else:
                dev = abs(info['avg_debias'] - 1.0)
            w[i] = 1.0 + self.weight_alpha * dev      # 偏差越大权重越高
        return w

    # ---------- 从 stat_table 构建每个价格桶的 epsilon ----------
    def build_eps_table(self):
        self.eps_table = {}
        for pb, info in self.stat_table.items():
            std = info.get('std_debias', 0.0) or 0.0
            eps = max(self.eps_min, std * self.eps_ratio)
            self.eps_table[int(pb)] = eps
        # 兜底值
        all_eps = [v for v in self.eps_table.values() if v > 0]
        self.default_eps = float(np.median(all_eps)) if all_eps else self.eps_min
        return self.eps_table

    def _eps_of(self, price_bucket):
        return np.array([self.eps_table.get(int(pb), self.default_eps) for pb in price_bucket])

    # ---------- 训练 ----------
    def train(self, inputs, price_bucket, sample_weight=None):
        """
        inputs: [N, dim] 量化目标
        price_bucket: [N] 每个样本的价格桶
        sample_weight: [N] 不传则内部按 stat_table 构建
        """
        if sample_weight is None:
            sample_weight = self.build_sample_weight(price_bucket)
        self.build_eps_table()
        eps_per_sample = self._eps_of(price_bucket)    # [N]

        x = np.asarray(inputs, dtype=float)
        n = x.shape[0]
        sample_mask = np.ones(n, dtype=int)            # 1=参与
        quant_codes = np.zeros((n, self.n_layers))

        for l in range(self.n_layers):
            km = GuidedKMeans(n_clusters=self.codebook_size[l],
                              random_state=self.random_state + l)
            labels = km.fit_predict(x, sample_mask, sample_weight)

            # 记录码字使用量（仅统计参与样本）
            cnt = [int(sample_mask.sum())]
            for k in range(self.codebook_size[l]):
                cnt.append(int(np.sum((labels == k) & (sample_mask == 1))))
            self.num_samples[l] = cnt

            # ★ 码本按值排序，同时把计数按同一顺序重排（修掉原代码 num_samples 与 codebooks 错位）
            order = np.argsort(km.centers_.flatten())
            centers_sorted = km.centers_[order]
            count_sorted = [cnt[0]] + [cnt[1 + o] for o in order]
            self.num_samples[l] = count_sorted

            # 量化值：labels=-1 的样本用 0 填充（不再依赖 -1 负索引）
            quant = np.zeros((n, self.dim))
            valid = (labels >= 0) & (sample_mask == 1)
            if np.any(valid):
                quant[valid] = km.centers_[labels[valid]]
            quant_codes[:, l] = quant[:, 0]
            residuals = (x - quant) * sample_mask[:, None]

            # ★ 死码字重置
            if self.reset_dead_codes and l < self.n_layers:
                n_active = int(sample_mask.sum())
                if n_active > 0:
                    dead_thr = max(1, int(self.dead_code_ratio * n_active))
                    # 用高权重、残差大的样本重置死码字
                    active_idx = np.where(sample_mask == 1)[0]
                    residual_mag = np.abs(residuals[active_idx, 0])
                    score = residual_mag * sample_weight[active_idx]
                    top_k = active_idx[np.argsort(-score)[:self.codebook_size[l]]]
                    reset_cnt = 0
                    for k in range(self.codebook_size[l]):
                        if self.num_samples[l][k + 1] < dead_thr and reset_cnt < len(top_k):
                            # 找到该死码字在 sorted centers 里的位置 k，用 top 残差样本值重置
                            centers_sorted[k] = x[top_k[reset_cnt], 0]
                            reset_cnt += 1
                    if reset_cnt > 0:
                        self.reset_log.append((l, reset_cnt))
                        # 重置后重新分配，更新 labels / residuals
                        d = np.abs(x[:, None, 0] - centers_sorted[None, :, 0]) if self.dim == 1 else \
                            np.sum((x[:, None, :] - centers_sorted[None, :, :])**2, axis=2)
                        labels2 = np.argmin(d, axis=1)
                        quant2 = np.zeros((n, self.dim))
                        v2 = (labels2 >= 0) & (sample_mask == 1)
                        quant2[v2] = centers_sorted[labels2[v2]]
                        quant_codes[:, l] = quant2[:, 0]
                        residuals = (x - quant2) * sample_mask[:, None]
                        cnt = [int(sample_mask.sum())] + [int(np.sum((labels2 == k) & (sample_mask == 1))) for k in range(self.codebook_size[l])]
                        self.num_samples[l] = cnt

            self.codebooks[l] = centers_sorted
            # ★ 自适应退出：用每个样本自己的 eps
            still_learning = np.all(np.abs(residuals) > eps_per_sample[:, None], axis=1).astype(int)
            sample_mask = sample_mask * still_learning
            x = residuals

    # ---------- 编码 ----------
    def encode(self, inputs, price_bucket):
        self.build_eps_table() if not self.eps_table else None
        eps_per_sample = self._eps_of(price_bucket)
        n = inputs.shape[0]
        codes = np.zeros((n, self.n_layers), dtype=int)
        quant_codes = np.zeros((n, self.n_layers))
        num_samples = np.zeros((self.n_layers, max(self.codebook_size) + 1))
        eos = np.zeros(n, dtype=int)
        x = np.reshape(np.asarray(inputs, dtype=float), (n, self.dim))

        for l in range(self.n_layers):
            d = np.sum((x[:, None, :] - self.codebooks[l][None, :, :]) ** 2, axis=2)
            codes[:, l] = np.where(eos == 1, -1, np.argmin(d, axis=1))
            valid = eos == 0
            quant_codes[:, l] = np.where(valid, self.codebooks[l][codes[:, l], 0], 0.0)
            num_samples[l, 0] = int(valid.sum())
            for k in range(self.codebook_size[l]):
                num_samples[l, k + 1] = int(np.sum(codes[:, l] == k))
            x = (x - quant_codes[:, [l]]) * (1 - eos[:, None])
            # ★ 自适应 eos
            eos = np.all(np.abs(x) < eps_per_sample[:, None], axis=1).astype(int)

        return codes, quant_codes, num_samples