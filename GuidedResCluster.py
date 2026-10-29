# guided_res_cluster.py
"""
Guided Residual Cluster
=======================
多层残差码本量化器，1 维标量输入的逐层量化。

每层训练流程：
    1) 只用"仍在学习"的样本（sample_mask=1）做加权 KMeans → 初始码本
    2) 码本按值升序排序（保证 code 单调可逆）
    3) 初始分配，得到残差与使用率统计
    4) ★ 死码字重置：用高权重×大残差样本填充低使用率码字，重排后再分配
    5) 去重：推开重置/KMeans 产生的相等码字，保证严格递增
    6) ★ 尾部极值锚定（仅第一层）：最后 n_tail 个码字两段式硬锚定：
         · 前 n_tail-1 个：在 (core_max, data_max] 等比展开（真实尾部分辨率）
         · 最后 1 个：固定钉在 tail_hard_max（设计哨兵，默认 100.0）
       使码本上界从"密度驱动"变为"固定设计上界"，推理时高偏差样本不再
       被全部钳到同一个码字上
    7) 逐样本 eps 判定退出，残差进入下一层
"""
import numpy as np
from GuidedKMeans import GuidedKMeans


class GuidedResCluster:
    def __init__(self, n_layers, codebook_size, dim,
                 stat_table,                      # dict: price_int -> {'avg_debias','std_debias','cnt'}
                 weight_alpha=1.0,                # 权重强度：0=不加权，1=标准
                 eps_ratio=0.02,                  # eps = std_debias * eps_ratio
                 eps_min=1e-3,
                 dead_code_ratio=0.005,           # 使用率低于此比例视为死码字
                 reset_dead_codes=True,
                 tail_reserve=4,                  # ★ 每层预留多少个码字给尾部极值
                 tail_hard_max=None,               # ★ 尾部锚定硬上界；None=用数据最大值
                 tail_reserve_layers=(0,),        # ★ 启用尾部锚定的层（默认仅第一层）
                 tail_infer_n=0,                  # ★ 在 [data_max, tail_hard_max] 之间额外插入的推理预留码字数
                 eps_ratio_schedule=None,         # ★ 逐层 eps 缩放系数；None=各层同阈值（原逻辑）
                 eps_hard_cap=None,               # ★ eps 绝对上限；防止高 std 桶 eps 过大而提前退出
                 random_state=42):
        self.n_layers = n_layers
        self.codebook_size = list(codebook_size)
        self.dim = dim
        self.stat_table = stat_table
        self.weight_alpha = weight_alpha
        self.eps_ratio = eps_ratio
        self.eps_min = eps_min
        self.dead_code_ratio = dead_code_ratio
        self.reset_dead_codes = reset_dead_codes

        # 尾部极值锚定
        self.tail_reserve = max(0, int(tail_reserve))
        self.tail_hard_max = float(tail_hard_max) if tail_hard_max is not None else None
        self.tail_reserve_layers = tuple(tail_reserve_layers) if tail_reserve_layers is not None else tuple()
        # 推理预留段：在 [data_max, tail_hard_max] 之间额外插入 n 个等比锚点
        # 配合 tail_hard_max 扩充推理上界时使用，0=不启用（等价于原逻辑）
        self.tail_infer_n = max(0, int(tail_infer_n))

        # 逐层 eps 缩放系数：长度 = n_layers，每层的 eps = eps_per_sample * scale[l]
        # None 表示各层统一阈值（原有逻辑，向后兼容）
        if eps_ratio_schedule is not None:
            sched = list(eps_ratio_schedule)
            if len(sched) != n_layers:
                raise ValueError(
                    f"eps_ratio_schedule 长度 {len(sched)} 与 n_layers {n_layers} 不符"
                )
            self.eps_ratio_schedule = [float(s) for s in sched]
        else:
            self.eps_ratio_schedule = None

        # eps 绝对上限：None=不限制（原逻辑）；设置后所有桶的 eps 不超过此值
        self.eps_hard_cap = float(eps_hard_cap) if eps_hard_cap is not None else None

        self.random_state = random_state
        self.codebooks = [None] * n_layers
        self.num_samples = [None] * n_layers
        self.eps_table = {}                      # price_int -> eps
        self.default_eps = eps_min
        self.reset_log = []                      # [(layer, 重置数量)]
        self.tail_log = []                       # [(layer, 锚定后的极值码字)]
        self.active_log = []                     # [(layer, 参与样本数)]

    # ---------- 从 stat_table 构建样本权重 ----------
    def build_sample_weight(self, price_bucket):
        """
        按价格桶的平均偏差构造样本权重：偏差越大权重越高，
        使 KMeans 的码字重心向"去偏幅度大"的商品倾斜。

        price_bucket: array [N]
        返回: array [N]
        """
        pb = np.asarray(price_bucket).astype(np.int64)
        n = len(pb)
        w = np.ones(n, dtype=float)
        if self.weight_alpha == 0 or n == 0:
            return w

        # 全局兜底：无统计信息的桶用全表平均偏差
        valid_avgs = [v['avg_debias'] for v in self.stat_table.values() if v.get('cnt', 0) > 0]
        global_dev = abs(float(np.mean(valid_avgs)) - 1.0) if valid_avgs else 0.0

        # 按唯一桶计算，避免 200 万次 dict 查找
        uniq, inv = np.unique(pb, return_inverse=True)
        dev = np.empty(len(uniq), dtype=float)
        for j, u in enumerate(uniq):
            info = self.stat_table.get(int(u))
            if info is None or info.get('cnt', 0) == 0:
                dev[j] = global_dev
            else:
                dev[j] = abs(info['avg_debias'] - 1.0)

        w = 1.0 + self.weight_alpha * dev[inv]
        return w

    # ---------- 从 stat_table 构建每个价格桶的 epsilon ----------
    def build_eps_table(self):
        self.eps_table = {}
        for pb, info in self.stat_table.items():
            std = info.get('std_debias', 0.0) or 0.0
            eps = max(self.eps_min, std * self.eps_ratio)
            # ★ eps_hard_cap：防止高 std 桶（如 pb=192/1000）的 eps 过大而提前退出，
            #   导致这批高偏差样本在浅层就"以为收敛"，最终残差集中炸到最深层
            if self.eps_hard_cap is not None:
                eps = min(eps, self.eps_hard_cap)
            self.eps_table[int(pb)] = eps

        all_eps = [v for v in self.eps_table.values() if v > 0]
        self.default_eps = float(np.median(all_eps)) if all_eps else self.eps_min
        return self.eps_table

    def _eps_of(self, price_bucket):
        pb = np.asarray(price_bucket).astype(np.int64)
        if pb.size == 0:
            return np.zeros(0, dtype=float)
        uniq, inv = np.unique(pb, return_inverse=True)
        e = np.array([self.eps_table.get(int(u), self.default_eps) for u in uniq], dtype=float)
        return e[inv]

    # ---------- 内部工具 ----------
    def _assign(self, x, centers, sample_mask):
        """按最近邻把 x 分配给 centers。未参与样本 labels=-1、quant=0。"""
        n = x.shape[0]
        d = np.sum((x[:, None, :] - centers[None, :, :]) ** 2, axis=2)
        nearest = np.argmin(d, axis=1)
        active = sample_mask == 1
        labels = np.where(active, nearest, -1)

        quant = np.zeros((n, self.dim))
        if np.any(active):
            quant[active] = centers[nearest[active]]
        residuals = (x - quant) * sample_mask[:, None]
        return labels, quant, residuals

    def _counts(self, labels, sample_mask, k_size):
        """num_samples = [参与样本数, code0 计数, code1 计数, ...]"""
        active = sample_mask == 1
        cnt = [int(active.sum())]
        for k in range(k_size):
            cnt.append(int(np.sum((labels == k) & active)))
        return cnt

    def _dedup_increasing(self, centers):
        """保证码本严格递增，重复值用微小偏移推开（去重，不影响语义）。"""
        c = centers.copy().flatten()
        span = max(abs(float(c.max())), abs(float(c.min())), 1.0)
        gap = 1e-6 * span
        for i in range(1, len(c)):
            if c[i] <= c[i - 1]:
                c[i] = c[i - 1] + gap
        return c.reshape(-1, self.dim)

    def _anchor_tail(self, centers, x, sample_mask, l):
        """
        ★ 尾部极值锚定（三段式：训练段 + 推理预留段 + 哨兵）

        最后 n_tail 个码字按以下规则分配：
          ① 训练数据段（n_train 个）：在 (core_max, data_max] 等比展开，
             密集覆盖训练数据真实尾部，解决训练集内的低估问题
          ② 推理预留段（tail_infer_n 个）：在 (data_max, tail_hard_max) 等比展开，
             提前为推理期"更大值"预留码字，避免推理时无码字可用
          ③ 哨兵（1 个）：固定钉在 tail_hard_max，作为码本名义上界

        当 tail_infer_n=0 时退化为原始两段结构（向后兼容）。
        """
        k_size = len(centers)
        if (self.tail_reserve <= 0
                or l not in self.tail_reserve_layers
                or self.tail_reserve >= k_size):
            return centers

        n_tail = self.tail_reserve
        centers = centers.copy()

        # core 上界 = 保留区最前一个码字（锚定从它之上开始）
        base = float(centers[k_size - n_tail - 1])

        vals = x[sample_mask == 1, 0]
        data_top = float(np.max(vals)) if vals.size > 0 else base

        # 硬上界：优先用 tail_hard_max，否则退回数据最大值
        hard_top = self.tail_hard_max if self.tail_hard_max is not None else data_top

        # 两种情况都退化，无法展开
        if max(data_top, hard_top) <= base:
            return centers

        gap = 1e-6 * max(abs(base), abs(hard_top), 1.0)

        infer_n = self.tail_infer_n  # 推理预留段码字数

        if hard_top > data_top + gap and n_tail >= 2:
            # 三段式：哨兵占 1，推理预留占 infer_n，剩余给训练数据段
            n_sentinel = 1
            n_infer    = min(infer_n, n_tail - n_sentinel - 1)  # 至少留 1 个给训练段
            n_train    = n_tail - n_sentinel - n_infer

            # ① 训练数据段
            train_anchors = self._geom_points(base, data_top, n_train) if n_train > 0 \
                            else np.array([], dtype=float)

            # ② 推理预留段（在 data_top 和 hard_top 之间）
            if n_infer > 0 and hard_top > data_top + gap:
                infer_anchors = self._geom_points(data_top, hard_top, n_infer + 1)[:-1]
                # 去掉最后一个（那个位置留给哨兵），去掉第一个（避免与 data_top 重复）
                infer_anchors = self._geom_points(data_top, hard_top, n_infer + 2)[1:-1]
            else:
                infer_anchors = np.array([], dtype=float)

            # ③ 哨兵
            anchors = np.concatenate([train_anchors, infer_anchors, [hard_top]])
        else:
            # 数据本身已覆盖上界：全部锚点铺到 hard_top
            anchors = self._geom_points(base, hard_top, n_tail)

        # 长度兜底（极端情况保护）
        if len(anchors) != n_tail:
            anchors = self._geom_points(base, hard_top, n_tail)

        centers[k_size - n_tail:] = anchors.reshape(n_tail, self.dim)
        self.tail_log.append((l, anchors.copy()))
        return centers

    @staticmethod
    def _geom_points(base, top, n):
        """
        在 (base, top] 上取 n 个严格递增的等比点（首点严格大于 base）。
        base<=0 时等比无定义，退化为等距。
        """
        if n <= 0 or top <= base:
            return np.array([], dtype=float)
        if base > 0:
            pts = np.geomspace(base, top, n + 1)[1:]   # 丢掉 base 本身，避免与 core 重复
        else:
            pts = np.linspace(base, top, n + 1)[1:]
        # 保证严格递增，且不超过 top
        gap = 1e-6 * max(abs(base), abs(top), 1.0)
        for i in range(1, n):
            if pts[i] <= pts[i - 1]:
                pts[i] = min(top, pts[i - 1] + gap)
        pts[-1] = top
        return pts

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
        sample_mask = np.ones(n, dtype=int)            # 1=参与本层学习
        quant_codes = np.zeros((n, self.n_layers))

        for l in range(self.n_layers):
            k_size = self.codebook_size[l]
            n_active = int(sample_mask.sum())

            # 全部样本已收敛：用空码本占位，保持结构完整
            if n_active == 0:
                self.codebooks[l] = np.zeros((k_size, self.dim))
                self.num_samples[l] = [0] + [0] * k_size
                continue

            # 1) 加权 KMeans（只用仍在学习的样本）
            km = GuidedKMeans(n_clusters=k_size, random_state=self.random_state + l)
            km.fit_predict(x, sample_mask, sample_weight)

            # 2) 码本按值升序排序，保证 code 有序
            order = np.argsort(km.centers_.flatten())
            centers_sorted = km.centers_[order].copy()

            # 3) 初始分配（用于死码字重置评分基础）
            labels, quant, residuals = self._assign(x, centers_sorted, sample_mask)
            self.num_samples[l] = self._counts(labels, sample_mask, k_size)
            quant_codes[:, l] = quant[:, 0]

            # 4) ★ 死码字重置：用"高权重 × 大残差"的样本值填充低使用率码字
            if self.reset_dead_codes:
                dead_thr = max(1, int(self.dead_code_ratio * n_active))
                active_idx = np.where(sample_mask == 1)[0]
                residual_mag = np.abs(residuals[active_idx, 0])
                score = residual_mag * sample_weight[active_idx]
                top_k = active_idx[np.argsort(-score)[:k_size]]

                reset_cnt = 0
                for k in range(k_size):
                    if self.num_samples[l][k + 1] < dead_thr and reset_cnt < len(top_k):
                        centers_sorted[k] = x[top_k[reset_cnt], 0]
                        reset_cnt += 1

                if reset_cnt > 0:
                    # 重置后码本可能失序，重新排序并再分配，保持可单调解码
                    reorder = np.argsort(centers_sorted.flatten())
                    centers_sorted = centers_sorted[reorder].copy()
                    labels, quant, residuals = self._assign(x, centers_sorted, sample_mask)
                    self.num_samples[l] = self._counts(labels, sample_mask, k_size)
                    quant_codes[:, l] = quant[:, 0]
                    self.reset_log.append((l, reset_cnt))

            # 5) ★ 死码字重置后去重：KMeans/重置可能产生相同码字，导致码本不严格递增
            centers_sorted = self._dedup_increasing(centers_sorted)

            # 6) ★ 尾部极值锚定（放在最后：确保锚点不会被重置/重排覆盖）
            centers_anchored = self._anchor_tail(centers_sorted, x, sample_mask, l)
            if centers_anchored is not centers_sorted:
                centers_sorted = centers_anchored
                labels, quant, residuals = self._assign(x, centers_sorted, sample_mask)
                self.num_samples[l] = self._counts(labels, sample_mask, k_size)
                quant_codes[:, l] = quant[:, 0]

            self.codebooks[l] = centers_sorted

            # 7) 逐样本 eps 判定：残差已足够小的样本退出后续层
            #    若设置了 eps_ratio_schedule，则本层阈值按 scale[l] 缩放：
            #    越深的层阈值越小，样本能继续深入，实现"粒度逐层细化"
            eps_layer = eps_per_sample if self.eps_ratio_schedule is None \
                else eps_per_sample * self.eps_ratio_schedule[l]
            still_learning = np.all(np.abs(residuals) > eps_layer[:, None], axis=1).astype(int)
            sample_mask = sample_mask * still_learning
            x = residuals
            self.active_log.append((l, int(sample_mask.sum())))

    # ---------- 编码 ----------
    def encode(self, inputs, price_bucket):
        """按训练好的多层码本逐层量化，返回 codes / quant_codes / num_samples。"""
        if not self.eps_table:
            self.build_eps_table()
        eps_per_sample = self._eps_of(price_bucket)

        x = np.reshape(np.asarray(inputs, dtype=float), (-1, self.dim))
        n = x.shape[0]
        codes = np.zeros((n, self.n_layers), dtype=int)
        quant_codes = np.zeros((n, self.n_layers))
        num_samples = np.zeros((self.n_layers, max(self.codebook_size) + 1))
        eos = np.zeros(n, dtype=int)                   # 1=已量化到位，不再继续

        for l in range(self.n_layers):
            d = np.sum((x[:, None, :] - self.codebooks[l][None, :, :]) ** 2, axis=2)
            nearest = np.argmin(d, axis=1)
            valid = eos == 0

            codes[:, l] = np.where(valid, nearest, -1)
            quant_codes[:, l] = np.where(valid, self.codebooks[l][nearest, 0], 0.0)

            num_samples[l, 0] = int(valid.sum())
            for k in range(self.codebook_size[l]):
                num_samples[l, k + 1] = int(np.sum((codes[:, l] == k) & valid))

            x = (x - quant_codes[:, [l]]) * valid[:, None]
            # ★ 自适应 eos：该样本自己的残差已小于 eps 就停
            #    与 train 保持一致：若设置了 eps_ratio_schedule 则按层缩放
            eps_layer = eps_per_sample if self.eps_ratio_schedule is None \
                else eps_per_sample * self.eps_ratio_schedule[l]
            eos = np.all(np.abs(x) < eps_layer[:, None], axis=1).astype(int)

        return codes, quant_codes, num_samples
