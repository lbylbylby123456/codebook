# main_4layer.py  ——  4 层 Guided Residual Codebook 训练入口
"""
5 层码本设计哲学
================
每层量化上一层的残差，粒度逐层细化：

  Layer 1  量化原始 debias_label
           · 范围 [0, 100]，尾部 4 个码字强制锚定：
             前 3 个铺满真实数据尾部 (core_max ~ data_max)，
             最后 1 个钉在 TAIL_HARD_MAX=100，作为上界哨兵
           · 样本在此层的残差通常在 ±0.5 以内

  Layer 2  量化 L1 残差（中等幅度修正）
           · 覆盖 L1 粗量化误差（~±2）
           · 残差已较小的样本（eps 满足）在此层退出

  Layer 3  量化 L2 残差（中细粒度）
           · 覆盖 ±0.5 量级的残差
           · 绝大多数普通价格段商品在此退出

  Layer 4  量化 L3 残差（细粒度，主要服务高偏差商品）
           · std 大的价格桶（pb=1000, pb=192 等）此时仍在学习
           · eps_ratio 调小到 0.01，让更多样本继续参与

  Layer 5  量化 L4 残差（极细精修，兜底层）
           · 仅极端样本（极高 std 价格桶）残存
           · 进一步降低 max_err

参数差异 vs 3 层版本：
  · n_layers  : 3  → 5
  · eps_ratio : 0.02 → 0.01   （让残差传播到更深层）
  · eps_min   : 1e-3 不变
  · 其余参数完全相同，保持可比性
"""
import os, pickle, logging, sys
import numpy as np
import pandas as pd
from GuidedResCluster import GuidedResCluster

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s: %(message)s',
                    stream=sys.stderr)

# ─── 数据路径 ─────────────────────────────────────────────────────────────────
STAT_CSV   = "stat_new.csv"       # 统计表：price_int → avg_debias / std_debias / cnt
DETAIL_CSV = "detail_new.csv"     # 明细表：price_int, debias_label（逐样本）
SAVE_DIR   = "./models_guided_4layer"  # ★ 独立输出目录，不覆盖其他版本

# ─── 码本配置 ─────────────────────────────────────────────────────────────────
#
#  5 层统一使用 64 个码字，总共 5×64=320 个码字。
#
#  ★ 尾部极值锚定（tail_reserve=4, tail_hard_max=100.0）：
#    仅第一层启用，最后 4 个码字两段式硬锚定：
#      · 前 3 个：在 (core_max, data_max] 等比展开，精确覆盖真实数据尾部
#      · 最后 1 个：固定钉在 TAIL_HARD_MAX（100.0），设计上界哨兵
#
CODEBOOK_SIZE  = [64, 64, 64, 64]   # 4 层均为 64
TAIL_HARD_MAX  = 100.0                   # ★ 第一层码本名义上界，与训练数据范围无关

# ─── 工具函数 ─────────────────────────────────────────────────────────────────
def load_stat_table(path, granularity='round(price)'):
    """
    读取统计 CSV，返回 dict: price_int -> {avg_debias, std_debias, cnt}。
    兼容列名带表别名前缀（如 "_u2.granularity"）的情况。
    """
    df = pd.read_csv(path)
    df = df.rename(columns={c: c.split('.')[-1] for c in df.columns})

    if 'granularity' in df.columns:
        df = df[df['granularity'] == granularity]
    else:
        logging.warning("stat CSV 中未找到 granularity 列，跳过粒度过滤")

    df = df.dropna(subset=['price_int'])

    tbl = {}
    for _, r in df.iterrows():
        tbl[int(r['price_int'])] = {
            'avg_debias': float(r['avg_debias_label']),
            'std_debias': float(r['std_debias_label']) if not pd.isna(r['std_debias_label']) else 0.0,
            'cnt':        int(r['cnt']),
        }

    logging.info(
        f"stat_table 加载 {len(tbl)} 个价格桶，"
        f"price_int 范围 {min(tbl) if tbl else 'N/A'} ~ {max(tbl) if tbl else 'N/A'}"
    )
    return tbl


# ─── 主流程 ───────────────────────────────────────────────────────────────────
def main():
    # ── 1. 加载数据 ────────────────────────────────────────────────────────────
    stat_table = load_stat_table(STAT_CSV)

    df = pd.read_csv(DETAIL_CSV)
    df = df.dropna(subset=['debias_label', 'price_int'])
    logging.info(f"原始样本数: {len(df)}")

    MAX_ROWS = 2_000_000
    if len(df) > MAX_ROWS:
        df = df.sample(n=MAX_ROWS, random_state=42).reset_index(drop=True)
        logging.info(f"抽样后样本数: {len(df)}")

    # ── 2. 截断极端值 ──────────────────────────────────────────────────────────
    q99   = float(df['debias_label'].quantile(0.99))
    q999  = float(df['debias_label'].quantile(0.999))
    q9999 = float(df['debias_label'].quantile(0.9999))
    q_max = float(df['debias_label'].max())
    logging.info(
        f"debias_label 分布: P99={q99:.4f}, P99.9={q999:.4f}, "
        f"P99.99={q9999:.4f}, max={q_max:.4f}"
    )

    clip = TAIL_HARD_MAX
    logging.info(f"debias_label clip 上限: TAIL_HARD_MAX = {clip:.4f}")
    df['debias_label'] = df['debias_label'].clip(upper=clip)

    X            = df[['debias_label']].values.astype(float)   # [N, 1]
    price_bucket = df['price_int'].values.astype(int)           # [N]

    # ── 3. 构建 5 层模型 ────────────────────────────────────────────────────────
    #
    # 与 3 层版本的关键差异：
    #   · n_layers=5
    #   · eps_ratio=0.01（退出阈值基准，控制各层收敛严格程度）
    #
    # ★ 关键修复 — eps_hard_cap=0.5：
    #   原版 eps = std_debias * eps_ratio，对高偏差桶（如 pb=192, std≈302）
    #   eps 会高达 6.04，远超实际量化误差需求，导致这批样本在 L2/L3 就
    #   "以为收敛"而退出，最终 L5 只剩 14 个人，码本被极端残差主导，
    #   粒度反而比 L4 粗了 90 倍，完全破坏"越来越细"的原则。
    #
    #   加了 hard_cap=0.5 后，任何桶的退出阈值都不超过 0.5，大偏差桶
    #   会持续参与更深层量化，各层样本分布更均衡，粒度真正逐层细化。
    #
    EPS_HARD_CAP = 0.5    # ★ eps 绝对上限；None=不限制（等价于旧版行为）

    model = GuidedResCluster(
        n_layers=4,
        codebook_size=CODEBOOK_SIZE,        # [64, 64, 64, 64, 64]
        dim=1,
        stat_table=stat_table,
        weight_alpha=1.0,                   # 0=不加权；1=标准；2=更激进
        eps_ratio=0.01,                     # 退出阈值基准：eps = std * ratio（再被 hard_cap 截断）
        eps_min=1e-3,                       # 最小 eps：避免数值噪声样本无限深入
        eps_hard_cap=EPS_HARD_CAP,          # ★ 截断过大的 eps，防止高偏差桶提前"假收敛"
        dead_code_ratio=0.005,              # 参与样本占比 < 0.5% 视为死码字
        reset_dead_codes=True,
        tail_reserve=12,                    # ★ 增大：11 个锚点密集覆盖 [base,data_max]+ 1 个哨兵
        tail_hard_max=TAIL_HARD_MAX,        # ★ 锚定上界固定为 100.0，与数据范围无关
        tail_reserve_layers=(0,),           # ★ 仅对第一层启用（L2~L5 处理残差无需锚定）
        eps_ratio_schedule=None,            # 各层同阈值（hard_cap 已解决深层分布问题）
        random_state=42,
    )

    # ── 4. 训练 ────────────────────────────────────────────────────────────────
    weight = model.build_sample_weight(price_bucket)
    logging.info(
        f"样本权重: min={weight.min():.3f}, max={weight.max():.3f}, mean={weight.mean():.3f}"
    )

    model.train(X, price_bucket, sample_weight=weight)
    codes, quant_codes, num_samples = model.encode(X, price_bucket)

    # ── 5. 诊断输出 ────────────────────────────────────────────────────────────
    logging.info("=== 死码字重置记录 ===")
    if model.reset_log:
        for l, cnt in model.reset_log:
            logging.info(f"  Layer {l+1}: 重置 {cnt} 个死码字")
    else:
        logging.info("  无死码字重置")

    logging.info("=== 尾部极值锚定记录 ===")
    if model.tail_log:
        for l, anchors in model.tail_log:
            logging.info(f"  Layer {l+1}: 锚定极值码字 = {np.round(anchors, 5).tolist()}")
    else:
        logging.info("  无尾部锚定")

    # ── 5a. 码本粒度分析（核心：验证粒度是否逐层细化、正负是否对称）──────────────
    logging.info("=== 码本粒度 & 对称性（核心指标）===")
    logging.info(f"  {'层':<6} {'范围':>10} {'粒度/码字':>12} {'负侧':>10} {'正侧':>10} {'对称比':>8} {'状态':>5}")
    prev_resolution = None
    all_good = True
    for l in range(model.n_layers):
        cb        = np.asarray(model.codebooks[l]).flatten()
        n_active  = int(num_samples[l, 0])
        k_size    = model.codebook_size[l]
        cb_range  = cb[-1] - cb[0]
        resolution= cb_range / k_size
        neg_range = abs(cb[0]) if cb[0] < 0 else 0.0
        pos_range = cb[-1]    if cb[-1] > 0 else 0.0
        sym_ratio = pos_range / neg_range if neg_range > 0 else float('inf')

        # 退化层：参与样本数 < 码字数时，粒度/对称比均无意义，不纳入单调判断
        if n_active < k_size:
            status = "⚠️ 退化"
            mono_ok = "⚠️"
        elif prev_resolution is None or resolution <= prev_resolution:
            status = "✅"
            mono_ok = "✅"
        else:
            status = "❌"
            mono_ok = "❌"
            all_good = False

        prev_resolution = resolution if n_active >= k_size else prev_resolution
        logging.info(
            f"  L{l+1} {mono_ok}  range={cb_range:>8.4f}  "
            f"res={resolution:>10.5f}  "
            f"neg={neg_range:>8.4f}  pos={pos_range:>8.4f}  "
            f"sym={sym_ratio:>6.2f}x  n={n_active}"
        )
    if all_good:
        logging.info("  ✅ 粒度严格逐层细化（退化层不纳入判断）")
    else:
        logging.info("  ❌ 存在层级粒度倒退，请检查 tail_reserve / eps 设置")

    # ── 5b. 利用率（体现样本分布均衡度）──────────────────────────────────────────
    logging.info("=== 各层码字利用率 ===")
    layer_names = [
        "L1 信号层  宽幅量化",
        "L2 中等修正 残差量化",
        "L3 中细粒度 残差量化",
        "L4 细粒度   兜底高偏差",
    ]
    for l in range(model.n_layers):
        active = int(num_samples[l, 0])
        used   = int(np.sum(num_samples[l, 1:model.codebook_size[l]+1] > 0))
        util   = used / model.codebook_size[l]
        name   = layer_names[l] if l < len(layer_names) else f"Layer{l+1}"
        logging.info(
            f"  {name}: 参与={active:,}, 码字={used}/{model.codebook_size[l]}, 利用率={util:.2%}"
        )

    logging.info("=== eps 分布（截断后）===")
    eps_vals = np.array(list(model.eps_table.values()))
    logging.info(
        f"  min={eps_vals.min():.5f}, median={np.median(eps_vals):.5f}, "
        f"max={eps_vals.max():.5f}  (hard_cap={EPS_HARD_CAP})"
    )

    logging.info("=== 重构误差 ===")
    recon = np.sum(quant_codes, axis=1)
    err   = recon - X[:, 0]
    logging.info(f"  全局 MAE={np.mean(np.abs(err)):.5f}, max_err={np.max(np.abs(err)):.5f}")
    per_bucket_mae = (
        pd.DataFrame({'pb': price_bucket, 'err': np.abs(err)})
        .groupby('pb')['err'].mean()
    )
    logging.info(f"  MAE 最高 5 个价格段:\n{per_bucket_mae.nlargest(5).to_string()}")

    # ── 5c. 逐层覆盖情况 ──────────────────────────────────────────────────────────
    logging.info("=== 逐层累计覆盖情况 ===")
    total = len(X)
    prev_active = total
    for l in range(model.n_layers):
        cur_active = int(num_samples[l, 0])
        exited = prev_active - cur_active
        logging.info(
            f"  Layer{l+1}: 剩余={cur_active:,} ({cur_active/total:.1%}), "
            f"本层退出={exited:,} ({exited/total:.1%})"
        )
        prev_active = cur_active

    # ── 6. 保存 pkl ────────────────────────────────────────────────────────────
    os.makedirs(SAVE_DIR, exist_ok=True)
    pkl_path = os.path.join(SAVE_DIR, "codebooks_4layer.pkl")
    with open(pkl_path, 'wb') as f:
        pickle.dump({
            'codebooks':   model.codebooks,
            'num_samples': model.num_samples,
            'eps_table':   model.eps_table,
            'stat_table':  stat_table,
            'clip_value':  clip,
            'config': {
                'n_layers':          4,
                'codebook_size':     CODEBOOK_SIZE,
                'dim':               1,
                'weight_alpha':      1.0,
                'eps_ratio':         0.01,
                'eps_min':           1e-3,
                'eps_hard_cap':      EPS_HARD_CAP,
                'dead_code_ratio':   0.005,
                'clip_value_source': f'TAIL_HARD_MAX={TAIL_HARD_MAX}',
                'tail_reserve':      12,
                'tail_hard_max':     TAIL_HARD_MAX,
            },
        }, f)
    logging.info(f"模型已保存: {pkl_path}")

    # ── 7. 导出 Excel（单 Sheet，5 行 × 64 列）────────────────────────────────
    from openpyxl import Workbook
    from openpyxl.styles import PatternFill, Font, Alignment
    from openpyxl.utils import get_column_letter

    xlsx_path = os.path.join(SAVE_DIR, "codebooks_4layer.xlsx")
    wb = Workbook()
    ws = wb.active
    ws.title = 'codebooks_4layer'

    n_codes = max(CODEBOOK_SIZE)   # = 64
    layers  = [np.asarray(model.codebooks[l]).reshape(-1) for l in range(model.n_layers)]

    # 行颜色：5 层用不同浅色区分，便于视觉识别粒度
    ROW_FILLS = [
        PatternFill("solid", fgColor="D9EAD3"),   # L1 绿：宽幅信号层
        PatternFill("solid", fgColor="CFE2F3"),   # L2 蓝：中等修正
        PatternFill("solid", fgColor="FFF2CC"),   # L3 黄：中细粒度
        PatternFill("solid", fgColor="FCE5CD"),   # L4 橙：细粒度
    ]

    # 第 1 行：表头
    header_row = ['idx'] + list(range(n_codes))
    ws.append(header_row)
    for cell in ws[1]:
        cell.font      = Font(bold=True)
        cell.alignment = Alignment(horizontal='center')

    # 第 2~6 行：每层一行
    for l, cb in enumerate(layers):
        row = [f'Layer{l+1}'] + [round(float(v), 6) for v in cb]
        row += [''] * (n_codes - len(cb))   # 防止金字塔时列不足
        ws.append(row)

        # 数值格式 + 行颜色
        row_idx = l + 2
        fill = ROW_FILLS[l] if l < len(ROW_FILLS) else None
        for col_idx, cell in enumerate(ws[row_idx]):
            if fill:
                cell.fill = fill
            if col_idx > 0 and isinstance(cell.value, float):
                cell.number_format = '0.0000'
            cell.alignment = Alignment(horizontal='center')

    # 列宽：标签列略宽
    ws.column_dimensions['A'].width = 12
    for i in range(n_codes):
        ws.column_dimensions[get_column_letter(2 + i)].width = 9

    wb.save(xlsx_path)
    logging.info(f"Excel 已导出: {xlsx_path}（单 Sheet，4 行 × {n_codes} 列）")


if __name__ == "__main__":
    main()
