# train_guided.py  ——  Guided Residual Codebook 训练入口
import os, pickle, logging, sys
import numpy as np
import pandas as pd
from GuidedResCluster import GuidedResCluster

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s', stream=sys.stderr)

# ─── 数据路径 ────────────────────────────────────────────────────────────────
STAT_CSV   = "stat_new.csv"    # 统计表：price_int → avg_debias / std_debias / cnt
DETAIL_CSV = "detail_new.csv"  # 明细表：price_int, debias_label（逐样本）
SAVE_DIR   = "./models_guided" # 输出目录

# ─── 码本配置 ────────────────────────────────────────────────────────────────
#
#  三层统一使用 64 个码字。
#
#  ★ 尾部极值锚定（tail_reserve=4, tail_hard_max=100.0）：
#    第一层码本最后 4 个码字不由 KMeans 密度决定，改为两段式硬锚定：
#      · 前 3 个：在 (core_max, data_max] 等比展开，精确覆盖真实数据尾部
#      · 最后 1 个：固定钉在 TAIL_HARD_MAX（100.0），作为码本名义上界哨兵
#    这样既保留了 16~49 这段真实尾部的分辨率，又保证码本上界达到 100+，
#    推理时高偏差样本不会被全部钳到同一个码字上。
#
CODEBOOK_SIZE  = [64, 64, 64]
TAIL_HARD_MAX  = 100.0   # ★ 第一层码本名义上界（设计哨兵），与训练数据范围无关


# ─── 工具函数 ─────────────────────────────────────────────────────────────────
def load_stat_table(path, granularity='round(price)'):
    """
    读取统计 CSV，返回 dict: price_int -> {avg_debias, std_debias, cnt}。
    兼容列名带表别名前缀（如 "_u2.granularity"）的情况。
    """
    df = pd.read_csv(path)
    # 去除列名中的表别名前缀（"_u2.xxx" → "xxx"）
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

    # 抽样：纯 numpy KMeans 建议先以 ≤200 万行试跑
    MAX_ROWS = 2_000_000
    if len(df) > MAX_ROWS:
        df = df.sample(n=MAX_ROWS, random_state=42).reset_index(drop=True)
        logging.info(f"抽样后样本数: {len(df)}")

    # ── 2. 截断极端值 ──────────────────────────────────────────────────────────
    # 打印关键分位数，便于评估 clip 设置是否合理
    q99    = float(df['debias_label'].quantile(0.99))
    q999   = float(df['debias_label'].quantile(0.999))
    q9999  = float(df['debias_label'].quantile(0.9999))
    q_max  = float(df['debias_label'].max())
    logging.info(
        f"debias_label 分布: P99={q99:.4f}, P99.9={q999:.4f}, "
        f"P99.99={q9999:.4f}, max={q_max:.4f}"
    )

    # clip 使用 TAIL_HARD_MAX（100.0），确保训练数据和推理数据都在
    # [0, 100] 范围内，与码本上界对齐；同时过滤掉 max=49.6 这类噪声极值。
    clip = TAIL_HARD_MAX
    logging.info(f"debias_label clip 上限: TAIL_HARD_MAX = {clip:.4f}")
    df['debias_label'] = df['debias_label'].clip(upper=clip)

    X            = df[['debias_label']].values.astype(float)  # [N, 1]
    price_bucket = df['price_int'].values.astype(int)          # [N]

    # ── 3. 构建模型 ────────────────────────────────────────────────────────────
    model = GuidedResCluster(
        n_layers=3,
        codebook_size=CODEBOOK_SIZE,        # [64, 64, 64]
        dim=1,
        stat_table=stat_table,
        weight_alpha=1.0,                   # 0=不加权；1=标准；2=更激进
        eps_ratio=0.02,
        eps_min=1e-3,
        dead_code_ratio=0.005,              # 参与样本占比 < 0.5% 视为死码字
        reset_dead_codes=True,
        tail_reserve=4,                     # ★ 第一层最后 4 个码字强制锚定到极值
        tail_hard_max=TAIL_HARD_MAX,        # ★ 锚定上界固定为 100.0，与数据范围无关
        tail_reserve_layers=(0,),           # ★ 仅对第一层启用（Layer2/3 处理残差无需锚定）
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

    logging.info("=== 各层码字利用率 ===")
    for l in range(model.n_layers):
        active = int(num_samples[l, 0])
        used   = int(np.sum(num_samples[l, 1:model.codebook_size[l]+1] > 0))
        util   = used / model.codebook_size[l]
        logging.info(
            f"  Layer {l+1}: 参与样本={active:,}, "
            f"使用码字={used}/{model.codebook_size[l]}, 利用率={util:.2%}"
        )

    logging.info("=== 各层 epsilon 分布 ===")
    eps_vals = np.array(list(model.eps_table.values()))
    logging.info(
        f"  eps: min={eps_vals.min():.5f}, "
        f"median={np.median(eps_vals):.5f}, max={eps_vals.max():.5f}"
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

    # ── 6. 保存 pkl ────────────────────────────────────────────────────────────
    os.makedirs(SAVE_DIR, exist_ok=True)
    pkl_path = os.path.join(SAVE_DIR, "codebooks.pkl")
    with open(pkl_path, 'wb') as f:
        pickle.dump({
            'codebooks':   model.codebooks,
            'num_samples': model.num_samples,
            'eps_table':   model.eps_table,
            'stat_table':  stat_table,
            'clip_value':  clip,
            'config': {
                'n_layers':         3,
                'codebook_size':    CODEBOOK_SIZE,
                'dim':              1,
                'weight_alpha':     1.0,
                'eps_ratio':        0.02,
                'clip_value_source': f'TAIL_HARD_MAX={TAIL_HARD_MAX}',
                'tail_reserve':      4,
                'tail_hard_max':     TAIL_HARD_MAX,
            },
        }, f)
    logging.info(f"模型已保存: {pkl_path}")

    # ── 7. 导出 Excel（单 Sheet，三行，每行 64 列）─────────────────────────────
    from openpyxl import Workbook

    xlsx_path = os.path.join(SAVE_DIR, "codebooks.xlsx")
    wb = Workbook()
    ws = wb.active
    ws.title = 'codebooks'

    # 每层码本展平为 (codebook_size[l],) 的一维向量，统一取 max size 对齐列数
    n_codes = max(CODEBOOK_SIZE)  # = 64
    layers = [np.asarray(model.codebooks[l]).reshape(-1) for l in range(model.n_layers)]

    # 第 1 行：表头 idx 0 ~ 63
    ws.append(['idx'] + list(range(n_codes)))

    # 第 2~4 行：每层一行，不足 n_codes 的补空
    for l, cb in enumerate(layers):
        row = [f'Layer{l+1}'] + [float(v) for v in cb]
        # 若该层码本长度 < n_codes（金字塔时），补空白对齐
        row += [''] * (n_codes - len(cb))
        ws.append(row)

    # 数值格式：4 位小数
    for row in ws.iter_rows(min_row=2, min_col=2, max_col=n_codes + 1):
        for cell in row:
            if isinstance(cell.value, float):
                cell.number_format = '0.0000'

    # 列宽
    ws.column_dimensions['A'].width = 10
    for i in range(n_codes):
        col_letter = ws.cell(row=1, column=2 + i).column_letter
        ws.column_dimensions[col_letter].width = 10

    wb.save(xlsx_path)
    logging.info(f"Excel 已导出: {xlsx_path}（单 Sheet，3 行 × {n_codes} 列）")


if __name__ == "__main__":
    main()
