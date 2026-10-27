# train_guided.py
import os, pickle, logging, sys
import numpy as np
import pandas as pd
from GuidedResCluster import GuidedResCluster

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s', stream=sys.stderr)

STAT_CSV   = "stat.csv"        # UNION ALL 输出，只保留 granularity='round(price/5.1)'
DETAIL_CSV = "detail.csv"      # 逐样本：price_int, debias_label
SAVE_DIR   = "./models_guided"

def load_stat_table(path, granularity='round(price)'):              # ← ①默认值改成 round(price)，跟你的 CSV 对齐
    df = pd.read_csv(path)                                           # ← ②读 CSV（不变）

    # ★★★ 新增这一行：把 "_u2.granularity" 变成 "granularity" ★★★
    df = df.rename(columns={c: c.split('.')[-1] for c in df.columns})

    if 'granularity' in df.columns:                                  # ← ③加 if 判断，列存在才过滤
        df = df[df['granularity'] == granularity]
    else:
        logging.warning("stat.csv 没有 granularity 列，跳过粒度过滤")
    df = df.dropna(subset=['price_int'])                             # ← ④单独 dropna（不变）

    tbl = {}
    for _, r in df.iterrows():
        tbl[int(r['price_int'])] = {
            'avg_debias': float(r['avg_debias_label']),
            'std_debias': float(r['std_debias_label']) if not pd.isna(r['std_debias_label']) else 0.0,
            'cnt': int(r['cnt']),
        }
    logging.info(f"stat_table 加载 {len(tbl)} 个价格桶, price_int 范围 {min(tbl) if tbl else 'N/A'}~{max(tbl) if tbl else 'N/A'}")
    return tbl

def main():
    stat_table = load_stat_table(STAT_CSV)
    df = pd.read_csv(DETAIL_CSV)
    df = df.dropna(subset=['debias_label', 'price_int'])
    logging.info(f"detail 样本数: {len(df)}")

    # 抽样（纯 numpy KMeans 建议先 ≤200 万行试跑）
    MAX_ROWS = 2_000_000
    if len(df) > MAX_ROWS:
        df = df.sample(n=MAX_ROWS, random_state=42).reset_index(drop=True)
        logging.info(f"抽样到 {len(df)} 行用于试跑")

    # 截断极端值（debias_label 通常在 [0, 2]，按 P99 截断）
    clip = float(df['debias_label'].quantile(0.99))
    logging.info(f"debias_label clip 到 P99={clip:.4f}")
    df['debias_label'] = df['debias_label'].clip(upper=clip)

    X = df[['debias_label']].values.astype(float)       # [N, 1]
    price_bucket = df['price_int'].values.astype(int)   # [N]

    model = GuidedResCluster(
        n_layers=3,
        codebook_size=[48, 48, 48],
        dim=1,
        stat_table=stat_table,
        weight_alpha=1.0,          # 0 = 不加权；1 = 标准；2 = 更激进
        eps_ratio=0.02,
        eps_min=1e-3,
        dead_code_ratio=0.005,
        reset_dead_codes=True,
        random_state=42,
    )

    weight = model.build_sample_weight(price_bucket)
    logging.info(f"权重范围: min={weight.min():.3f}, max={weight.max():.3f}, mean={weight.mean():.3f}")

    model.train(X, price_bucket, sample_weight=weight)
    codes, quant_codes, num_samples = model.encode(X, price_bucket)

    # ---------- 诊断输出 ----------
    logging.info("=== 死码字重置日志 ===")
    for l, cnt in model.reset_log:
        logging.info(f"layer {l+1}: 重置 {cnt} 个死码字")

    logging.info("=== 各层码字利用率 ===")
    for l in range(model.n_layers):
        active = num_samples[l, 0]
        used = int(np.sum(num_samples[l, 1:model.codebook_size[l]+1] > 0))
        logging.info(f"layer {l+1}: 参与样本={int(active)}, 使用码字={used}/{model.codebook_size[l]}, "
                     f"利用率={used/model.codebook_size[l]:.2%}")

    logging.info("=== 各层 epsilon 分布 ===")
    eps_vals = np.array(list(model.eps_table.values()))
    logging.info(f"eps min={eps_vals.min():.5f}, median={np.median(eps_vals):.5f}, max={eps_vals.max():.5f}")

    # 重构误差（整体 + 按价格段）
    recon = np.sum(quant_codes, axis=1)
    err = recon - X[:, 0]
    logging.info(f"MAE={np.mean(np.abs(err)):.5f}, max_err={np.max(np.abs(err)):.5f}")
    per_bucket_mae = pd.DataFrame({'pb': price_bucket, 'err': np.abs(err)}).groupby('pb')['err'].mean()
    logging.info(f"按价格段 MAE 最高 5 段:\n{per_bucket_mae.nlargest(5).to_string()}")

    # 保存
    os.makedirs(SAVE_DIR, exist_ok=True)
    with open(os.path.join(SAVE_DIR, "codebooks.pkl"), 'wb') as f:
        pickle.dump({
            'codebooks': model.codebooks,
            'num_samples': model.num_samples,
            'eps_table': model.eps_table,
            'stat_table': stat_table,
            'clip_value': clip,
            'config': {'n_layers': 3, 'codebook_size': [48,48,48], 'dim': 1,
                       'weight_alpha': 1.0, 'eps_ratio': 0.02},
        }, f)
    logging.info(f"已保存到 {SAVE_DIR}/codebooks.pkl")

    from openpyxl import Workbook

    xlsx_path = os.path.join(SAVE_DIR, "codebooks.xlsx")
    wb = Workbook()
    ws = wb.active
    ws.title = 'codebooks'

    # 每个码本先展平成 (48,)
    layers = [np.asarray(model.codebooks[l]).reshape(-1) for l in range(model.n_layers)]

    # 第 1 行：表头 idx 0 ~ 47
    ws.append(['idx'] + list(range(len(layers[0]))))

    # 第 2~4 行：Layer1 / Layer2 / Layer3，每行 48 个值
    for l, cb in enumerate(layers):
        ws.append([f'Layer{l+1}'] + [float(v) for v in cb])

    # 数值格式：4 位小数
    for row in ws.iter_rows(min_row=2, min_col=2):
        for c in row:
            c.number_format = '0.0000'

    # 列宽
    ws.column_dimensions['A'].width = 10
    for i in range(len(layers[0])):
        ws.column_dimensions[ws.cell(row=1, column=2+i).column_letter].width = 12

    wb.save(xlsx_path)
    logging.info(f"已导出 Excel: {xlsx_path}")

if __name__ == "__main__":
    main()