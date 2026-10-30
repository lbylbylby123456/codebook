# main_4layer_v2.py  ——  4 层 Guided Residual Codebook 训练入口（修订版）
"""
v2 vs v1 的核心改动
=====================

问题诊断（来自 v1 运行数据）
─────────────────────────────
1. debias_label 在 [10, 50] 区间存在系统性低估（mean_err ≈ −0.019）
   根因：L1 的 tail_reserve=12 中，44.28 → 49.64 之间 gap≈5.4，
   该区间样本被强制吸引到 49.64，产生约 −2~−5 的负残差，
   而 L2 负侧最大只有 −1.05，根本兜不住 → 低估残留到最终输出。

2. 推理期如果 debias_label > 49.64（训练集 max），
   L1 码本从 49.64 直接跳到哨兵 100.0，中间 50 个单位完全空白，
   推理期高值商品全部被拉到 100.0，产生巨大负残差。

v2 改动（三处）
─────────────────────────────
① L1 tail 三段式锚定（tail_infer_n=4）
   · 训练数据段：12-1-4-1 = 6 个码字覆盖 (core_max, 49.64]，gap 从 5.4 压到 ~1.6
   · 推理预留段：4 个码字均匀覆盖 (49.64, 300)，提前为推理大值铺路
   · 哨兵：1 个固定在 300.0
   · tail_reserve 从 12 → 13（多出 1 个给推理段）

② TAIL_HARD_MAX 从 100 → 300
   推理期理论最大 debias_label 更大时，码本上界也有保障。
   同时 clip 上限也改为 300，与码本上界对齐。

③ 四层码本统一 COODEBOOK_SIZE=[64, 64, 64, 64]
   L3/L4 保持 64 码字，与 L1/L2 完全一致。
   每层编码空间宽度统一，推理端固定按 64 取码，无需按层特判。
   精度提升全部来自 tail 三段式锚定 + 上界扩容，不靠加宽码本。

各层设计职责
─────────────────────────────
  L1  量化原始 debias_label [0, 300]
      · core 区（code[0..50]）：KMeans 拟合 [0, ~14]，覆盖 99%+ 的普通商品
      · tail 区（code[51..63]）：三段式锚定
          - code[51..57]（6 个）：等比铺满 [14, 49.64]，修复低估
          - code[58..61]（4 个）：等比铺满 (49.64, 300)，推理预留
          - code[62]（1 个）：哨兵 300.0
      after L1 残差 ≤ ±1（训练段）

  L2  量化 L1 残差（±1 量级，中等修正）
      · 64 个码字覆盖 [−1.05, +2.24]，修复 L1 粗量化误差
      · ~16% 普通样本在此退出

  L3  量化 L2 残差（±0.3 量级，中细粒度）
      · 64 个码字覆盖 L2 之后的残差分布
      · ~76% 样本在此退出（大量高偏差桶也在此收敛）

  L4  量化 L3 残差（±0.05 量级，细粒度兜底）
      · 64 个码字 → 粒度 ~0.001/code
      · 仅少数极端样本参与，作为最后一道兼底
"""
import os, pickle, logging, sys
import numpy as np
import pandas as pd
from GuidedResCluster import GuidedResCluster

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s: %(message)s',
                    stream=sys.stderr)

# ─── 数据路径 ─────────────────────────────────────────────────────────────────
STAT_CSV   = "stat_new.csv"
DETAIL_CSV = "detail_new.csv"
SAVE_DIR   = "./models_guided_4layer_v2"  # ★ 独立目录，不覆盖 v1

# ─── 码本配置 ─────────────────────────────────────────────────────────────────
#
#  4 层统一使用 64 个码字，总共 4×64=256 个码字。
#  每层编码空间宽度一致，推理端无需按层特判。
#
CODEBOOK_SIZE = [64, 64, 64, 64]

#  ★ 推理上界从 100 扩充到 300
#    v1 中 data_max=49.64，上界 100 对推理期 >100 的商品仍有风险；
#    v2 改为 300，三段式锚定在 49.64~300 之间铺路，推理安全边际大幅提升
TAIL_HARD_MAX = 300.0

#  ★ 推理预留段码字数（新参数）
#    tail_reserve=13 中：6 个给训练数据段 + 4 个推理预留 + 1 个哨兵（+2 给 infer 段）
#    实际分配由 GuidedResCluster._anchor_tail 自动计算
TAIL_RESERVE   = 13
TAIL_INFER_N   = 4     # 在 (data_max, tail_hard_max) 之间额外插入 4 个推理预留码字


# ─── 工具函数 ─────────────────────────────────────────────────────────────────
def load_stat_table(path, granularity='round(price)'):
    """读取统计 CSV，返回 dict: price_int -> {avg_debias, std_debias, cnt}。"""
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

    # ── 2. 分析数据分布 & clip ────────────────────────────────────────────────
    q99   = float(df['debias_label'].quantile(0.99))
    q999  = float(df['debias_label'].quantile(0.999))
    q9999 = float(df['debias_label'].quantile(0.9999))
    q_max = float(df['debias_label'].max())
    logging.info(
        f"debias_label 分布: P99={q99:.4f}, P99.9={q999:.4f}, "
        f"P99.99={q9999:.4f}, max={q_max:.4f}"
    )

    # ★ clip 对齐到 TAIL_HARD_MAX=300（v1 是 100）
    clip = TAIL_HARD_MAX
    logging.info(f"debias_label clip 上限: TAIL_HARD_MAX = {clip:.1f}（v1 为 100.0）")
    df['debias_label'] = df['debias_label'].clip(upper=clip)

    X            = df[['debias_label']].values.astype(float)   # [N, 1]
    price_bucket = df['price_int'].values.astype(int)           # [N]

    # ── 3. 构建 4 层模型（v2 参数）────────────────────────────────────────────
    EPS_HARD_CAP = 0.5    # eps 绝对上限，防止高偏差桶提前"假收敛"

    model = GuidedResCluster(
        n_layers=4,
        codebook_size=CODEBOOK_SIZE,        # [64, 64, 64, 64]，每层统一 64 码字
        dim=1,
        stat_table=stat_table,
        weight_alpha=1.0,
        eps_ratio=0.01,
        eps_min=1e-3,
        eps_hard_cap=EPS_HARD_CAP,          # ★ 防高偏差桶假收敛
        dead_code_ratio=0.005,
        reset_dead_codes=True,
        tail_reserve=TAIL_RESERVE,          # ★ 13 个锚点（↑ from 12）
        tail_hard_max=TAIL_HARD_MAX,        # ★ 推理上界 300.0（↑ from 100.0）
        tail_reserve_layers=(0,),           # 仅第一层启用锚定
        tail_infer_n=TAIL_INFER_N,          # ★ 4 个推理预留码字（新增）
        eps_ratio_schedule=None,
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
            logging.info(f"  Layer {l+1}: 锚点({len(anchors)}个) = {np.round(anchors, 3).tolist()}")
    else:
        logging.info("  无尾部锚定")

    # ── 5a. 码本粒度 & 对称性 ────────────────────────────────────────────────
    logging.info("=== 码本粒度 & 对称性（核心指标）===")
    prev_resolution = None
    all_good = True
    for l in range(model.n_layers):
        cb       = np.asarray(model.codebooks[l]).flatten()
        n_active = int(num_samples[l, 0])
        k_size   = model.codebook_size[l]
        cb_range = cb[-1] - cb[0]
        resolution = cb_range / k_size
        neg_range  = abs(cb[0]) if cb[0] < 0 else 0.0
        pos_range  = cb[-1]    if cb[-1] > 0 else 0.0
        sym_ratio  = pos_range / neg_range if neg_range > 0 else float('inf')

        if n_active < k_size:
            mono_ok = "⚠️"
        elif prev_resolution is None or resolution <= prev_resolution:
            mono_ok = "✅"
        else:
            mono_ok = "❌"
            all_good = False

        prev_resolution = resolution if n_active >= k_size else prev_resolution
        logging.info(
            f"  L{l+1} {mono_ok}  k={k_size}  range={cb_range:>8.4f}  "
            f"res={resolution:>9.5f}/code  "
            f"neg={neg_range:>7.4f}  pos={pos_range:>7.4f}  "
            f"sym={sym_ratio:>5.2f}x  n={n_active:,}"
        )
    logging.info("  ✅ 粒度严格逐层细化" if all_good else "  ❌ 存在层级粒度倒退")

    # ── 5b. 码字利用率 ───────────────────────────────────────────────────────
    logging.info("=== 各层码字利用率 ===")
    layer_descs = [
        f"L1 信号层  [0,{TAIL_HARD_MAX:.0f}] 三段式锚定",
        "L2 中等修正 64码字 残差量化",
        "L3 中细粒度 64码字 残差量化",
        "L4 细粒度   64码字 兜底高偏差",
    ]
    for l in range(model.n_layers):
        active = int(num_samples[l, 0])
        used   = int(np.sum(num_samples[l, 1:model.codebook_size[l]+1] > 0))
        util   = used / model.codebook_size[l]
        desc   = layer_descs[l] if l < len(layer_descs) else f"Layer{l+1}"
        logging.info(
            f"  {desc}: 参与={active:,}, 码字={used}/{model.codebook_size[l]}, 利用率={util:.2%}"
        )

    # ── 5c. eps 分布 ─────────────────────────────────────────────────────────
    logging.info("=== eps 分布（截断后）===")
    eps_vals = np.array(list(model.eps_table.values()))
    logging.info(
        f"  min={eps_vals.min():.5f}, median={np.median(eps_vals):.5f}, "
        f"max={eps_vals.max():.5f}  (hard_cap={EPS_HARD_CAP})"
    )

    # ── 5d. 重构误差（含区间分布，重点检查 10-50 段）─────────────────────────
    logging.info("=== 重构误差 ===")
    recon = np.sum(quant_codes, axis=1)
    err   = recon - X[:, 0]
    logging.info(
        f"  全局 MAE={np.mean(np.abs(err)):.5f}, "
        f"bias={err.mean():.6f}, max_err={np.max(np.abs(err)):.5f}"
    )

    # 按区间显示均值误差（重点检查 v1 的低估区间）
    df2 = pd.DataFrame({'val': X[:, 0], 'err': err})
    bins   = [0, 0.5, 1.0, 2.0, 5.0, 10.0, 50.0, TAIL_HARD_MAX + 1]
    labels = ['0-0.5', '0.5-1', '1-2', '2-5', '5-10', f'10-50', f'50+']
    df2['bin'] = pd.cut(df2['val'], bins=bins, labels=labels, right=False)
    seg = df2.groupby('bin', observed=True)['err'].agg(['mean', 'count'])
    logging.info(f"  分区间均值误差（负=低估）:\n{seg.to_string()}")

    per_bucket_mae = (
        pd.DataFrame({'pb': price_bucket, 'err': np.abs(err)})
        .groupby('pb')['err'].mean()
    )
    logging.info(f"  MAE 最高 5 个价格段:\n{per_bucket_mae.nlargest(5).to_string()}")

    # ── 5e. 逐层覆盖情况 ─────────────────────────────────────────────────────
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
    pkl_path = os.path.join(SAVE_DIR, "codebooks_4layer_v2.pkl")
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
                'tail_reserve':      TAIL_RESERVE,
                'tail_infer_n':      TAIL_INFER_N,
                'tail_hard_max':     TAIL_HARD_MAX,
                'version':           'v2',
            },
        }, f)
    logging.info(f"模型已保存: {pkl_path}")

    # ── 7. 导出 Excel ─────────────────────────────────────────────────────────
    from openpyxl import Workbook
    from openpyxl.styles import PatternFill, Font, Alignment
    from openpyxl.utils import get_column_letter

    xlsx_path = os.path.join(SAVE_DIR, "codebooks_4layer_v2.xlsx")
    wb = Workbook()
    ws = wb.active
    ws.title = 'codebooks_4layer_v2'

    n_codes = max(CODEBOOK_SIZE)   # = 64
    layers  = [np.asarray(model.codebooks[l]).reshape(-1) for l in range(model.n_layers)]

    ROW_FILLS = [
        PatternFill("solid", fgColor="D9EAD3"),   # L1 绿：信号层
        PatternFill("solid", fgColor="CFE2F3"),   # L2 蓝：中等修正
        PatternFill("solid", fgColor="FFF2CC"),   # L3 黄：中细粒度
        PatternFill("solid", fgColor="FCE5CD"),   # L4 橙：细粒度兜底
    ]

    # 第 1 行：表头（4 层各 64 列，idx + code[0..63]）
    header_row = ['idx'] + list(range(n_codes))
    ws.append(header_row)
    hdr_fill = PatternFill("solid", fgColor="D9D9D9")
    for cell in ws[1]:
        cell.font      = Font(bold=True)
        cell.alignment = Alignment(horizontal='center')
        cell.fill      = hdr_fill

    for l, cb in enumerate(layers):
        row = [f'Layer{l+1}'] + [round(float(v), 6) for v in cb]
        row += [''] * (n_codes - len(cb))
        ws.append(row)

        row_idx = l + 2
        fill    = ROW_FILLS[l] if l < len(ROW_FILLS) else None
        for col_idx, cell in enumerate(ws[row_idx]):
            if fill:
                cell.fill = fill
            if col_idx > 0 and isinstance(cell.value, float):
                cell.number_format = '0.0000'
            cell.alignment = Alignment(horizontal='center')

    # 列宽
    ws.column_dimensions['A'].width = 14
    for i in range(n_codes):
        ws.column_dimensions[get_column_letter(2 + i)].width = 9

    wb.save(xlsx_path)
    logging.info(f"Excel 已导出: {xlsx_path}（单 Sheet，4 行 × {n_codes} 列）")


if __name__ == "__main__":
    main()
