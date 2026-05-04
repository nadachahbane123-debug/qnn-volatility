# ============================================================
# FICHIER : generate_report.py — design premium dark
# RÔLE    : Génère un rapport PDF entièrement sombre par ticker
# ============================================================

import pickle, os, io
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from datetime import datetime

from reportlab.pdfgen import canvas as rl_canvas
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm, mm
from reportlab.lib import colors
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer,
                                 Image, Table, TableStyle, PageBreak,
                                 HRFlowable, KeepTogether)
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_RIGHT

# ── Palette ──
BG      = colors.HexColor("#080B0F")
SURFACE = colors.HexColor("#0E1318")
CARD    = colors.HexColor("#131920")
BORDER  = colors.HexColor("#1E2730")
BORDER2 = colors.HexColor("#2A3540")
GOLD    = colors.HexColor("#C9A84C")
GOLD2   = colors.HexColor("#E8C97A")
TEAL    = colors.HexColor("#4ECDC4")
RED     = colors.HexColor("#E05C5C")
BLUE    = colors.HexColor("#5B9BD5")
TEXT    = colors.HexColor("#F0F4F8")
MUTED   = colors.HexColor("#6B7E8F")
DIM     = colors.HexColor("#3A4A5A")
WHITE   = colors.white

# ── Matplotlib dark style ──
plt.rcParams.update({
    'figure.facecolor'  : '#080B0F',
    'axes.facecolor'    : '#080B0F',
    'axes.edgecolor'    : '#2A3540',
    'axes.labelcolor'   : '#6B7E8F',
    'text.color'        : '#F0F4F8',
    'xtick.color'       : '#6B7E8F',
    'ytick.color'       : '#6B7E8F',
    'grid.color'        : '#1E2730',
    'grid.alpha'        : 0.5,
    'grid.linestyle'    : ':',
    'font.family'       : 'monospace',
    'axes.spines.top'   : False,
    'axes.spines.right' : False,
    'axes.grid'         : True,
    'legend.facecolor'  : '#0E1318',
    'legend.edgecolor'  : '#2A3540',
    'legend.framealpha' : 0.9,
})

W_PAGE = A4[0]
H_PAGE = A4[1]
MARGIN = 1.8 * cm
CONTENT_W = W_PAGE - 2 * MARGIN


# ══════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════

def fig_to_img(fig, width_cm=17, ratio=0.42):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=150,
                bbox_inches='tight',
                facecolor='#080B0F', edgecolor='none')
    buf.seek(0)
    plt.close(fig)
    w = width_cm * cm
    h = w * ratio
    return Image(buf, width=w, height=h)


def S(name, **kw):
    """Shortcut to create a ParagraphStyle."""
    defaults = dict(
        fontName='Helvetica', fontSize=9,
        textColor=TEXT, leading=14,
        spaceAfter=4, spaceBefore=0,
    )
    defaults.update(kw)
    return ParagraphStyle(name, **defaults)


def page_bg(c, doc):
    """Paint full dark background on every page."""
    c.saveState()
    c.setFillColor(BG)
    c.rect(0, 0, W_PAGE, H_PAGE, fill=1, stroke=0)
    # Gold top bar
    c.setFillColor(GOLD)
    c.rect(0, H_PAGE - 3*mm, W_PAGE, 3*mm, fill=1, stroke=0)
    # Page number bottom right
    c.setFont('Helvetica', 7)
    c.setFillColor(MUTED)
    c.drawRightString(W_PAGE - MARGIN,
                      10*mm,
                      f"Page {doc.page}")
    # Footer line
    c.setStrokeColor(BORDER)
    c.setLineWidth(0.4)
    c.line(MARGIN, 16*mm, W_PAGE - MARGIN, 16*mm)
    c.restoreState()


def divider_line():
    return HRFlowable(width="100%", thickness=0.5,
                      color=BORDER2,
                      spaceBefore=6, spaceAfter=8)


def section_title(text, num):
    """Numbered section title with gold accent."""
    return Table([[
        Paragraph(
            f"<font color='#{GOLD.hexval()[2:]}' size='8'>{num:02d}</font>",
            S('sn', fontName='Helvetica-Bold', fontSize=8,
              textColor=GOLD, leading=10)),
        Paragraph(text, S('st', fontName='Helvetica-Bold',
                           fontSize=14, textColor=TEXT,
                           leading=17)),
    ]], colWidths=[1.2*cm, CONTENT_W - 1.2*cm],
    style=[
        ('VALIGN', (0,0), (-1,-1), 'BOTTOM'),
        ('LEFTPADDING', (0,0), (-1,-1), 0),
        ('RIGHTPADDING', (0,0), (-1,-1), 0),
        ('BOTTOMPADDING', (0,0), (-1,-1), 6),
        ('TOPPADDING', (0,0), (-1,-1), 12),
    ])


def insight(label, text):
    """Dark insight card with gold left border."""
    inner = Table([[
        Paragraph(f"— {label.upper()}",
                  S('il', fontName='Courier-Bold', fontSize=7,
                    textColor=GOLD, leading=9, spaceAfter=4)),
    ], [
        Paragraph(text, S('it', fontName='Helvetica', fontSize=8.5,
                           textColor=TEXT, leading=13)),
    ]], colWidths=[CONTENT_W - 0.6*cm])
    inner.setStyle(TableStyle([
        ('BACKGROUND',    (0,0), (-1,-1), CARD),
        ('LEFTPADDING',   (0,0), (-1,-1), 12),
        ('RIGHTPADDING',  (0,0), (-1,-1), 14),
        ('TOPPADDING',    (0,0), (0,0),   8),
        ('BOTTOMPADDING', (0,-1),(-1,-1), 10),
        ('TOPPADDING',    (0,1), (-1,-1), 3),
        ('LINEBEFORE',    (0,0), (0,-1),  3, GOLD),
    ]))
    return inner


def kpi_band(data_rows, col_widths=None):
    """Header row + value row KPI band."""
    if col_widths is None:
        n = len(data_rows[0])
        col_widths = [CONTENT_W / n] * n
    t = Table(data_rows, colWidths=col_widths)
    t.setStyle(TableStyle([
        ('BACKGROUND',    (0,0), (-1,0),  SURFACE),
        ('TEXTCOLOR',     (0,0), (-1,0),  MUTED),
        ('FONTNAME',      (0,0), (-1,0),  'Courier-Bold'),
        ('FONTSIZE',      (0,0), (-1,0),  7),
        ('LETTERSPACE',   (0,0), (-1,0),  1),
        ('BACKGROUND',    (0,1), (-1,-1), CARD),
        ('TEXTCOLOR',     (0,1), (-1,-1), TEXT),
        ('FONTNAME',      (0,1), (-1,-1), 'Courier-Bold'),
        ('FONTSIZE',      (0,1), (-1,-1), 13),
        ('ALIGN',         (0,0), (-1,-1), 'CENTER'),
        ('VALIGN',        (0,0), (-1,-1), 'MIDDLE'),
        ('TOPPADDING',    (0,0), (-1,-1), 9),
        ('BOTTOMPADDING', (0,0), (-1,-1), 9),
        ('LINEABOVE',     (0,0), (-1,0),  2, GOLD),
        ('LINEBELOW',     (0,-1),(-1,-1), 0.3, BORDER),
        ('LINEBEFORE',    (1,0), (-1,-1), 0.3, BORDER),
    ]))
    return t


def dark_table(headers, rows, col_widths=None, accent=TEAL):
    """Styled dark data table."""
    if col_widths is None:
        n = len(headers)
        col_widths = [CONTENT_W / n] * n
    data = [headers] + rows
    t = Table(data, colWidths=col_widths)
    t.setStyle(TableStyle([
        ('BACKGROUND',     (0,0), (-1,0),   SURFACE),
        ('TEXTCOLOR',      (0,0), (-1,0),   GOLD),
        ('FONTNAME',       (0,0), (-1,0),   'Courier-Bold'),
        ('FONTSIZE',       (0,0), (-1,-1),  8),
        ('ROWBACKGROUNDS', (0,1), (-1,-1),  [BG, CARD]),
        ('TEXTCOLOR',      (0,1), (-1,-1),  TEXT),
        ('FONTNAME',       (0,1), (-1,-1),  'Courier'),
        ('ALIGN',          (0,0), (-1,-1),  'CENTER'),
        ('VALIGN',         (0,0), (-1,-1),  'MIDDLE'),
        ('TOPPADDING',     (0,0), (-1,-1),  6),
        ('BOTTOMPADDING',  (0,0), (-1,-1),  6),
        ('GRID',           (0,0), (-1,-1),  0.3, BORDER),
        ('LINEABOVE',      (0,0), (-1,0),   2, accent),
    ]))
    return t


# ══════════════════════════════════════════════════
# GRAPHIQUES
# ══════════════════════════════════════════════════

def plot_serie(df, ticker):
    fig, axes = plt.subplots(3, 1, figsize=(12, 6.5),
                              gridspec_kw={'hspace': 0.55})
    fig.patch.set_facecolor('#080B0F')

    axes[0].plot(df["date"], df["close"],
                 color='#4ECDC4', linewidth=0.9)
    axes[0].fill_between(df["date"], df["close"],
                         alpha=0.08, color='#4ECDC4')
    axes[0].set_title(f"PRIX DE CLOTURE  ·  {ticker}",
                      fontsize=8, fontweight='bold',
                      color='#6B7E8F', loc='left', pad=6)
    axes[0].yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))

    pos = df["rendement"] > 0
    axes[1].bar(df["date"][pos],  df["rendement"][pos],
                color='#4ECDC4', width=1, alpha=0.75)
    axes[1].bar(df["date"][~pos], df["rendement"][~pos],
                color='#E05C5C', width=1, alpha=0.75)
    axes[1].axhline(0, color='#2A3540', linewidth=0.8)
    axes[1].set_title("RENDEMENTS LOGARITHMIQUES",
                      fontsize=8, fontweight='bold',
                      color='#6B7E8F', loc='left', pad=6)

    axes[2].plot(df["date"], df["volatilite"],
                 color='#C9A84C', linewidth=0.6, alpha=0.9)
    axes[2].fill_between(df["date"], df["volatilite"],
                         alpha=0.12, color='#C9A84C')
    axes[2].set_title("VOLATILITE  ·  |rendement|",
                      fontsize=8, fontweight='bold',
                      color='#6B7E8F', loc='left', pad=6)

    for ax in axes:
        ax.tick_params(length=0, labelsize=7)
        for sp in ax.spines.values():
            sp.set_color('#2A3540')
    fig.subplots_adjust(left=0.06, right=0.98)
    return fig_to_img(fig, width_cm=17, ratio=0.50)


def plot_preds(dates, y_te, preds, nom):
    COLS = {0.1: '#4ECDC4', 0.5: '#C9A84C', 0.9: '#E05C5C'}
    LABS = {0.1: 'Plancher', 0.5: 'Mediane', 0.9: 'Risque extreme'}
    fig, axes = plt.subplots(3, 1, figsize=(12, 7),
                              gridspec_kw={'hspace': 0.55})
    fig.patch.set_facecolor('#080B0F')
    for i, tau in enumerate([0.1, 0.5, 0.9]):
        axes[i].plot(dates, y_te,
                     color='#3A4A5A', linewidth=0.5,
                     alpha=0.5, label="Reel")
        axes[i].plot(dates, preds[tau],
                     color=COLS[tau], linewidth=1.3,
                     label=f"{nom} tau={tau}", zorder=3)
        axes[i].fill_between(dates, preds[tau],
                             alpha=0.1, color=COLS[tau])
        axes[i].set_title(
            f"{nom}  |  TAU = {tau}  |  {LABS[tau].upper()}",
            fontsize=8, fontweight='bold',
            color=COLS[tau], loc='left', pad=6)
        axes[i].legend(fontsize=7, loc='upper right')
        axes[i].tick_params(length=0, labelsize=7)
        for sp in axes[i].spines.values():
            sp.set_color('#2A3540')
    fig.subplots_adjust(left=0.06, right=0.98)
    return fig_to_img(fig, width_cm=17, ratio=0.52)


def plot_comparaison(dates, y_te, preds_qr, preds_qnn):
    fig, ax = plt.subplots(figsize=(12, 3.8))
    fig.patch.set_facecolor('#080B0F')
    ax.plot(dates, y_te,
            color='#3A4A5A', linewidth=0.5,
            alpha=0.5, label="Reel", zorder=1)
    ax.plot(dates, preds_qr[0.9],
            color='#5B9BD5', linewidth=1.3,
            label="QR  tau=0.9", alpha=0.85, zorder=2)
    ax.plot(dates, preds_qnn[0.9],
            color='#E05C5C', linewidth=1.3,
            label="QNN  tau=0.9", alpha=0.85, zorder=3)
    ax.fill_between(dates, preds_qr[0.9], preds_qnn[0.9],
                    alpha=0.08, color='#C9A84C')
    ax.legend(fontsize=8)
    ax.set_title("QR  vs  QNN  |  tau = 0.9  — Risque extreme",
                 fontsize=8, fontweight='bold',
                 color='#6B7E8F', loc='left', pad=6)
    ax.tick_params(length=0, labelsize=7)
    for sp in ax.spines.values():
        sp.set_color('#2A3540')
    fig.subplots_adjust(left=0.05, right=0.98)
    return fig_to_img(fig, width_cm=17, ratio=0.30)


def plot_shap(shap_vals, FEATURES):
    importance = np.abs(shap_vals).mean(axis=0)
    indices    = np.argsort(importance)[::-1]
    fig, ax = plt.subplots(figsize=(10, 4.5))
    fig.patch.set_facecolor('#080B0F')
    bar_colors = ['#C9A84C' if i == 0
                  else '#4ECDC4' if i == 1
                  else '#3A4A5ABB'
                  for i in range(len(FEATURES))]
    bars = ax.barh(
        [FEATURES[i] for i in indices[::-1]],
        importance[indices[::-1]],
        color=bar_colors[::-1],
        height=0.55, edgecolor='none'
    )
    for bar, val in zip(bars, importance[indices[::-1]]):
        ax.text(val + max(importance)*0.012,
                bar.get_y() + bar.get_height()/2,
                f"{val:.5f}", va='center',
                fontsize=7.5, color='#6B7E8F')
    ax.set_xlabel("Importance SHAP moyenne (|valeur|)",
                  fontsize=8)
    ax.set_title("SHAP FEATURE IMPORTANCE  |  QNN tau=0.9",
                 fontsize=8, fontweight='bold',
                 color='#6B7E8F', loc='left', pad=6)
    ax.tick_params(length=0, labelsize=8)
    ax.set_xlim(0, max(importance) * 1.32)
    for sp in ax.spines.values():
        sp.set_color('#2A3540')
    fig.subplots_adjust(left=0.14, right=0.96)
    return fig_to_img(fig, width_cm=15, ratio=0.48), indices, importance


# ══════════════════════════════════════════════════
# FONCTION PRINCIPALE
# ══════════════════════════════════════════════════

def generer_rapport_pdf(ticker, output_path=None):
    cache_path = f"data/cache_{ticker}.pkl"
    if not os.path.exists(cache_path):
        raise FileNotFoundError(f"Cache introuvable : {cache_path}")

    with open(cache_path, "rb") as f:
        r = pickle.load(f)

    df        = r["df"]
    X_te      = r["X_te"]
    y_te      = r["y_te"]
    dates     = r["dates"]
    preds_qr  = r["preds_qr"]
    preds_qnn = r["preds_qnn"]
    metriques = r["metriques"]
    shap_vals = r["shap_vals"]

    FEATURES = ["lag1","lag2","lag3","lag4","lag5",
                "vol_moy_5j","vol_moy_10j","vol_moy_20j"]

    arch_row = adf_row = None
    if os.path.exists("data/arch_test_all_tickers.csv"):
        df_a = pd.read_csv("data/arch_test_all_tickers.csv")
        r_a  = df_a[df_a["ticker"] == ticker]
        if len(r_a): arch_row = r_a.iloc[0]
    if os.path.exists("data/adf_test_all_tickers.csv"):
        df_d = pd.read_csv("data/adf_test_all_tickers.csv")
        r_d  = df_d[df_d["ticker"] == ticker]
        if len(r_d): adf_row = r_d.iloc[0]

    col_q = 'tau' if 'tau' in metriques.columns else 'τ'
    col_m = 'Modele' if 'Modele' in metriques.columns else 'Modèle'
    qr_rmse  = float(metriques[(metriques[col_m]=='QR') &(metriques[col_q]==0.9)]['RMSE'].values[0])
    qnn_rmse = float(metriques[(metriques[col_m]=='QNN')&(metriques[col_q]==0.9)]['RMSE'].values[0])
    gagnant  = "QNN" if qnn_rmse < qr_rmse else "QR"
    gain     = abs(qr_rmse - qnn_rmse) / qr_rmse * 100

    if output_path is None:
        os.makedirs("reports", exist_ok=True)
        output_path = (f"reports/rapport_{ticker}_"
                       f"{datetime.now().strftime('%Y%m%d')}.pdf")

    # ── Story ──
    story = []
    sp = lambda h: Spacer(1, h * cm)

    # ────────────────────────────────
    # PAGE 1 — TITRE
    # ────────────────────────────────

    # Badge plateforme
    story.append(sp(0.3))
    story.append(Table([[
        Paragraph("QNN VOLATILITY RESEARCH  ·  EMI 2026",
                  S('badge', fontName='Courier-Bold', fontSize=7,
                    textColor=GOLD, leading=9)),
    ]], colWidths=[CONTENT_W],
    style=[('BACKGROUND',(0,0),(-1,-1),SURFACE),
           ('TOPPADDING',(0,0),(-1,-1),6),
           ('BOTTOMPADDING',(0,0),(-1,-1),6),
           ('LEFTPADDING',(0,0),(-1,-1),10),
           ('LINEABOVE',(0,0),(-1,0),2,GOLD)]))

    story.append(sp(0.4))

    # Titre principal
    story.append(Table([[
        Paragraph(
            f"<font size='32'><b>Prediction</b></font>",
            S('t1', fontName='Helvetica-Bold', fontSize=32,
              textColor=TEXT, leading=36)),
    ],[
        Paragraph(
            f"<font size='32'><b>de Volatilite</b></font>  "
            f"<font size='24' color='#{GOLD.hexval()[2:]}'>"
            f"— {ticker}</font>",
            S('t2', fontName='Helvetica-Bold', fontSize=32,
              textColor=TEXT, leading=38)),
    ],[
        Paragraph(
            "Regression Quantile  vs  Quantile Neural Network  "
            "|  Explainability SHAP  |  Nasdaq 100",
            S('t3', fontName='Helvetica', fontSize=10,
              textColor=MUTED, leading=14, spaceBefore=6)),
    ]], colWidths=[CONTENT_W],
    style=[('BACKGROUND',(0,0),(-1,-1),CARD),
           ('LEFTPADDING',(0,0),(-1,-1),18),
           ('TOPPADDING',(0,0),(0,0),16),
           ('BOTTOMPADDING',(0,-1),(-1,-1),16),
           ('TOPPADDING',(0,1),(-1,-1),0),
           ('LINEBEFORE',(0,0),(0,-1),4,GOLD)]))

    story.append(sp(0.5))

    # KPIs
    split_date = str(pd.to_datetime(dates[0]))[:10]
    skew = float(df["rendement"].skew())
    kurt = float(df["rendement"].kurtosis())

    story.append(kpi_band([
        ["INSTRUMENT", "OBSERVATIONS", "PERIODE", "VOL. MOYENNE", "QNN RMSE t=0.9"],
        [ticker,
         f"{len(df):,}",
         f"{str(df['date'].min())[:7]} → {str(df['date'].max())[:7]}",
         f"{y_te.mean():.5f}",
         f"{qnn_rmse:.5f}"],
    ]))

    story.append(sp(0.4))

    # Info bande
    story.append(Table([[
        Paragraph(
            f"Ecole Mohammadia d'Ingenieurs  ·  "
            f"Series Chronologiques  ·  "
            f"Genere le {datetime.now().strftime('%d/%m/%Y a %H:%M')}  ·  N.C.",
            S('inf', fontName='Courier', fontSize=7,
              textColor=MUTED, alignment=TA_CENTER)),
    ]], colWidths=[CONTENT_W],
    style=[('BACKGROUND',(0,0),(-1,-1),SURFACE),
           ('TOPPADDING',(0,0),(-1,-1),5),
           ('BOTTOMPADDING',(0,0),(-1,-1),5)]))

    story.append(sp(0.6))

    # ── Section 1 — Serie temporelle ──
    story.append(section_title("Analyse de la serie temporelle", 1))
    story.append(divider_line())
    story.append(plot_serie(df, ticker))
    story.append(sp(0.25))

    stats = df["volatilite"].describe()
    story.append(insight(
        "Statistiques descriptives",
        f"La serie <b>{ticker}</b> couvre <b>{len(df):,} jours</b> "
        f"de cotation ({str(df['date'].min())[:10]} — {str(df['date'].max())[:10]}). "
        f"Volatilite moyenne : <b>{stats['mean']:.5f}</b> | "
        f"Ecart-type : <b>{stats['std']:.5f}</b> | "
        f"Maximum : <b>{stats['max']:.5f}</b>. "
        f"Skewness : <b>{skew:.3f}</b> "
        f"({'asymetrie negative — chutes plus fortes que les hausses' if skew < 0 else 'asymetrie positive'}) | "
        f"Kurtosis : <b>{kurt:.1f}</b> "
        f"({'queues tres epaisses — evenements extremes frequents' if kurt > 3 else 'distribution proche de la normale'}). "
        f"Ces caracteristiques motivent l'approche par regression quantile plutot que par la moyenne conditionnelle."
    ))

    story.append(PageBreak())

    # ── Section 2 — Tests statistiques ──
    story.append(section_title("Tests statistiques de validation", 2))
    story.append(divider_line())

    test_headers = ["TEST", "STATISTIQUE", "P-VALUE", "CONCLUSION"]
    test_rows = []
    arch_txt = adf_txt = ""

    if arch_row is not None:
        arch_ok = bool(arch_row["arch_detecte"])
        test_rows.append([
            "ARCH-LM  (Engle 1982)",
            f"{arch_row['lm_stat']:.4f}",
            f"{arch_row['lm_pval']:.2e}",
            "Effets ARCH detectes" if arch_ok else "Pas d'effets ARCH",
        ])
        arch_txt = (
            f"Le test ARCH-LM de Engle (1982) confirme la presence "
            f"d'heteroscedasticite conditionnelle dans les rendements de <b>{ticker}</b> "
            f"(stat LM = <b>{arch_row['lm_stat']:.2f}</b>, "
            f"p-value = <b>{arch_row['lm_pval']:.2e}</b>). "
            "La variance des rendements n'est pas constante dans le temps — "
            "c'est le phenomene de <b>volatility clustering</b>. "
            "Ce resultat justifie statistiquement la modelisation de la volatilite "
            "par des approches adaptatives comme le QNN."
        ) if arch_ok else (
            f"Le test ARCH-LM ne detecte pas d'effets ARCH significatifs "
            f"pour <b>{ticker}</b> (stat = {arch_row['lm_stat']:.2f}, "
            f"p = {arch_row['lm_pval']:.2e})."
        )

    if adf_row is not None:
        adf_ok = bool(adf_row["stationnaire"])
        test_rows.append([
            "ADF  (Dickey-Fuller augmente)",
            f"{adf_row['adf_stat']:.4f}",
            f"{adf_row['adf_pval']:.2e}",
            "Serie stationnaire" if adf_ok else "Racine unitaire",
        ])
        adf_txt = (
            f"Le test ADF confirme la stationnarite des rendements logarithmiques "
            f"de <b>{ticker}</b> (stat = <b>{adf_row['adf_stat']:.2f}</b>, "
            f"p = <b>{adf_row['adf_pval']:.2e}</b>). "
            "L'hypothese nulle de racine unitaire est rejetee — "
            "les rendements log peuvent etre modelises directement "
            "sans differentiation prealable. Ce resultat est coherent avec "
            "la theorie de l'efficience faible des marches financiers."
        ) if adf_ok else (
            f"Le test ADF detecte une racine unitaire dans les rendements "
            f"de <b>{ticker}</b> (stat = {adf_row['adf_stat']:.2f}, "
            f"p = {adf_row['adf_pval']:.2e}). Une differentiation est recommandee."
        )

    if test_rows:
        story.append(dark_table(
            test_headers, test_rows,
            col_widths=[7*cm, 3.2*cm, 3.2*cm, 4.6*cm],
            accent=TEAL
        ))
        story.append(sp(0.3))

    if arch_txt:
        story.append(insight("Test ARCH-LM", arch_txt))
        story.append(sp(0.2))
    if adf_txt:
        story.append(insight("Test ADF — Stationnarite", adf_txt))

    story.append(PageBreak())

    # ── Section 3 — QR ──
    story.append(section_title("Regression Quantile Lineaire  (QR)", 3))
    story.append(divider_line())
    story.append(plot_preds(dates, y_te, preds_qr, "QR"))
    story.append(sp(0.25))
    story.append(insight(
        "Interpretation — QR",
        "La regression quantile lineaire (Koenker & Bassett, 1978) estime "
        "directement les quantiles de la distribution conditionnelle de la volatilite "
        "en minimisant la <b>pinball loss</b> asymetrique. "
        "Elle est rapide, interpretable et presente une bonne calibration theorique "
        "sur les quantiles intermediaires (tau=0.5). "
        "Sa limite principale est l'hypothese de linearite entre les predicteurs "
        "et les quantiles de la volatilite, qui peut etre insuffisante pour capturer "
        "les episodes de forte volatilite lors des crises de marche."
    ))

    story.append(PageBreak())

    # ── Section 4 — QNN ──
    story.append(section_title("Quantile Neural Network  (QNN)", 4))
    story.append(divider_line())
    story.append(plot_preds(dates, y_te, preds_qnn, "QNN"))
    story.append(sp(0.25))
    story.append(insight(
        "Interpretation — QNN",
        "Le Quantile Neural Network (Taylor, 2000) generalise la regression quantile "
        "au cadre non-lineaire en substituant le modele lineaire par un reseau de "
        "neurones (couches <b>Linear → ReLU → Dropout</b>). "
        "La meme pinball loss est utilisee comme critere d'optimisation, "
        "permettant l'apprentissage par descente de gradient. "
        f"Pour le quantile extreme tau=0.9, le QNN obtient un RMSE de "
        f"<b>{qnn_rmse:.5f}</b> versus <b>{qr_rmse:.5f}</b> pour la QR, "
        f"soit une amelioration de <b>{gain:.1f}%</b>. "
        "Le QNN capture des non-linearites invisibles au modele lineaire, "
        "notamment lors des clusters de forte volatilite."
    ))

    story.append(PageBreak())

    # ── Section 5 — Comparaison ──
    story.append(section_title("Comparaison  QR  vs  QNN", 5))
    story.append(divider_line())

    # Tableau metriques
    story.append(Paragraph(
        "Metriques comparatives — tous quantiles",
        S('sh', fontName='Courier-Bold', fontSize=8,
          textColor=MUTED, leading=10, spaceAfter=6)
    ))

    met_headers = ["MODELE", "TAU", "RMSE", "MAE", "MAPE", "PINBALL"]
    met_rows = []
    for _, row in metriques.iterrows():
        met_rows.append([
            str(row[col_m]),
            str(row[col_q]),
            f"{row['RMSE']:.4f}",
            f"{row['MAE']:.4f}",
            f"{row['MAPE']:.1f}%",
            f"{row['Pinball']:.4f}",
        ])
    story.append(dark_table(
        met_headers, met_rows,
        col_widths=[3*cm, 2.2*cm, 2.8*cm, 2.8*cm, 2.8*cm, 2.8*cm],
        accent=GOLD
    ))

    story.append(sp(0.4))
    story.append(plot_comparaison(dates, y_te, preds_qr, preds_qnn))
    story.append(sp(0.25))

    # Verdict
    verdict_txt = (
        f"Pour le quantile extreme <b>tau=0.9</b> — le plus pertinent "
        f"pour la gestion du risque — le <b>{gagnant}</b> remporte la comparaison "
        f"avec un RMSE de <b>{min(qr_rmse, qnn_rmse):.5f}</b> "
        f"contre <b>{max(qr_rmse, qnn_rmse):.5f}</b> pour l'autre modele, "
        f"soit une amelioration de <b>{gain:.1f}%</b>. "
        + ("Le QNN capte des non-linearites et des effets de regime "
           "invisibles a la regression lineaire, notamment lors des "
           "clusters de forte volatilite observes apres 2020."
           if gagnant == "QNN" else
           "La regression quantile presente une meilleure generalisation "
           "sur ce ticker, suggerant que les dynamiques de volatilite "
           "sont principalement lineaires.")
    )
    story.append(insight(f"Verdict — tau = 0.9  |  {gagnant} remporte", verdict_txt))

    story.append(sp(0.4))

    # Calibration
    story.append(Paragraph(
        "Calibration empirique",
        S('sh2', fontName='Courier-Bold', fontSize=8,
          textColor=MUTED, leading=10, spaceAfter=6)
    ))
    cal_headers = ["MODELE", "TAU", "ATTENDU", "OBSERVE", "ECART"]
    cal_rows = []
    for tau in [0.1, 0.5, 0.9]:
        for n, p in [("QR", preds_qr), ("QNN", preds_qnn)]:
            couv = np.mean(y_te <= p[tau]) * 100
            ecart = couv - tau * 100
            cal_rows.append([
                n, f"{tau}",
                f"{tau*100:.1f}%",
                f"{couv:.1f}%",
                f"{ecart:+.1f} pts",
            ])
    story.append(dark_table(
        cal_headers, cal_rows,
        col_widths=[3.4*cm, 2.6*cm, 3.4*cm, 3.4*cm, 3.4*cm],
        accent=TEAL
    ))
    story.append(sp(0.25))
    story.append(insight(
        "Interpretation — Calibration",
        "Un modele parfaitement calibre verrait la volatilite reelle depasser "
        "le quantile tau=0.9 exactement <b>10% du temps</b>. "
        "La QR presente generalement une meilleure calibration theorique "
        "car elle minimise directement la pinball loss sans regularisation. "
        "Le QNN peut presenter de legers ecarts de calibration en echange "
        "d'une meilleure precision sur les valeurs extremes. "
        "Un ecart negatif (sous-couverture) signifie que le modele sous-estime "
        "le risque de queue — a surveiller en gestion des risques."
    ))

    story.append(PageBreak())

    # ── Section 6 — SHAP ──
    story.append(section_title("Explainability  —  SHAP", 6))
    story.append(divider_line())

    img_shap, indices, importance = plot_shap(shap_vals, FEATURES)
    top3 = [FEATURES[i] for i in indices[:3]]
    top5 = [(FEATURES[i], importance[i]) for i in indices[:5]]

    # Graphique + ranking cote a cote
    rank_rows = [[
        Paragraph("RANG", S('rh', fontName='Courier-Bold',
                             fontSize=7, textColor=GOLD, leading=9)),
        Paragraph("VARIABLE", S('rh2', fontName='Courier-Bold',
                                 fontSize=7, textColor=GOLD, leading=9)),
        Paragraph("IMPORTANCE", S('rh3', fontName='Courier-Bold',
                                   fontSize=7, textColor=GOLD,
                                   leading=9, alignment=TA_RIGHT)),
    ]]
    medals = ["01", "02", "03", "04", "05"]
    for i, (feat, imp) in enumerate(top5):
        col = GOLD if i == 0 else TEAL if i == 1 else MUTED
        rank_rows.append([
            Paragraph(medals[i],
                      S(f'r{i}', fontName='Courier-Bold',
                        fontSize=10, textColor=col, leading=12)),
            Paragraph(feat,
                      S(f'f{i}', fontName='Courier-Bold',
                        fontSize=10, textColor=TEXT, leading=12)),
            Paragraph(f"{imp:.5f}",
                      S(f'v{i}', fontName='Courier',
                        fontSize=9, textColor=col,
                        leading=12, alignment=TA_RIGHT)),
        ])

    rank_table = Table(rank_rows,
                       colWidths=[1.5*cm, 4*cm, 3*cm])
    rank_table.setStyle(TableStyle([
        ('BACKGROUND',   (0,0), (-1,0),  SURFACE),
        ('BACKGROUND',   (0,1), (-1,-1), CARD),
        ('ROWBACKGROUNDS',(0,1),(-1,-1), [CARD, BG]),
        ('TOPPADDING',   (0,0), (-1,-1), 8),
        ('BOTTOMPADDING',(0,0), (-1,-1), 8),
        ('LEFTPADDING',  (0,0), (-1,-1), 8),
        ('RIGHTPADDING', (0,0), (-1,-1), 8),
        ('GRID',         (0,0), (-1,-1), 0.3, BORDER),
        ('LINEABOVE',    (0,0), (-1,0),  2, GOLD),
        ('VALIGN',       (0,0), (-1,-1), 'MIDDLE'),
    ]))

    side_by_side = Table([[img_shap, rank_table]],
                         colWidths=[12*cm, 6*cm])
    side_by_side.setStyle(TableStyle([
        ('VALIGN', (0,0), (-1,-1), 'TOP'),
        ('LEFTPADDING', (0,0), (-1,-1), 0),
        ('RIGHTPADDING',(0,0), (-1,-1), 0),
        ('TOPPADDING',  (0,0), (-1,-1), 0),
    ]))
    story.append(side_by_side)
    story.append(sp(0.3))

    shap_interp = (
        f"L'analyse SHAP (SHapley Additive exPlanations) identifie les variables "
        f"les plus importantes pour les predictions du QNN sur <b>{ticker}</b>. "
        f"Les trois variables dominantes sont <b>{top3[0]}</b> "
        f"(importance = {importance[indices[0]]:.5f}), "
        f"<b>{top3[1]}</b> ({importance[indices[1]]:.5f}) et "
        f"<b>{top3[2]}</b> ({importance[indices[2]]:.5f}). "
        + (
            "La predominance des <b>moyennes mobiles (memoire longue)</b> "
            "indique que le QNN capte un regime de volatilite persistant "
            "sur plusieurs semaines — coherent avec le volatility clustering."
            if 'moy' in top3[0] else
            "La predominance des <b>lags recents (memoire courte)</b> "
            "indique que le QNN exploite principalement la dependance "
            "aux mouvements de la semaine precedente."
        ) + " Les valeurs SHAP positives elevees correspondent aux "
            "periodes de forte volatilite persistante."
    )
    story.append(insight("Interpretation SHAP", shap_interp))

    story.append(PageBreak())

    # ── Section 7 — Conclusion ──
    story.append(section_title("Conclusion et synthese", 7))
    story.append(divider_line())

    conclu = (
        f"Cette analyse complete de l'instrument <b>{ticker}</b> confirme "
        f"les trois objectifs du projet. "
        f"<b>Premierement</b>, les tests statistiques valident la demarche : "
        f"le test ARCH-LM confirme la presence d'heteroscedasticite conditionnelle "
        f"et le test ADF confirme la stationnarite des rendements logarithmiques. "
        f"<b>Deuxiemement</b>, la comparaison QR vs QNN montre que le <b>{gagnant}</b> "
        f"obtient de meilleures performances sur le quantile extreme tau=0.9 "
        f"(amelioration RMSE de <b>{gain:.1f}%</b>), confirmant l'apport des "
        f"reseaux de neurones pour la prediction des episodes de forte volatilite. "
        f"<b>Troisiemement</b>, l'analyse SHAP revele que <b>{top3[0]}</b> est "
        f"la variable la plus predictive, suggerant que "
        + ("la persistance de la volatilite sur plusieurs semaines "
           if 'moy' in top3[0] else
           "la memoire a court terme des mouvements recents ")
        + "est le facteur cle pour ce titre."
    )
    story.append(insight("Synthese", conclu))

    story.append(sp(0.5))

    # Recap chiffres cles
    story.append(Paragraph(
        "Chiffres cles du projet",
        S('ck', fontName='Courier-Bold', fontSize=8,
          textColor=MUTED, leading=10, spaceAfter=8)
    ))
    story.append(kpi_band([
        ["QR RMSE t=0.9", "QNN RMSE t=0.9", "AMELIORATION", "VARIABLE N1", "ARCH LM"],
        [f"{qr_rmse:.5f}",
         f"{qnn_rmse:.5f}",
         f"+{gain:.1f}%",
         top3[0],
         f"{arch_row['lm_stat']:.1f}" if arch_row is not None else "N/A"],
    ]))

    story.append(sp(0.5))

    # Footer block
    story.append(Table([[
        Paragraph(
            "QNN Research Dashboard  |  "
            "Ecole Mohammadia d'Ingenieurs  |  "
            "Series Chronologiques 2026  |  N.C.",
            S('ft', fontName='Courier', fontSize=7,
              textColor=MUTED, alignment=TA_CENTER)),
    ]], colWidths=[CONTENT_W],
    style=[('BACKGROUND',(0,0),(-1,-1),SURFACE),
           ('TOPPADDING',(0,0),(-1,-1),7),
           ('BOTTOMPADDING',(0,0),(-1,-1),7),
           ('LINEABOVE',(0,0),(-1,0),0.4,BORDER)]))

    # ── Build ──
    doc = SimpleDocTemplate(
        output_path, pagesize=A4,
        leftMargin=MARGIN, rightMargin=MARGIN,
        topMargin=2.2*cm, bottomMargin=2.2*cm,
        title=f"Rapport QNN — {ticker}",
        author="N.C. — EMI 2026",
    )
    doc.build(story,
              onFirstPage=page_bg,
              onLaterPages=page_bg)

    print(f"Rapport genere : {output_path}")
    return output_path