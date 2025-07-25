# ---
# jupyter:
#   jupytext:
#     formats: py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import io

st.set_page_config(page_title="購買確率シミュレーション", layout="centered")

st.title("📊 購買確率シミュレーション")

# 線形補間関数
def linear_interp(df, low_col, high_col, low_val, high_val, new_val, new_col):
    if high_val == low_val:
        df[new_col] = df[low_col]  # または df[high_col]
    else:
        df[new_col] = df[low_col] + (df[high_col] - df[low_col]) * ((new_val - low_val) / (high_val - low_val))
    return df


# CSV読み込み（Streamlit用）
def safe_read_csv(uploaded_file, **kwargs):
    try:
        content = uploaded_file.getvalue()
        if len(content) == 0:
            st.warning(f"{uploaded_file.name} は空のファイルです")
            return None
        return pd.read_csv(io.BytesIO(content), **kwargs)
    except Exception as e:
        st.error(f"{uploaded_file.name} の読み込み中にエラーが発生しました: {e}")
        return None        

# ファイルアップローダー
level_file = st.file_uploader("level.csvをアップロードしてください", type="csv")
rule_file = st.file_uploader("interpolation_rules.csvをアップロードしてください", type="csv")
default_file = st.file_uploader("default.csvをアップロードしてください", type="csv")
uploaded_file = st.file_uploader("効用値_ID別.csv をアップロードしてください", type="csv")

df_levels = None

if uploaded_file is not None and default_file is not None:
    df = safe_read_csv(uploaded_file)
    df_default = safe_read_csv(default_file)
    
    if df is None or df_default is None:
        st.stop()
        
    st.success("✅ 効用値とデフォルト構成の読み込み成功！")

    if rule_file is not None:
        rules = safe_read_csv(rule_file)
        if rules is not None:
            for _, row in rules.iterrows():
                df = linear_interp(
                    df,
                    low_col=row["from_label"],
                    high_col=row["to_label"],
                    low_val=float(row["from_val"]),
                    high_val=float(row["to_val"]),
                    new_val=float(row["interp_val"]),
                    new_col=row["new_label"]
                )
            st.success("✅ 補間処理完了！")

    if level_file is not None:
        df_levels = safe_read_csv(level_file, header=None)
        if df_levels is not None:
            st.success("✅ level.csv 読み込み成功！")
            desired_order = df_levels[0].tolist()
        else:
            desired_order = []
    else:
        st.warning("level.csvをアップロードしてください")
        desired_order = []

    existing_cols = [c for c in desired_order if c in df.columns]
    remaining_cols = [c for c in df.columns if c not in existing_cols]
    df = df[existing_cols + remaining_cols]

    st.subheader("✅ 補間済み効用値データ")
    st.dataframe(df)

    st.subheader("✅ デフォルト製品構成")
    st.markdown("---")
    st.dataframe(df_default)

if df_levels is not None:
    df_levels.columns = ['level']

    brand_keywords = ["iPhone", "Android", "Galaxy"]

    def detect_category(val):
        val = str(val).lower()
        if 'mah' in val:
            return 'battery'
        elif 'インチ' in val:
            return 'screen'
        elif '万' in val:
            return 'price'
        elif any(keyword.lower() in val for keyword in brand_keywords):
            return 'brand'
        else:
            return 'unknown'

    df_levels['category'] = df_levels['level'].astype(str).apply(detect_category)
    grouped_levels = df_levels.groupby('category')['level'].apply(list).to_dict()

    brand_options = grouped_levels.get('brand', [])
    battery_options = grouped_levels.get('battery', [])
    screen_options = grouped_levels.get('screen', [])
    price_options = grouped_levels.get('price', [])

    target_brand = st.selectbox("変更するブランドを選んでください", brand_options)
    battery_choice = st.selectbox("バッテリー容量", battery_options)
    screen_choice = st.selectbox("画面サイズ", screen_options)
    price_choice = st.selectbox("金額", price_options)

    default_dict = {
        row["OS"]: {
            "battery": row["バッテリー"],
            "screen": row["画面サイズ"],
            "price": row["価格"]
        } for _, row in df_default.iterrows()
    }

    def get_config(brand_name):
        cols = [brand_name,
                default_dict[brand_name]["battery"],
                default_dict[brand_name]["screen"],
                default_dict[brand_name]["price"]]
        missing_cols = [col for col in cols if col not in df.columns]
        if missing_cols:
            st.warning(f"{brand_name} の構成に以下の列が見つかりません: {missing_cols}")
            return None
        else:
            return [df[col] for col in cols]

    iPhone_conf = get_config("iPhone")
    android_conf = get_config("Android（一般）")
    galaxy_conf = get_config("Galaxy M51（高バッテリー）")

    if target_brand == "iPhone":
        iPhone_conf = [df["iPhone"], df[battery_choice], df[screen_choice], df[price_choice]]
    elif target_brand == "Android（一般）":
        android_conf = [df["Android（一般）"], df[battery_choice], df[screen_choice], df[price_choice]]
    elif target_brand == "Galaxy M51（高バッテリー）":
        galaxy_conf = [df["Galaxy M51（高バッテリー）"], df[battery_choice], df[screen_choice], df[price_choice]]

    df["sum_iPhone"] = sum(iPhone_conf)
    df["sum_Android"] = sum(android_conf)
    df["sum_Galaxy"] = sum(galaxy_conf)

    exp_i, exp_a, exp_g = np.exp(df["sum_iPhone"]), np.exp(df["sum_Android"]), np.exp(df["sum_Galaxy"])
    denom = exp_i + exp_a + exp_g

    df["p_iPhone"] = exp_i / denom
    df["p_Android"] = exp_a / denom
    df["p_Galaxy"] = exp_g / denom

    # 指定順に並べ替えた平均
    mean_probs_selected = df[["p_iPhone", "p_Android", "p_Galaxy"]].mean()
    mean_probs_selected.index = ["iPhone", "Android（一般）", "Galaxy M51（高バッテリー）"]
    mean_probs_selected = mean_probs_selected.loc[["iPhone", "Android（一般）", "Galaxy M51（高バッテリー）"]]

    
    brands, sum_eff = [], []
    for _, row in df_default.iterrows():
        total_eff = sum([df[row[col]].mean() for col in ["OS", "バッテリー", "画面サイズ", "価格"]])
        brands.append(row["OS"])
        sum_eff.append(total_eff)

    df_sum = pd.DataFrame({"OS": brands, "default_sum": sum_eff})
    exp_vals = np.exp(df_sum["default_sum"])
    df_sum["p_default"] = exp_vals / exp_vals.sum()

    df_compare = pd.DataFrame({
        "OS": df_sum["OS"],
        "デフォルト構成": df_sum["p_default"],
        "選択構成": mean_probs_selected.values
    })

    fig = px.bar(
        df_compare,
        x="OS", y=["デフォルト構成", "選択構成"],
        barmode="group", height=500,
        color_discrete_sequence=["blue", "red"]
    )
    fig.update_traces(texttemplate="%{y:.0%}", textposition="outside")
    fig.update_layout(
        yaxis_tickformat=".0%", yaxis_range=[0, 1.1],
        font=dict(size=18),
        title="購買確率の比較（デフォルト構成 vs 選択構成）"
    )
    st.plotly_chart(fig, use_container_width=True)

else:
    st.warning("ファイルをアップロードしてください")

# %%
