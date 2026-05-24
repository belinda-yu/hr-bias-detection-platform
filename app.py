import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from bias_engine import HRSchemaMapper, AutoFeatureEngineer, GeneralizedBiasEngine

# 圖表中文 (支援 Mac 與 Windows)
plt.rcParams['font.sans-serif'] = [
    'PingFang TC', 'Heiti TC', 'Arial Unicode MS',   # macOS
    'Microsoft JhengHei', 'SimHei',                  # Windows
    'Noto Sans CJK TC', 'WenQuanYi Zen Hei'          # Linux / Streamlit Cloud
]
plt.rcParams['axes.unicode_minus'] = False  # 確保負號 (-) 正常顯示

st.set_page_config(page_title="HR 績效偏誤偵測平台", layout="wide")
st.title("📊 組織績效評核偏誤自動化偵測框架")
st.markdown("上傳企業 HR 資料，自定義欄位對應，一鍵產出偏誤偵測報告與校正建議。")

with st.expander("⚠️ 系統使用規範與免責聲明（操作前請詳閱）", expanded=False):
    st.info(
        """
        **1. 系統定位為「決策輔助」而非「絕對真理」**
        本系統提供的「偏誤值 (Bias)」與「校正建議」是基於機器學習演算法之歷史資料預測結果，
        請勿作為員工晉升、降職或薪資調整的「唯一」依據。

        **2. 演算法的限制（Garbage In, Garbage Out）**
        「客觀能力值」的預測準確度高度仰賴您選擇的「客觀特徵」。
        建議至少選擇 3 項以上有效的客觀指標進行分析。

        **3. 公平性與多樣性**
        本框架旨在揭露潛在的系統性偏見，協助企業打造更公平的職場環境。
        """
    )

with st.expander("📖 範例資料集（cleaned_hr_data.csv）欄位說明速查表", expanded=False):
    st.markdown(
        """
        以下欄位**僅存在於本研究範例資料集**；若上傳貴公司自有資料，請依實際欄位選擇即可，不需具備這些欄位。

        **時間加權指標（`X_...`）**：越近期的紀錄權重越高，反映員工「當下」的真實狀態。
        - `X_JobSatisfaction`／`X_EnvironmentSatisfaction`／`X_RelationshipSatisfaction`／`X_WorkLifeBalance`：各項滿意度的時間加權平均
        - `X_TrainingOpportunitiesTaken`：實際參與培訓機會的時間加權平均
        - `X_EngagementRate`：上述五項的綜合平均（員工整體投入度）

        **趨勢斜率指標（`T_...`）**：以線性迴歸斜率表示變化方向，正值＝改善中、負值＝惡化中。
        - `T_JobSatisfaction`／`T_WorkLifeBalance`／`T_ManagerRating`

        **潛在變數**
        - `Experience_SEM`：以結構方程模型（SEM）萃取的「員工整體體驗」分數，已處理多重共線性。
        """
    )

st.divider()

# ── 側邊欄 ──────────────────────────────────────────────────────────────────
st.sidebar.header("1. 上傳資料")
st.sidebar.warning(
    "🔒 **資料隱私提醒**\n\n"
    "上傳前請確保資料已**去識別化**（移除員工姓名、身分證字號等直接識別資訊）。"
)
uploaded_file = st.sidebar.file_uploader("請上傳 HR 資料集（CSV 格式）", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("### 📂 原始資料預覽")
    st.dataframe(df.head(), use_container_width=True)
    st.caption(f"共 {len(df)} 筆資料、{len(df.columns)} 個欄位")

    st.sidebar.header("2. 欄位設定")
    st.sidebar.info(
        "💡 **欄位選擇規則**\n"
        "- 🎯 **目標變數**：數值型評分欄位（如主管評分 1–5）\n"
        "- ⚠️ **敏感特徵**：類別型背景欄位（如性別、種族）\n"
        "- 📈 **客觀特徵**：工作表現指標，排除 ID、姓名等欄位"
    )

    # 自動過濾 ID 類欄位
    def is_id_col(col_name, series):
        return 'id' in str(col_name).lower() or series.nunique() == len(series)

    valid_cols     = [c for c in df.columns if not is_id_col(c, df[c])]
    num_cols       = [c for c in df.select_dtypes(include=['int64','float64']).columns if c in valid_cols]
    cat_cols       = [c for c in df.select_dtypes(include=['object','category','bool']).columns if c in valid_cols]

    target_col = st.sidebar.selectbox("🎯 目標變數（必須為數值型）", options=num_cols)

    available_sensitive = [c for c in cat_cols if c != target_col]
    sensitive_cols = st.sidebar.multiselect(
        "⚠️ 敏感特徵（如性別、種族）", options=available_sensitive
    )

    available_objective = [c for c in valid_cols if c != target_col and c not in sensitive_cols]
    objective_cols = st.sidebar.multiselect(
        "📈 客觀特徵（如年資、滿意度）", options=available_objective
    )

    # ── 分析方法選擇（新增）─────────────────────────────────────────────────
    st.sidebar.header("3. 分析方法")
    analysis_method = st.sidebar.radio(
        "選擇偏誤偵測方法",
        options=["residual", "group_fairness"],
        format_func=lambda x: "殘差法（個人層級）" if x == "residual" else "群體公平性法（群體層級）",
        help=(
            "**殘差法**：為每位員工計算「實際評分 − 客觀預測值」，找出個人層級的偏誤。\n\n"
            "**群體公平性法**：比較不同群體（如男女）的高績效比例，找出系統性偏誤。"
        )
    )

    # 群體公平性法的額外參數
    sensitive_target_gf, privileged_group_gf = None, None
    if analysis_method == "group_fairness":
        if sensitive_cols:
            sensitive_target_gf = st.sidebar.selectbox(
                "比較的敏感特徵", options=sensitive_cols
            )
            # 🔴 改成下拉選單，並加入自動判斷
            unique_values = df[sensitive_target_gf].dropna().unique().tolist()
            options_list = ["(自動判斷最高分群體)"] + [str(val) for val in unique_values]
            
            selected_pg = st.sidebar.selectbox(
                "基準群體 (Privileged Group)", 
                options=options_list
            )
            privileged_group_gf = None if selected_pg == "(自動判斷最高分群體)" else selected_pg
        else:
            st.sidebar.warning("請先選擇至少一個敏感特徵。")

    # ── 執行按鈕 ─────────────────────────────────────────────────────────────
    if st.sidebar.button("🚀 開始執行偏誤分析"):
        if not sensitive_cols or not objective_cols:
            st.warning("⚠️ 請至少選擇一個敏感特徵與一個客觀特徵！")
        # 🔴 刪除原本強制要求 text_input 有值的 elif 警告
        else:
            with st.spinner("系統正在執行自動特徵工程與偏誤偵測模型..."):
                try:
                    config = {
                        "target"    : target_col,
                        "sensitive" : sensitive_cols,
                        "objective" : objective_cols
                    }

                    # Step 1：清洗
                    mapper   = HRSchemaMapper(config)
                    clean_df = mapper.validate_and_clean(df)

                    # Step 2：特徵工程
                    engineer = AutoFeatureEngineer()
                    processed_df, final_obj = engineer.fit_transform(
                        clean_df, mapper.objective   # mapper.objective 是副本，不會被汙染
                    )

                    # ── 特徵洩漏防呆：客觀特徵若與目標高度相關，會讓殘差趨近 0 ──
                    leak_warn = []
                    for col in final_obj:
                        if pd.api.types.is_numeric_dtype(processed_df[col]):
                            r = processed_df[[col, target_col]].corr().iloc[0, 1]
                            if pd.notna(r) and abs(r) >= 0.9:
                                leak_warn.append(f"{col}（r={r:.2f}）")
                    if leak_warn:
                        st.warning(
                            "⚠️ 偵測到潛在特徵洩漏：以下客觀特徵與目標變數高度相關（|r|≥0.9），"
                            "可能讓模型 R² 虛高、殘差趨近 0，使偏誤分析失真：" + "、".join(leak_warn)
                        )

                    # Step 3：偵測
                    engine = GeneralizedBiasEngine(mapper.target, final_obj, mapper.sensitive)

                    st.success("✅ 分析完成！")
                    st.divider()

                    # ── 殘差法輸出 ──────────────────────────────────────────
                    if analysis_method == "residual":
                        res_df = engine.run(processed_df, method="residual")

                        c1, c2, c3 = st.columns(3)
                        c1.metric("總分析人數",          f"{len(res_df)} 人")
                        c2.metric("建議向上校正（被打壓）", f"{(res_df['Bias_Flag']=='Suggest Higher').sum()} 人")
                        c3.metric("建議向下校正（被偏袒）", f"{(res_df['Bias_Flag']=='Suggest Lower').sum()} 人")

                        st.write("### 📝 偏誤偵測結果清單")
                        display_cols = [target_col, 'Objective_Rating', 'Bias', 'Bias_Flag'] + sensitive_cols
                        display_cols = [c for c in display_cols if c in res_df.columns]
                        st.dataframe(res_df[display_cols].round(2), use_container_width=True)

                        csv = res_df.to_csv(index=False).encode('utf-8-sig')
                        st.download_button(
                            "📥 下載完整報告（CSV）", data=csv,
                            file_name="HR_Bias_Detection_Report.csv", mime="text/csv"
                        )

                        # 偏誤分布圖
                        st.write("### 📊 偏誤值分布圖")
                        fig, ax = plt.subplots(figsize=(10, 4))
                        ax.hist(res_df['Bias'].dropna(), bins=30, color='steelblue', edgecolor='white', alpha=0.85)
                        ax.axvline(0, color='red', linestyle='--', label='No Bias (0)')
                        ax.set_xlabel('Bias = Actual − Predicted Rating')
                        ax.set_ylabel('Employee Count')
                        ax.legend(); st.pyplot(fig); plt.close(fig)

                        # 各敏感特徵偏誤比較圖
                        if sensitive_cols:
                            st.write("### 📊 各群體平均偏誤比較")
                            for col in sensitive_cols:
                                orig_col_vals = clean_df[col]
                                gb = pd.Series(res_df['Bias'].values, index=orig_col_vals.index) \
                                       .groupby(orig_col_vals).mean().sort_values()
                                fig2, ax2 = plt.subplots(figsize=(8, 3))
                                
                                # 🔴 改用 Seaborn 的漸層色畫圖
                                sns.barplot(x=gb.values, y=gb.index.astype(str), ax=ax2, palette="vlag")
                                ax2.axvline(0, color='black', lw=0.8)
                                ax2.set_title(f'{col} 各群體平均偏誤')
                                ax2.set_xlabel('Mean Bias')
                                st.pyplot(fig2); plt.close(fig2)

                        # 統計檢定
                        st.write("### 🔬 統計偏誤檢定報告（p-value + Effect Size）")
                        st.info(
                            "**如何解讀：**\n"
                            "- p < 0.05 → 該特徵在不同群體之間的評分偏誤具統計顯著性\n"
                            "- Effect Size（η²）< 0.01=小；0.01–0.06=中；>0.14=大"
                        )
                        test_df = engine.run_bias_tests(res_df, clean_df)
                        if not test_df.empty:
                            st.dataframe(test_df, use_container_width=True)

                        # SHAP 圖
                        st.write("### 🧠 AI 決策邏輯解析（SHAP 特徵貢獻度）")
                        st.info(
                            "Y 軸越上方 = 對 AI 預測績效影響越大；"
                            "X 軸右側 = 加分；左側 = 扣分；"
                            "紅色 = 特徵值高；藍色 = 特徵值低。"
                        )
                        try:
                            shap_fig = engine.generate_shap_summary()
                            st.pyplot(shap_fig)      # ← 修正後的 fig 是正確的
                            plt.close(shap_fig)
                        except ImportError:
                            st.warning("⚠️ SHAP 未安裝，無法顯示此圖。請執行：pip install shap")
                        except Exception as e:
                            st.error(f"SHAP 圖表錯誤：{e}")

                    # ── 群體公平性法輸出 ────────────────────────────────────
                    else:
                        gf_result = engine.run(
                            processed_df, method="group_fairness",
                            sensitive_target=sensitive_target_gf,
                            privileged_group=privileged_group_gf
                        )
                        st.write(f"### 📊 群體公平性分析結果（基準群體：{privileged_group_gf}）")

                        di_df  = pd.DataFrame.from_dict(
                            gf_result['Disparate_Impact'], orient='index', columns=['Disparate Impact']
                        ).round(4)
                        spd_df = pd.DataFrame.from_dict(
                            gf_result['Statistical_Parity_Difference'], orient='index',
                            columns=['Statistical Parity Difference']
                        ).round(4)
                        hr_df  = pd.DataFrame.from_dict(
                            gf_result['High_Performer_Rates'], orient='index', columns=['High Performer Rate']
                        ).round(4)

                        result_table = pd.concat([hr_df, di_df, spd_df], axis=1)
                        result_table['Status'] = result_table['Disparate Impact'].apply(
                            lambda x: "⚠️ 潛在弱勢 (違反 80% 法則)" if x < 0.8 else ("✅ 合格" if x <= 1.25 else "⚠️ 過度優勢")
                        )
                        st.dataframe(result_table, use_container_width=True)
                        st.info(
                            "**如何解讀 Disparate Impact（差別影響）：**\n"
                            "- DI ≈ 1.0 → 公平\n"
                            "- DI < 0.8 → 該群體高績效機會明顯偏低（80% 法則）\n"
                            "- DI > 1.2 → 該群體高績效機會明顯偏高"
                        )

                except Exception as e:
                    st.error(f"⚠️ 分析過程發生錯誤，請檢查資料格式：{e}")
                    st.exception(e)

else:
    st.info("👈 請先從左側面板上傳一份 CSV 檔案來啟動平台。")
    st.markdown("""
    **快速開始指南：**
    1. 上傳已清理的 HR 資料（CSV 格式）
    2. 設定目標變數、敏感特徵與客觀特徵
    3. 選擇分析方法，點擊「開始執行」
    4. 下載偏誤偵測報告
    """)