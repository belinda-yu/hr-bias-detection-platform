# 通用型組織績效評核偏誤自動化偵測框架

本專案提供一套 **可跨組織、跨資料結構** 的績效評核偏誤偵測工具。在排除敏感特徵的條件下，以機器學習建立員工「應得評分」的客觀基準，並從 **個體層級(殘差分析)** 與 **群體層級(公平性指標)** 雙軌偵測潛在偏誤，輔以統計檢定與 SHAP 可解釋性分析。使用者無需具備程式能力，即可透過 Streamlit 網頁平台上傳資料、設定欄位、一鍵產出偏誤偵測報告與校正建議。

本研究實證以 **主資料(N = 1,224)** 為主，並另以 **兩份結構迥異的公開資料集(INX Future Inc, N = 1,200；Rich Huebner's HR, N = 311)** 進行跨資料集通用性驗證,佐證框架之外部效度與重現性。

---

## 線上平台

公開可存取的 Streamlit 平台:
https://hr-bias-detection-platform-belinda-yu.streamlit.app/

無須安裝，瀏覽器直接使用。

---

## 功能特色

- **欄位抽象化**：透過 `HRSchemaMapper` 將任意 HR 資料欄位對應至統一分析結構，降低導入門檻。
- **自動特徵工程**：`AutoFeatureEngineer` 自動判斷型態、編碼與標準化。
- **無偏基準模型**：隨機森林 + 5-fold GroupKFold 交叉驗證預測，避免資料洩漏導致殘差虛低。
- **雙軌偏誤偵測**：個體層級殘差(±1.5σ 動態門檻) + 群體層級公平性(Disparate Impact、SPD、80% 法則)。
- **統計檢定**：T-test / ANOVA 搭配效果量(Cohen's d、η²)。
- **可解釋性**：SHAP(TreeExplainer)呈現全域與個體層級的特徵貢獻。
- **跨資料集驗證**：三角驗證構念測量(α + SEM + VIF) + Baseline 對照 + 跨資料集通用性壓力測試。

---

## 專案結構

```
project-root/
├── app.py                          # Streamlit 平台入口
├── bias_engine.py                  # 核心模組(與 app.py 同層)
├── requirements.txt                # Python 套件清單
├── packages.txt                    # 系統套件(中文字型 fonts-noto-cjk)
├── README.md
└── notebooks/
    ├── cleaned_hr_data.ipynb       # 資料清洗與特徵工程，產出 cleaned_hr_data.csv
    ├── validation.ipynb            # 合成資料集驗證
    ├── validation_02.ipynb         # 跨資料集驗證
    └── data/
        ├── raw/                        # 原始資料
        ├── cleaned_hr_data.csv         # 經清理的主資料(N = 1,224, 47 欄位)
        └── validation/
            ├── 1.inx_future.csv        # INX Future Inc(IABAC, N = 1,200)
            ├── 2.rich_hr.csv           # Rich Huebner's HR(MSHRM, N = 311)
            └── README.rtf              # 驗證資料來源與授權說明
```

---

## 環境需求與安裝

建議使用 Python 3.10 以上版本。

```bash
# 1. 建立並啟用虛擬環境
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

# 2. 安裝套件
pip install -r requirements.txt
```

---

## 使用方式

### 步驟一：準備資料(產出 cleaned_hr_data.csv)

開啟 `notebooks/cleaned_hr_data.ipynb` 並依序執行所有 cell，完成資料清洗、情境式 IQR 截斷、時間加權與趨勢特徵、以及 SEM 潛在變數萃取，最終輸出 `notebooks/data/cleaned_hr_data.csv`。

### 步驟二:啟動偏誤偵測平台

```bash
streamlit run app.py
```

於瀏覽器開啟後：
1. 從左側上傳已清理的 HR 資料(CSV)。
2. 設定**目標變數**(數值型評分，如主管評分)、**敏感特徵**(如性別、種族)、**客觀特徵**(如年資、滿意度)。
3. 選擇分析方法：殘差法(個人層級)或群體公平性法(群體層級)。
4. 點擊「開始執行偏誤分析」，檢視報告與圖表，並下載完整 CSV 結果。

### 步驟三(選用)：重現跨資料集驗證

開啟 `notebooks/validation_02.ipynb` 並依序執行所有 cell，可重現論文第四章第十節之三項驗證：
- **動作一**：框架通用性(主資料、INX、Rich's HR 三份資料集分別執行完整偵測管線)
- **動作二**：Baseline 對照(主資料與 INX 上比較 Naive / Linear / Ridge / GBM / RF 五模型)
- **動作三**：形成性 vs 反映性構念分析(主資料與 INX 上的 α、VIF、外部效度三角驗證)

---

## 線上部署(Streamlit Community Cloud)

1. 將整個專案推送至 GitHub。
2. 於 [share.streamlit.io](https://share.streamlit.io) 連結該 repo，主程式指定為 `app.py`。
3. 平台會自動依 `requirements.txt` 安裝 Python 套件、依 `packages.txt` 安裝中文字型(`fonts-noto-cjk`)，確保圖表中文正常顯示。

---

## 資料來源

| 資料集 | 來源 | 樣本量 | 授權 | 用途 |
|---|---|---|---|---|
| **主資料集** | Abdallah (2024), Kaggle | 1,224(經清理) | 公開 | 本研究主資料，模型訓練與雙軌偵測 |
| **INX Future Inc** | IABAC, Kaggle | 1,200 | 公開 | 跨資料集驗證(全部三項動作) |
| **Rich Huebner's HR** | Rich Huebner, MSHRM 教學資料 | 311 | 學術公開 | 框架通用性驗證(動作一) |

詳細資料來源、原始連結與授權說明請見 `notebooks/data/cross_validation/README.txt`。

---

## 方法摘要

| 階段 | 內容 |
|---|---|
| 資料前處理 | 缺失值填補(數值→中位數、類別→unknown)、情境式 IQR 薪資截斷 |
| 特徵工程 | 時間加權指標(X_)、趨勢斜率(T_)、SEM 潛在變數(Experience_SEM) |
| 構念效度 | 三角驗證：Cronbach's α(反映性) + SEM 配適指標(共變異) + VIF(形成性) |
| 客觀基準建模 | 隨機森林迴歸 + 5-fold GroupKFold cross_val_predict(排除敏感特徵) |
| 個體偏誤 | 殘差 = 實際評分 − 客觀預測值，±1.5σ 動態門檻 |
| 群體偏誤 | Disparate Impact(80% 法則)、Statistical Parity Difference |
| 統計檢定 | T-test / ANOVA + Cohen's d / η² |
| 可解釋性 | SHAP(TreeExplainer) |
| 外部效度 | 三份結構迥異公開資料集驗證(主資料 + INX + Rich's HR) |

---

## 限制與免責

- 本系統定位為 **決策輔助與偏誤診斷工具**，非自動化決策系統，不應作為晉升、降職或薪資調整的唯一依據。
- 客觀基準仍依賴既有資料分布；若原始資料已含歷史偏誤，模型預測可能延續該偏誤。
- 公平性具情境依賴性，不同指標之間可能存在衡量差異，偏誤判斷不具唯一標準。
- 本研究驗證資料集均為公開教學或合成性質，非真實企業績效紀錄，外部效度推廣至真實企業情境時需個案驗證。
- 偏誤來源歸因(主要與員工自評行為共變)是建立於橫斷面資料的相關性分析，嚴格因果推論需要仰賴縱貫資料或實驗設計。

---

## 引用

如使用本框架，請引用:

> 余姿瑩 (2026)。*基於結構方程與機器學習方法之通用型組織績效評核偏誤偵測*。東吳大學資料科學系碩士在職專班碩士論文。
