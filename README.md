# 通用型組織績效評核偏誤自動化偵測框架

> 以機器學習為基礎之通用型組織績效評核偏誤自動化偵測框架
> 東吳大學巨量資料管理學院 資料科學系碩士在職專班碩士論文
> 研究生：余姿瑩　指導教授：葉向原博士

本專案提供一套**可跨組織、跨資料結構**的績效評核偏誤偵測工具。在排除敏感特徵的條件下，以機器學習建立員工「應得評分」的客觀基準，並從**個體層級（殘差分析）**與**群體層級（公平性指標）**雙軌偵測潛在偏誤，輔以統計檢定與 SHAP 可解釋性分析。使用者無需具備程式能力，即可透過 Streamlit 網頁平台上傳資料、設定欄位、一鍵產出偏誤偵測報告與校正建議。

---

## 功能特色

- **欄位抽象化**：透過 `HRSchemaMapper` 將任意 HR 資料欄位對應至統一分析結構，降低導入門檻。
- **自動特徵工程**：`AutoFeatureEngineer` 自動判斷型態、編碼與標準化。
- **無偏基準模型**：隨機森林 + 5-fold 交叉驗證預測（`cross_val_predict`），避免資料洩漏導致殘差虛低。
- **雙軌偏誤偵測**：個體層級殘差（±1.5σ 動態門檻）＋ 群體層級公平性（Disparate Impact、SPD、80% 法則）。
- **統計檢定**：T-test / ANOVA 搭配效果量（Cohen's d、η²）。
- **可解釋性**：SHAP（TreeExplainer）呈現全域與個體層級的特徵貢獻。

---

## 專案結構

```
project-root/
├── app.py                          # Streamlit 平台入口
├── bias_engine.py                  # 核心模組（與 app.py 同層，請勿移至子資料夾）
├── requirements.txt                # Python 套件清單
├── packages.txt                    # 系統套件（中文字型，供 Streamlit Cloud）
├── README.md
├── notebooks/
│   ├── teacher_baseline.ipynb      # 老師原始版本（離職預測，備存對照）
│   ├── cleaned_hr_data.ipynb       # 資料清洗與特徵工程，產出 cleaned_hr_data.csv
│   └── validation.ipynb            # （選用）端到端驗證，佐證框架通用性
│   └── data/
│       └── cleaned_hr_data.csv     # 處理後、可直接上傳平台的範例資料
│       └── raw/                    # 原始資料
└── docs/
    └── 論文.pdf                     # （選用）論文全文
```

> **重要**：`app.py` 與 `bias_engine.py` 必須位於同一層（根目錄）。`app.py` 以 `from bias_engine import ...` 匯入核心模組，若將其移入子資料夾將導致匯入失敗，且 Streamlit Cloud 預設於根目錄尋找 `app.py`、`requirements.txt`、`packages.txt`。

---

## 環境需求與安裝

建議使用 Python 3.10 以上版本。

```bash
# 1. 建立並啟用虛擬環境
python -m venv venv
source venv/bin/activate          # Windows： venv\Scripts\activate

# 2. 安裝套件
pip install -r requirements.txt
```

---

## 使用方式

### 步驟一：準備資料（產出 cleaned_hr_data.csv）

開啟 `notebooks/cleaned_hr_data.ipynb` 並依序執行所有 cell，完成資料清洗、情境式 IQR 截斷、時間加權與趨勢特徵、以及 SEM 潛在變數萃取，最終輸出 `notebooks/data/cleaned_hr_data.csv`。

> notebook 中的資料路徑可由環境變數覆寫：`export HR_DATA_DIR=./HRDataset_v2/`

### 步驟二：啟動偏誤偵測平台

```bash
streamlit run app.py
```

於瀏覽器開啟後：

1. 從左側上傳已清理的 HR 資料（CSV）。
2. 設定**目標變數**（數值型評分，如主管評分）、**敏感特徵**（如性別、種族）、**客觀特徵**（如年資、滿意度）。
3. 選擇分析方法：殘差法（個人層級）或群體公平性法（群體層級）。
4. 點擊「開始執行偏誤分析」，檢視報告與圖表，並下載完整 CSV 結果。

> **資料隱私**：上傳前請先去識別化，移除姓名、身分證字號等直接識別資訊。

---

## 線上部署（Streamlit Community Cloud）

1. 將整個專案推送至 GitHub。
2. 於 [share.streamlit.io](https://share.streamlit.io) 連結該 repo，主程式指定為 `app.py`。
3. 平台會自動依 `requirements.txt` 安裝 Python 套件、依 `packages.txt`（`fonts-noto-cjk`）安裝中文字型，確保圖表中文正常顯示。

---

## 資料來源

本研究使用 Kaggle 開源之虛擬 HR 資料集（共 1,470 名員工，績效評核紀錄涵蓋 2012–2022 年），包含 `Employee.csv`、`PerformanceRating.csv`、`EducationLevel.csv` 等檔案。

> 請於此處填入實際的 Kaggle 資料集連結與授權說明：`https://www.kaggle.com/datasets/mahmoudemadabdallah/hr-analytics-employee-attrition-and-performance`
> 原始資料集因檔案大小與授權考量，未隨 repo 提供；請自行下載後置於 `HRDataset_v2/`。

---

## 方法摘要

| 階段 | 內容 |
|---|---|
| 資料前處理 | 缺失值填補（數值→中位數、類別→unknown）、情境式 IQR 薪資截斷 |
| 特徵工程 | 時間加權指標（X_）、趨勢斜率（T_）、SEM 潛在變數（Experience_SEM） |
| 客觀基準建模 | 隨機森林迴歸 + 5-fold cross_val_predict（排除敏感特徵） |
| 個體偏誤 | 殘差 = 實際評分 − 客觀預測值，±1.5σ 動態門檻 |
| 群體偏誤 | Disparate Impact（80% 法則）、Statistical Parity Difference |
| 統計檢定 | T-test / ANOVA + Cohen's d / η² |
| 可解釋性 | SHAP（TreeExplainer） |

詳細方法請參閱論文第三章。

---

## 限制與免責

- 本系統定位為**決策輔助與偏誤診斷工具**，非自動化決策系統，不應作為晉升、降職或薪資調整的唯一依據。
- 客觀基準仍依賴既有資料分布；若原始資料已含歷史偏誤，模型預測可能延續該偏誤。
- 公平性具情境依賴性，不同指標之間可能存在衡量差異，偏誤判斷不具唯一標準。
- 本研究以單一公開資料集進行概念驗證（POC），外部效度仍有限。

---

## 作者與論文資訊

- 研究生：余姿瑩
- 指導教授：葉向原博士
- 系所：東吳大學巨量資料管理學院 資料科學系碩士在職專班
