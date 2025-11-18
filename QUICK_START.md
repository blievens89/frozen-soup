# Quick Start Guide

## Installation

```bash
pip install -r requirements.txt
```

## Running the App

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

---

## First Time Setup

1. **Configure API Credentials:**
   - Create a `.streamlit/secrets.toml` file
   - Add your DataForSEO credentials:
     ```toml
     DATAFORSEO_LOGIN = "your_login"
     DATAFORSEO_PASSWORD = "your_password"
     APP_PASSWORD = "your_app_password"
     ```

2. **Set Your Defaults:**
   - Language code (default: "en")
   - Location (default: "United Kingdom")
   - USD to GBP exchange rate (default: 0.79)

---

## 5-Minute Workflow

### For Quick Campaign Ideas:

1. **Enter a seed keyword** (e.g., "remortgage")
2. Click **"Analyse Keywords"**
3. Review **Quick Wins** section
4. Download **Quick Wins CSV**
5. Import to Google Ads

### For Budget Planning:

1. Run keyword analysis
2. Toggle **"Set Custom Budget"** in sidebar
3. Enter your monthly budget
4. Toggle **"Show Budget Optimization"**
5. Review recommended allocation
6. Download **Excel Report**

### For Competitor Research:

1. Select **"Scan Website for Keywords"**
2. Enter competitor URL
3. Click **"Analyse Keywords"**
4. Review **Competitive Insights**
5. Sort by **Opportunity Score**
6. Export top opportunities

---

## Key Features at a Glance

### Sidebar Controls:

| Feature | What It Does |
|---------|-------------|
| **Analysis Mode** | Choose: Seed keyword, Upload list, or Scan website |
| **Advanced Filters** | Filter by volume, CPC, competition, intent |
| **CTR/CVR Assumptions** | Set expected click and conversion rates by intent |
| **Custom Budget** | Plan spending and allocation |
| **ROI Calculator** | Project revenue and profit |
| **Budget Optimization** | Get recommended budget split |

### Main Results:

| Section | What You Get |
|---------|-------------|
| **Keyword Analysis Results** | All keywords with difficulty & opportunity scores |
| **Quick Wins** | Low-hanging fruit keywords |
| **Competitive Insights** | High-competition vs low-competition keywords |
| **Negative Keywords** | Suggested exclusions to save budget |
| **Keyword Groups** | Auto-clustered ad group ideas |
| **Intent Summary** | Metrics by search intent |
| **Blended Overview** | Portfolio-level metrics |
| **Visualizations** | Charts and scatter plots |
| **Export Data** | CSV and Excel downloads |

---

## Common Tasks

### Filter to Your Target CPC:

1. Toggle **"Enable Filtering"** in sidebar
2. Set **Min CPC** and **Max CPC** to your range
3. Results update automatically

### Find Quick Wins:

1. Run analysis
2. Scroll to **"Quick Wins"** section
3. Look for keywords with:
   - Opportunity Score ≥ 60
   - Difficulty Score ≤ 40

### Create Ad Groups:

1. Run analysis
2. Expand **"Keyword Groups"** section
3. Select a group from dropdown
4. Review keywords in that group
5. Use as ad group template

### Calculate ROI:

1. Toggle **"Enable ROI Calculator"** in sidebar
2. Enter **Average Order Value**
3. Set **Profit Margin %**
4. Results show revenue, profit, ROAS, ROI

### Optimize Budget:

1. Toggle **"Set Custom Budget"**
2. Enter total monthly budget
3. Toggle **"Show Budget Optimization"**
4. Compare current vs recommended allocation

---

## Tips & Tricks

### For Best Results:

✅ **DO:**
- Start with broad seed keywords for discovery
- Use filters to focus on actionable keywords
- Review Quick Wins first for immediate opportunities
- Enable ROI Calculator for stakeholder presentations
- Download Excel report for comprehensive sharing
- Check Negative Keywords to prevent waste

❌ **DON'T:**
- Set filters too strict initially (might miss opportunities)
- Ignore high-difficulty keywords (they convert well)
- Skip the Keyword Groups (saves hours in campaign setup)
- Forget to adjust CTR/CVR based on your historical data

### Power User Features:

1. **Multi-Mode Analysis:**
   - Run seed keyword analysis
   - Export keywords
   - Re-import via "Analyse My Keyword List"
   - Add filters for refinement

2. **Competitive Gap Analysis:**
   - Scan your own site
   - Scan competitor site
   - Compare keyword lists manually
   - Find gaps to target

3. **Budget Scenario Planning:**
   - Run analysis once
   - Adjust budget allocation percentages
   - See impact on clicks/conversions
   - Find optimal split

---

## Understanding the Scores

### Difficulty Score (0-100):
- **0-30:** Easy to rank/bid for
- **31-60:** Moderate competition
- **61-100:** Very competitive

**Formula:** 50% competition + 50% CPC-normalized

### Opportunity Score (0-100):
- **0-30:** Low priority
- **31-60:** Worth considering
- **61-100:** High priority

**Factors:**
- Search volume (log scale)
- CPC sweet spot (£0.50-£3.00 ideal)
- Low competition
- Intent type (transactional > commercial > navigational > informational)

---

## Keyboard Shortcuts

When app is focused:
- `Ctrl+R` or `Cmd+R`: Refresh app
- `C`: Clear cache (in Streamlit menu)

---

## Troubleshooting

### "No data returned from API"
- Check API credentials in secrets.toml
- Verify your DataForSEO account has credits
- Try with a different keyword

### "Could not generate intent summary"
- Some keywords may not have intent data
- Check "Debug" expander for unmatched keywords
- Results still show other metrics

### Slow performance with large lists
- Limit to 1000 keywords at a time
- Use filters to reduce dataset
- Clear cache periodically (Streamlit menu → Clear cache)

### Excel export fails
- Ensure openpyxl is installed: `pip install openpyxl`
- Check disk space
- Try CSV export instead

---

## Example Workflows

### Workflow 1: New Campaign Launch
```
1. Enter seed: "life insurance"
2. Set filters: Min volume 100, Max CPC £2.50
3. Run analysis
4. Review Quick Wins (found 23 keywords)
5. Download Quick Wins CSV
6. Check Keyword Groups → 5 groups identified
7. Use groups as ad group structure
8. Add suggested negative keywords
9. Download Excel report for team
```

### Workflow 2: Budget Optimization
```
1. Upload existing keyword list (200 keywords)
2. Enable ROI Calculator (AOV: £500, Margin: 40%)
3. Set custom budget (£5,000/month)
4. Enable budget optimization
5. See recommended allocation:
   - Transactional: 45% (was 25%)
   - Commercial: 30% (was 25%)
   - Informational: 15% (was 25%)
   - Navigational: 10% (was 25%)
6. Review projected ROI: 180%
7. Export Excel report with ROI tab
```

### Workflow 3: Competitive Research
```
1. Select "Scan Website for Keywords"
2. Enter competitor URL
3. Run analysis
4. Sort by Opportunity Score
5. Filter: Only transactional intent
6. Review Competitive Insights
7. Export top 50 opportunities
8. Cross-reference with own keyword list
9. Add gaps to campaign
```

---

## Support

- **Full Documentation:** See `ENHANCEMENTS.md`
- **Version History:** See `CHANGELOG.md`
- **Overview:** See `EXECUTIVE_SUMMARY.md`

---

## Next Steps

1. **Familiarize yourself** with the interface
2. **Run a test analysis** with a familiar keyword
3. **Explore each section** to understand the features
4. **Adjust settings** to match your business metrics
5. **Export and share** results with your team

---

**Pro Tip:** Bookmark the "Quick Wins" and "Keyword Groups" sections - these are the highest-value features for most users.

Happy keyword researching! 🚀
