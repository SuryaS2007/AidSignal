# AidSignal

**A data-driven decision-support dashboard to identify the world’s most overlooked humanitarian crises and recommend how aid should be delivered based on governance, corruption risk, and market price instability.**  

Built to help humanitarian coordinators (e.g., **OCHA**) and donor organizations answer three questions:  
1. **Where is aid most needed?**  
2. **Where is it not arriving?**  
3. **When it does arrive, will it reach the people it’s meant for?**  

---

## Table of Contents
- [Key Idea](#key-idea)
- [What the Dashboard Shows](#what-the-dashboard-shows)
- [Data Sources (2026)](#data-sources-2026)
- [Methodology](#methodology)
- [Mismatch Index (0–100)](#mismatch-index-0-100)
- [Diversion Risk Score (0–100)](#diversion-risk-score-0-100)
- [Market Volatility Score (0–100)](#market-volatility-score-0-100)
- [Recommended Aid Delivery Modality](#recommended-aid-delivery-modality)
- [AI Features (Claude API)](#ai-features-claude-api)
- [Getting Started](#getting-started)
- [Large Files / Dataset Download](#large-files--dataset-download)
- [Project Structure](#project-structure)
- [Ethics, Limitations, and Safety Notes](#ethics-limitations-and-safety-notes)
- [Roadmap](#roadmap)
- [Team](#team)
- [License](#license)
- [Acknowledgments](#acknowledgments)

---

## Key Idea
Humanitarian funding and attention do not always align with severity—and even when funding exists, delivery conditions can determine whether people actually receive help. This tool combines three lenses:  

- **Mismatch (overlooked crises):** severe needs + weak funding flows  
- **Diversion risk:** probability aid is captured/obstructed before reaching beneficiaries  
- **Market volatility:** price instability that can trigger essentials insecurity  

---

## What the Dashboard Shows
- Interactive **Plotly choropleth map** colored by **Mismatch Index** (red → green scale)  
- Clicking a country opens a detail panel showing:  
  - **Mismatch score**  
  - **Diversion risk gauge**  
  - **Market volatility score** (badge or mini-gauge)  
- Stats grid:  
  - People in need  
  - Funding required vs received  
  - Funding gap  
  - $/person  
  - CBPF allocation & allocation per person  
  - **Recommended delivery modality** as a colored badge  
- Dropdown listing active **crisis clusters** (Health, WASH, Food Security, Shelter, Protection, etc.) with per-cluster funding breakdowns  
- AI buttons:  
  - **AI Logistics Plan** (cluster-level)  
  - **AI Briefing** (country-level, 3–4 sentence summary)  

---

## Data Sources (2026)
- **UN Humanitarian Needs Overview (HNO):** People-in-need and targeting ratios  
- **Financial Tracking Service (FTS):** Requirements and funding received  
- **Country-Based Pooled Fund (CBPF):** Allocation totals and allocation per person  
- **World Bank Worldwide Governance Indicators (WGI):** Governance indicators for diversion risk  
- **IMF Consumer Price Index (CPI) dataset** *(included in project pipeline)*  

**Note:** Large dataset CSVs are too big for GitHub. See https://drive.google.com/file/d/1Ksme_0hJeCz1j51ic53cQp7R5aKBWLIH/view?usp=drive_web.
There is no API Key for github privacy reasons, please use your own anthropic API key.

---

## Getting Started

### Prerequisites
- Python 3.10+ **or** Node 18+ (depending on implementation)  
- Claude API key (optional, for AI features)  
- Download the 2026 dataset files (see below)

### Install & Run
```bash
# Clone the repo
git clone https://github.com/SuryaS2007/AidSignal.git

# Enter the project folder
cd AidSignal

# Install dependencies (Python example)
pip install -r requirements.txt

# Run the dashboard
python app.py
