"""
Smart CRM & Funnel Optimization Dashboard (Streamlit)
Project: Smart CRM & Funnel Optimization Dashboard 2026
Single-file prototype for: Funnel design simulation, CRM structuring, Nurturing tracks, Funnel analytics & CAC optimization.

Requirements:
- Python 3.8+
- streamlit
- pandas
- numpy
- plotly

Run:
streamlit run smart_crm_streamlit_2026.py

This file is intentionally self-contained. It generates mock data and demonstrates:
- Funnel stage definitions
- CRM contact list simulation
- Nurturing templates per intent
- Channel-level CAC analysis and experiment suggestions
- Interactive charts and tables

Note: This prototype does not integrate with real email/CRM providers. It is designed as a polished demo you can show to interviewers or extend into a full product.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import random

st.set_page_config(page_title="Smart CRM & Funnel Optimization 2026", layout="wide")

# ---------------------------
# Utilities & Mock Data
# ---------------------------

def seed(seed_val=42):
    random.seed(seed_val)
    np.random.seed(seed_val)

seed(2026)

INDUSTRIES = ["SaaS", "E-Commerce", "Healthcare", "FinTech", "EdTech", "Manufacturing"]
SOURCES = ["Facebook Ads", "Email Campaign", "LinkedIn DMs", "Referral", "Organic", "Webinar"]


def generate_contacts(n=1000, start_date=None):
    if start_date is None:
        start_date = datetime.now() - timedelta(days=90)

    rows = []
    for i in range(n):
        created = start_date + timedelta(days=int(np.random.exponential(scale=20)))
        source = np.random.choice(SOURCES, p=[0.35, 0.2, 0.15, 0.1, 0.15, 0.05])
        industry = np.random.choice(INDUSTRIES)
        name = f"Lead_{i}"
        email = f"lead{i}@example.com"
        company = f"Company_{random.randint(1,300)}"

        # engagement score 0-100
        base = 20
        if source == "Email Campaign": base += 15
        if source == "Facebook Ads": base += np.random.randint(-5,10)
        if source == "LinkedIn DMs": base += 20
        if source == "Webinar": base += 30

        engagement = min(100, max(0, int(base + np.random.normal(0, 18))))

        # mock lifecycle - probability of conversion depends on engagement
        converted = np.random.rand() < (engagement / 300)
        conversion_date = created + timedelta(days=random.randint(7,45)) if converted else None

        rows.append({
            'contact_id': i,
            'name': name,
            'email': email,
            'company': company,
            'industry': industry,
            'source': source,
            'created_at': created.date(),
            'engagement_score': engagement,
            'converted': converted,
            'conversion_date': conversion_date.date() if conversion_date else None
        })

    return pd.DataFrame(rows)


@st.cache_data
def load_mock_data(n=2000):
    df = generate_contacts(n=n)
    # Attach mock costs per acquisition by source
    cost_map = {
        'Facebook Ads': 30_000,  # total spent
        'Email Campaign': 10_000,
        'LinkedIn DMs': 25_000,
        'Referral': 2_500,
        'Organic': 0,
        'Webinar': 5_000
    }
    df['channel_cost_total'] = df['source'].map(cost_map)
    return df

# ---------------------------
# Funnel Definitions
# ---------------------------

FUNNEL_STAGES = [
    'Lead',        # any inbound contact
    'MQL',         # engaged multiple times or visited pricing
    'SQL',         # demo booked or positive reply
    'POC',         # trial/agreed POC (bonus stage)
    'Customer'
]


def assign_stage(row):
    s = row['engagement_score']
    if row['converted']:
        return 'Customer'
    if s >= 65:
        # high engagement -> SQL or POC
        return 'SQL' if np.random.rand() < 0.6 else 'POC'
    if s >= 40:
        return 'MQL'
    return 'Lead'


# ---------------------------
# Analytics & Metrics
# ---------------------------

@st.cache_data
def prepare_dataset(n=2000):
    df = load_mock_data(n)
    df['funnel_stage'] = df.apply(assign_stage, axis=1)
    # per-channel aggregated numbers (mock conversions pulled from converted flag)
    agg = df.groupby('source').agg(
        leads=('contact_id', 'count'),
        conversions=('converted', 'sum'),
        avg_engagement=('engagement_score', 'mean')
    ).reset_index()

    # simulate actual cost per channel (proportional to count to keep numbers sensible)
    total_leads = agg['leads'].sum()
    # get base costs from df (first matching)
    base_costs = df[['source','channel_cost_total']].drop_duplicates().set_index('source')['channel_cost_total'].to_dict()
    agg['cost_incurred'] = agg['source'].map(base_costs)
    agg['conversion_rate'] = (agg['conversions'] / agg['leads']).fillna(0)
    agg['cost_per_conversion'] = agg.apply(lambda r: r['cost_incurred'] / r['conversions'] if r['conversions']>0 else np.nan, axis=1)

    return df, agg


# ---------------------------
# Nurturing Templates (no external AI calls here)
# ---------------------------

NURTURING_TEMPLATES = {
    'high': {
        'subject': "Quick follow-up: tailored plan for {company}",
        'body': (
            "Hi {name},\n\nThanks for taking a demo with us. "
            "I wanted to share a short case study showing how we reduced CAC for a similar {industry} company by 40%. "
            "If you're open, I can block a 15-min call this week to discuss a custom pilot.\n\nBest,\nFounder"
        )
    },
    'mid': {
        'subject': "Here’s the recording + case study from our webinar",
        'body': (
            "Hi {name},\n\nThanks for attending our webinar. "
            "Here’s the recording and a short case study that outlines a step-by-step process to optimize CAC for {industry} teams. "
            "If any part interests you, reply and we’ll set a 10-min discovery.\n\nCheers,\nGrowth Team"
        )
    },
    'low': {
        'subject': "Monthly insights: LTV hacks for {industry}",
        'body': (
            "Hello {name},\n\nWe share a short monthly note with actionable LTV improvements. "
            "This month: 3 quick experiments to raise average order value for {industry} businesses. "
            "If you'd like a tailored note, tell us your main goal.\n\nWarmly,\nFounder"
        )
    }
}


# ---------------------------
# Streamlit UI
# ---------------------------

st.title("Smart CRM & Funnel Optimization — 2026 Prototype")
st.caption("Interactive prototype: funnel simulation, CRM list, nurturing templates, CAC analytics and quick experiments.")

# Sidebar: controls
with st.sidebar:
    st.header("Controls")
    n = st.slider("# of mock contacts", min_value=200, max_value=5000, value=1200, step=100)
    show_raw = st.checkbox("Show raw contacts table", value=False)
    date_filter = st.date_input("Created after (filter)", value=(datetime.now() - timedelta(days=90)).date())
    st.markdown("---")
    st.markdown("**Quick actions**")
    if st.button("Download CSV (contacts)"):
        df_all, agg = prepare_dataset(n)
        csv = df_all.to_csv(index=False)
        st.download_button("Click to download CSV", data=csv, file_name="smart_crm_contacts.csv")

# Load data
with st.spinner("Generating dataset..."):
    df_all, agg = prepare_dataset(n)

# Apply date filter
if date_filter:
    df = df_all[df_all['created_at'] >= pd.to_datetime(date_filter).date()].copy()
else:
    df = df_all.copy()

# Top row metrics
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Total Leads", int(df.shape[0]))
with col2:
    st.metric("Customers", int(df['converted'].sum()))
with col3:
    avg_eng = round(df['engagement_score'].mean(), 1)
    st.metric("Avg Engagement", avg_eng)
with col4:
    st.metric("Unique Sources", int(df['source'].nunique()))

# Funnel visualization
st.subheader("Funnel Stage Distribution")
st.write("Funnel stages are assigned by engagement score and conversion flag (prototype logic).")

stage_counts = df['funnel_stage'].value_counts().reindex(FUNNEL_STAGES, fill_value=0)
fig = go.Figure(go.Funnel(
    y=stage_counts.index.tolist(),
    x=stage_counts.values.tolist(),
    textinfo="value+percent initial"
))
fig.update_layout(height=420, margin=dict(l=50,r=50,t=40,b=40))
st.plotly_chart(fig, use_container_width=True)

# Channel analytics
st.subheader("Channel Performance & CAC")
st.write("Overview per source: leads, conversions, conversion rate, cost and cost-per-conversion.")

# merge agg with df filter (recompute counts for filtered df)
agg_filtered = df.groupby('source').agg(
    leads=('contact_id', 'count'),
    conversions=('converted', 'sum'),
    avg_engagement=('engagement_score', 'mean')
).reset_index()

# keep cost mapping from full dataset
cost_map = df_all[['source','channel_cost_total']].drop_duplicates().set_index('source')['channel_cost_total'].to_dict()
agg_filtered['cost_incurred'] = agg_filtered['source'].map(cost_map).fillna(0)
agg_filtered['conversion_rate'] = (agg_filtered['conversions'] / agg_filtered['leads']).fillna(0)
agg_filtered['cost_per_conversion'] = agg_filtered.apply(lambda r: r['cost_incurred'] / r['conversions'] if r['conversions']>0 else np.nan, axis=1)

st.dataframe(agg_filtered.sort_values('leads', ascending=False))

# Identify underperforming channel (simple heuristic)
st.markdown("**Signal:** Identify the highest cost-per-conversion channel and lowest conversion-rate channel.")
underperforming = agg_filtered.sort_values('cost_per_conversion', ascending=False).iloc[0]
st.info(f"Underperforming channel (by cost per conversion): {underperforming['source']} — Cost/conv: {underperforming['cost_per_conversion']:.0f}")

# Show chart of conversion rates
fig2 = px.bar(agg_filtered, x='source', y='conversion_rate', title='Conversion Rate by Channel')
st.plotly_chart(fig2, use_container_width=True)

# Suggest experiments
st.subheader("Experiment Suggestions to Improve Weak Channel")
weak_channel = underperforming['source']
if weak_channel:
    st.markdown(f"**Channel:** {weak_channel}")
    colA, colB = st.columns(2)
    with colA:
        st.markdown("**Experiment 1 — Audience & Creative Refresh**")
        st.write("Target smaller lookalike audiences based on converted customers. Use dynamic creatives and A/B test 3 creative families.")
    with colB:
        st.markdown("**Experiment 2 — Retargeting + Landing Page Optimization**")
        st.write("Reduce broad prospecting spend. Run retargeting to high-engagement visitors; create intent-focused landing pages (one-click CTA).")

# CAC:LTV Dashboard (simple)
st.subheader("CAC : LTV Monitoring (Simulation)")

# Simulated LTV per industry
ltv_map = {ind: np.round(np.random.uniform(12_000, 60_000), 0) for ind in INDUSTRIES}

# compute CAC per source
cac_map = agg_filtered.set_index('source')['cost_per_conversion'].to_dict()

ltv_rows = []
for ind in INDUSTRIES:
    ltv = ltv_map[ind]
    # pick a dominant acquisition channel randomly
    ch = random.choice(list(cac_map.keys())) if len(cac_map)>0 else 'Email Campaign'
    cac = cac_map.get(ch, np.nan)
    ltv_rows.append({'industry': ind, 'ltv': ltv, 'dominant_channel': ch, 'cac': cac, 'ltv_cac_ratio': round(ltv / cac, 2) if pd.notna(cac) and cac>0 else np.nan})

ltv_df = pd.DataFrame(ltv_rows)
st.dataframe(ltv_df)

st.markdown("**Who should see this weekly?**")
st.write("- Growth Manager: Weekly — to spot channel drift and experiments.\n- Sales Reps: Weekly — pipeline and nurture tasks.\n- CEO: Weekly snapshot — CAC vs LTV, runway impact.")

# Nurturing module demo
st.subheader("Nurturing Tracks — Generate Personalized Message")
st.write("Pick a lead and choose an intent track. The app will generate a message using internal templates (example).")

lead_options = df.sample(min(20, df.shape[0]))[['contact_id','name','company','industry','email','funnel_stage']]
lead_map = {f"{r.contact_id} | {r.name} | {r.company} | {r.funnel_stage}": r.contact_id for r in lead_options.itertuples()}
selected = st.selectbox("Choose a lead", options=list(lead_map.keys()))
intent = st.radio("Intent level", options=['high','mid','low'], index=1, horizontal=True)

if selected:
    contact_id = lead_map[selected]
    lead_row = df[df['contact_id']==contact_id].iloc[0]
    template = NURTURING_TEMPLATES[intent]
    subject = template['subject'].format(name=lead_row['name'], company=lead_row['company'], industry=lead_row['industry'])
    body = template['body'].format(name=lead_row['name'], company=lead_row['company'], industry=lead_row['industry'])

    st.markdown("**Preview**")
    st.text_input("Email Subject", value=subject, key='subject')
    st.text_area("Email Body", value=body, height=220, key='body')

    colS1, colS2 = st.columns(2)
    with colS1:
        if st.button("Mark as Sent (simulated)"):
            st.success("Message marked as sent. (Simulation)")
    with colS2:
        if st.button("Add Follow-up Task"):
            st.info("Follow-up task created for Sales Rep: 3 days from now.")

# Raw table
if show_raw:
    st.subheader("Raw Contacts (sample)")
    st.dataframe(df.sample(min(500, df.shape[0])))

# Documentation and next steps
st.sidebar.markdown("---")
st.sidebar.header("Project Artifacts")
st.sidebar.write("This prototype contains:")
st.sidebar.write("- Funnel simulation logic (engagement-based)")
st.sidebar.write("- CRM-like contact table & filters")
st.sidebar.write("- Nurturing template generator")
st.sidebar.write("- Channel analytics and experiment suggestions")

st.sidebar.markdown("**Next steps to productionize**")
st.sidebar.write("1. Replace mock data with real CRM API (HubSpot/Zoho).\n2. Add event tracking (page visits, email opens) and webhooks.\n3. Integrate real email/WhatsApp/LinkedIn senders.\n4. Add GPT-based personalization with safe prompt templates.\n5. Add authentication & role-based dashboards.")

st.markdown("---")
st.subheader("Strategic Summary (auto-generated)")
st.write(
    "Design funnels for decisions, not just stages. Use engagement signals to route leads, automate low-value follow-ups and reserve human reps for high-intent conversations. Track CAC by channel weekly and run small, measurable experiments to reallocate spend. Combine crisp data storytelling with simple automation to scale revenue while preserving the founder's visibility into growth levers."
)

st.caption("Prototype by: Smart CRM & Funnel Optimization — 2026 (demo). Customize for your company and integrate with real CRM/automation tools for production.")
