import streamlit as st
import pandas as pd
import os
from typing import TypedDict, List, Optional, Annotated
from langgraph.graph import StateGraph, START, END
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

st.set_page_config(page_title="InsightAgent", page_icon="", layout="wide", initial_sidebar_state="expanded")

# --- Custom CSS for premium dark theme ---
st.markdown("""
<style>
    .stApp {
        background-color: #0B1020;
    }
    section[data-testid="stSidebar"] {
        background-color: #111827;
    }
    div[data-testid="stMetric"] {
        background-color: #1E293B;
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 16px;
    }
    div[data-testid="stMetricLabel"] {
        color: #94A3B8;
    }
    div[data-testid="stMetricValue"] {
        color: #F8FAFC;
    }
    .stButton button {
        background-color: #1E293B;
        color: #F8FAFC;
        border: 1px solid #334155;
        border-radius: 8px;
    }
    .stButton button:hover {
        border: 1px solid #3B82F6;
        color: #3B82F6;
    }
    .agent-node {
        background-color: #1E293B;
        border: 1px solid #334155;
        border-radius: 10px;
        padding: 12px 20px;
        text-align: center;
        color: #F8FAFC;
        font-weight: 600;
        margin: 4px;
    }
    .agent-node-active {
        border: 1px solid #22C55E;
        color: #22C55E;
    }
    .cause-card {
        background-color: #1E293B;
        border-left: 4px solid #EF4444;
        border-radius: 8px;
        padding: 16px;
        margin-bottom: 12px;
    }
    .cause-card-medium {
        border-left: 4px solid #F59E0B;
    }
    .cause-card-low {
        border-left: 4px solid #3B82F6;
    }
    h1, h2, h3 {
        color: #F8FAFC !important;
    }
    p, span, div {
        color: #F8FAFC;
    }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<style>
/* Target the dropdown options list */
ul[data-testid="stSelectboxVirtualDropdown"] li {
    background-color: #1e1e2f !important;
    color: #ffffff !important;
}
ul[data-testid="stSelectboxVirtualDropdown"] li:hover {
    background-color: #333355 !important;
}
</style>
""", unsafe_allow_html=True)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "processed", "master_df.csv")

@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)

master_df = load_data()

def last_value(existing, new):
    return new if new is not None else existing

class InsightAgentState(TypedDict):
    user_question: str
    investigation_type: List[str]
    review_findings: Annotated[Optional[dict], last_value]
    delivery_findings: Annotated[Optional[dict], last_value]
    seller_findings: Annotated[Optional[dict], last_value]
    category_findings: Annotated[Optional[dict], last_value]
    root_cause_summary: Annotated[Optional[dict], last_value]
    final_report: Annotated[Optional[str], last_value]

def planner_agent(state):
    question = state['user_question'].lower()
    domains = []
    review_keywords = ['review', 'rating', 'satisfaction', 'unhappy', 'happy', 'negative']
    if any(kw in question for kw in review_keywords):
        domains.append('review')
    delivery_keywords = ['deliver', 'delay', 'late', 'slow', 'shipping']
    if any(kw in question for kw in delivery_keywords):
        domains.append('delivery')
    seller_keywords = ['seller', 'vendor', 'supplier']
    if any(kw in question for kw in seller_keywords):
        domains.append('seller')
    category_keywords = ['categor', 'product type']
    if any(kw in question for kw in category_keywords):
        domains.append('category')
    broad_framing_keywords = ['risky', 'risk', 'factors']
    if any(kw in question for kw in broad_framing_keywords):
        for d in ['review', 'delivery', 'seller', 'category']:
            if d not in domains:
                domains.append(d)
    if not domains:
        domains = ['review', 'delivery', 'seller', 'category']
    return {'investigation_type': domains}

def review_agent(state):
    df = master_df
    avg_review = df['review_score'].mean()
    # delay_flag is nullable boolean (NA = undelivered order). Drop those rows
    # first — comparing a nullable-boolean column directly with `== True/False`
    # leaves NA in the mask, and pandas raises a ValueError when you try to
    # index a DataFrame with a boolean mask that contains NA values.
    known = df.dropna(subset=['delay_flag'])
    delayed_avg = known.loc[known['delay_flag'] == True, 'review_score'].mean()
    ontime_avg = known.loc[known['delay_flag'] == False, 'review_score'].mean()
    return {'review_findings': {
        'avg_review_score': round(avg_review, 2),
        'delayed_order_review_score': round(delayed_avg, 2),
        'ontime_order_review_score': round(ontime_avg, 2),
    }}

def delivery_agent(state):
    df = master_df
    avg_delivery_days = df['delivery_days'].mean()
    overall_delay_rate = df['delay_flag'].mean()
    state_delay = (
        df.dropna(subset=['delay_flag'])
          .groupby('customer_state')['delay_flag']
          .mean()
          .sort_values(ascending=False)
          .head(5)
    )
    return {'delivery_findings': {
        'avg_delivery_days': round(avg_delivery_days, 2),
        'overall_delay_rate': round(overall_delay_rate, 4),
        'worst_states_by_delay_rate': state_delay.round(4).to_dict(),
    }}

def seller_agent(state):
    df = master_df
    seller_summary = (
        df.groupby('seller_id')
          .agg(
              seller_review_score=('review_score', 'mean'),
              seller_delay_rate=('delay_flag', 'mean'),
              order_volume=('order_id', 'count')
          )
    )
    reliable = seller_summary[seller_summary['order_volume'] >= 10]
    worst_delay = reliable.sort_values('seller_delay_rate', ascending=False).head(5)
    worst_review = reliable.sort_values('seller_review_score', ascending=True).head(5)
    return {'seller_findings': {
        'total_sellers': df['seller_id'].nunique(),
        'sellers_with_min_volume': reliable.shape[0],
        'worst_sellers_by_delay_rate': worst_delay.round(4).to_dict(orient='index'),
        'worst_sellers_by_review_score': worst_review.round(4).to_dict(orient='index'),
    }}

def category_agent(state):
    df = master_df
    cat_summary = (
        df.groupby('product_category_name_english')
          .agg(
              category_review_score=('review_score', 'mean'),
              category_delay_rate=('delay_flag', 'mean'),
              order_volume=('order_id', 'count')
          )
    )
    reliable = cat_summary[cat_summary['order_volume'] >= 30]
    worst_review = reliable.sort_values('category_review_score', ascending=True).head(5)
    worst_delay = reliable.sort_values('category_delay_rate', ascending=False).head(5)
    return {'category_findings': {
        'total_categories': df['product_category_name_english'].nunique(),
        'categories_with_min_volume': reliable.shape[0],
        'worst_categories_by_review_score': worst_review.round(4).to_dict(orient='index'),
        'worst_categories_by_delay_rate': worst_delay.round(4).to_dict(orient='index'),
    }}

def root_cause_agent(state):
    causes = []
    if state.get('review_findings'):
        rf = state['review_findings']
        avg = float(rf['avg_review_score'])
        delayed_avg = float(rf['delayed_order_review_score'])
        ontime_avg = float(rf['ontime_order_review_score'])
        gap = round(ontime_avg - delayed_avg, 2)
        if gap > 1.0:
            causes.append({'cause': 'Delivery delays driving customer dissatisfaction',
                          'evidence': f'Delayed orders average {delayed_avg} stars vs {ontime_avg} for on-time — a {gap} point gap',
                          'severity': gap})
        if avg < 4.0:
            causes.append({'cause': 'Overall customer satisfaction below benchmark',
                          'evidence': f'Average review score is {avg}/5.0',
                          'severity': round(4.0 - avg, 2)})
    if state.get('delivery_findings'):
        df = state['delivery_findings']
        delay_rate = float(df['overall_delay_rate'])
        avg_days = float(df['avg_delivery_days'])
        worst_states = df['worst_states_by_delay_rate']
        if delay_rate > 0.05:
            causes.append({'cause': 'Significant delivery delay rate across platform',
                          'evidence': f'{round(delay_rate*100,1)}% of orders delayed, averaging {avg_days} days',
                          'severity': round(delay_rate * 10, 4)})
        if worst_states:
            top_state = list(worst_states.keys())[0]
            top_rate = round(float(list(worst_states.values())[0]) * 100, 1)
            if top_rate > 15:
                causes.append({'cause': f'Regional delivery concentration — {top_state} severely underserved',
                              'evidence': f'{top_state} has {top_rate}% delay rate vs {round(delay_rate*100,1)}% national average',
                              'severity': round(top_rate / 100, 4)})
    if state.get('seller_findings'):
        sf = state['seller_findings']
        delay_vals = list(sf['worst_sellers_by_delay_rate'].values())
        review_vals = list(sf['worst_sellers_by_review_score'].values())
        if delay_vals and float(delay_vals[0]['seller_delay_rate']) > 0.3:
            worst_delay = delay_vals[0]
            causes.append({'cause': 'High-risk sellers with extreme delay rates',
                          'evidence': f'Top problem seller has {round(float(worst_delay["seller_delay_rate"])*100,1)}% delay rate and {round(float(worst_delay["seller_review_score"]),2)} avg review score',
                          'severity': float(worst_delay['seller_delay_rate'])})
        if review_vals and float(review_vals[0]['seller_review_score']) < 2.0:
            worst_review = review_vals[0]
            causes.append({'cause': 'Critically underperforming sellers dragging platform rating',
                          'evidence': f'Worst seller averages only {round(float(worst_review["seller_review_score"]),2)}/5.0',
                          'severity': round(2.0 - float(worst_review['seller_review_score']), 4)})
    if state.get('category_findings'):
        cf = state['category_findings']
        review_items = list(cf['worst_categories_by_review_score'].items())
        delay_items = list(cf['worst_categories_by_delay_rate'].items())
        if review_items:
            worst_cat_review = review_items[0]
            if float(worst_cat_review[1]['category_review_score']) < 3.5:
                causes.append({'cause': f'Category quality issues — {worst_cat_review[0]} underperforming',
                              'evidence': f'{worst_cat_review[0]} averages {round(float(worst_cat_review[1]["category_review_score"]),2)}/5.0 with only {round(float(worst_cat_review[1]["category_delay_rate"])*100,1)}% delay rate (not a delivery problem)',
                              'severity': round(3.5 - float(worst_cat_review[1]['category_review_score']), 4)})
        if delay_items:
            worst_cat_delay = delay_items[0]
            if float(worst_cat_delay[1]['category_delay_rate']) > 0.12:
                causes.append({'cause': f'Category delivery failures — {worst_cat_delay[0]} severely delayed',
                              'evidence': f'{worst_cat_delay[0]} has {round(float(worst_cat_delay[1]["category_delay_rate"])*100,1)}% delay rate',
                              'severity': float(worst_cat_delay[1]['category_delay_rate'])})
    causes.sort(key=lambda x: x['severity'], reverse=True)
    return {'root_cause_summary': {'total_causes_identified': len(causes), 'ranked_causes': causes}}

def report_agent(state):
    question = state['user_question']
    investigation_type = state['investigation_type']
    root_cause_summary = state['root_cause_summary']
    causes_text = "\n".join([
        f"{i+1}. {c['cause']}: {c['evidence']} (severity: {round(float(c['severity']),3)})"
        for i, c in enumerate(root_cause_summary['ranked_causes'])
    ])
    prompt = f"""You are a senior business analyst writing an executive report.

Business Question: {question}
Domains Investigated: {', '.join(investigation_type)}
Total Causes Identified: {root_cause_summary['total_causes_identified']}

Ranked Root Causes (by severity):
{causes_text}

Write a professional executive report with these exact sections:
- Executive Summary (2-3 sentences)
- Key Findings (bullet points with specific numbers)
- Root Cause Analysis (ranked causes with business impact)
- Recommendations (3-4 actionable items)

Be concise, data-driven, and professional. Use the exact numbers provided."""
    try:
        groq_key = os.getenv('GROQ_API_KEY') or st.secrets.get('GROQ_API_KEY')
        client = Groq(api_key=groq_key)
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1000,
            temperature=0.3
        )
        final_report = response.choices[0].message.content
    except Exception as e:
        final_report = f"LLM unavailable ({e}). Causes:\n{causes_text}"
    return {'final_report': final_report}

def route_after_planner(state):
    domain_to_node = {
        'review': 'review_agent_node',
        'delivery': 'delivery_agent_node',
        'seller': 'seller_agent_node',
        'category': 'category_agent_node',
    }
    return [domain_to_node[d] for d in state['investigation_type']] or END

@st.cache_resource
def build_graph():
    graph_builder = StateGraph(InsightAgentState)
    graph_builder.add_node("planner", planner_agent)
    graph_builder.add_node("review_agent_node", review_agent)
    graph_builder.add_node("delivery_agent_node", delivery_agent)
    graph_builder.add_node("seller_agent_node", seller_agent)
    graph_builder.add_node("category_agent_node", category_agent)
    graph_builder.add_node("root_cause_node", root_cause_agent)
    graph_builder.add_node("report_node", report_agent)
    graph_builder.add_edge(START, "planner")
    graph_builder.add_conditional_edges("planner", route_after_planner, {
        "review_agent_node": "review_agent_node",
        "delivery_agent_node": "delivery_agent_node",
        "seller_agent_node": "seller_agent_node",
        "category_agent_node": "category_agent_node",
        END: END
    })
    graph_builder.add_edge("review_agent_node", "root_cause_node")
    graph_builder.add_edge("delivery_agent_node", "root_cause_node")
    graph_builder.add_edge("seller_agent_node", "root_cause_node")
    graph_builder.add_edge("category_agent_node", "root_cause_node")
    graph_builder.add_edge("root_cause_node", "report_node")
    graph_builder.add_edge("report_node", END)
    return graph_builder.compile()

graph = build_graph()

# --- HEADER ---
st.markdown("# InsightAgent")
st.markdown("##### AI-Powered E-Commerce KPI Investigation System")
st.markdown("---")

# --- KPI OVERVIEW ROW ---
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Avg Review Score", f"{master_df['review_score'].mean():.2f} / 5.0")
with col2:
    st.metric("Delay Rate", f"{master_df['delay_flag'].mean()*100:.1f}%")
with col3:
    st.metric("Avg Delivery Days", f"{master_df['delivery_days'].mean():.1f}")
with col4:
    st.metric("Total Orders Analyzed", f"{master_df.shape[0]:,}")

st.markdown("---")

# --- SIDEBAR ---
with st.sidebar:
    st.markdown("### About")
    st.write("Multi-agent system analyzing 112,650+ Olist e-commerce transactions across customer satisfaction, delivery, seller, and category performance.")
    st.markdown("### Tech Stack")
    st.write("LangGraph · Groq (LLaMA 3.3 70B) · Pandas")
    st.markdown("### Architecture")
    st.write("Planner → 4 Parallel Analysis Agents → Root Cause Ranking → LLM Report Generation")

# --- INVESTIGATION CENTER ---
st.markdown("## Investigation Center")
question_input = st.text_input("Ask InsightAgent a business question...", placeholder="e.g., Why are customer review scores low?", key="question_box", label_visibility="collapsed")

all_questions = [
    "-- Select an example question --",
    "Why are customer review scores low?",
    "What is causing poor customer satisfaction?",
    "Why are customers unhappy?",
    "Which factors affect customer ratings the most?",
    "What drives negative reviews?",
    "Why are deliveries delayed?",
    "What causes late deliveries?",
    "Which sellers contribute most to delays?",
    "Which product categories experience the most delays?",
    "Which states have the slowest deliveries?",
    "Which sellers negatively affect customer experience?",
    "Who are the worst-performing sellers?",
    "Which sellers have the lowest ratings?",
    "Which sellers have the highest delay rates?",
    "Which sellers should be investigated?",
    "Which product categories have poor ratings?",
    "Which categories create customer dissatisfaction?",
    "Which categories suffer from delivery issues?",
    "Which categories are risky for customer experience?",
    "Which categories underperform overall?",
]

selected_example = st.selectbox("**Or choose a sample question:**", all_questions)

question_to_run = None
if st.button("Investigate", type="primary"):
    if question_input:
        question_to_run = question_input
    elif selected_example != "-- Select an example question --":
        question_to_run = selected_example



if question_to_run:
    # --- Agent pipeline visual ---
    with st.spinner("Running multi-agent investigation..."):
        result = graph.invoke({"user_question": question_to_run})

    st.markdown("### Agent Pipeline")
    domains_run = result['investigation_type']
    pipeline_cols = st.columns(6)
    for i, stage in enumerate(["Planner", "Review", "Delivery", "Seller", "Category", "Root Cause+Report"]):
        active = stage == "Planner" or stage.lower() in [d.lower() for d in domains_run] or stage == "Root Cause+Report"
        css_class = "agent-node agent-node-active" if active else "agent-node"
        pipeline_cols[i].markdown(f'<div class="{css_class}">{stage}</div>', unsafe_allow_html=True)

    st.markdown("---")
    st.markdown(f"### Investigation: *{question_to_run}*")
    st.write(f"**Domains Investigated:** {', '.join(result['investigation_type']).title()}")

    st.markdown("### Executive Report")
    st.markdown(result['final_report'])

    st.markdown("### Root Cause Evidence")
    for i, cause in enumerate(result['root_cause_summary']['ranked_causes'], 1):
        severity = float(cause['severity'])
        css_class = "cause-card" if severity > 0.5 else "cause-card-medium" if severity > 0.2 else "cause-card-low"
        st.markdown(f"""
        <div class="{css_class}">
        <b>Cause #{i}: {cause['cause']}</b><br>
        {cause['evidence']}<br>
        <span style="color:#94A3B8;">Severity score: {round(severity, 3)}</span>
        </div>
        """, unsafe_allow_html=True)