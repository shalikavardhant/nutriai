# ============================================================
# NutriAI — app.py  (Day 4: Streamlit UI)
# BAX-423 Final Project | UC Davis GSM
# ============================================================
# Run locally:  streamlit run app.py
# Deploy:       Push to GitHub → Streamlit Community Cloud
# ============================================================

import streamlit as st
import sqlite3
import json
import time
import pickle
import numpy as np
import pandas as pd
import faiss
from pybloom_live import BloomFilter
from sentence_transformers import SentenceTransformer
from fpdf import FPDF
import io
import os

# ── Page config ───────────────────────────────────────────────
st.set_page_config(
    page_title="NutriAI",
    page_icon="🥗",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Sans:wght@300;400;500;600&display=swap');

html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }

h1, h2, h3 { font-family: 'DM Serif Display', serif; }

.main { background: #FAFAF7; }

.stButton > button {
    background: #1A1A2E;
    color: white;
    border: none;
    border-radius: 8px;
    padding: 0.6rem 1.8rem;
    font-family: 'DM Sans', sans-serif;
    font-weight: 500;
    font-size: 0.95rem;
    transition: background 0.2s;
}
.stButton > button:hover { background: #2D2D4E; }

.meal-card {
    background: white;
    border-radius: 12px;
    padding: 1rem 1.2rem;
    margin-bottom: 0.6rem;
    border-left: 4px solid #4CAF82;
    box-shadow: 0 1px 4px rgba(0,0,0,0.06);
}
.meal-card h4 { margin: 0 0 0.3rem 0; font-size: 0.95rem; color: #1A1A2E; }
.meal-card p  { margin: 0; font-size: 0.8rem; color: #666; }

.macro-pill {
    display: inline-block;
    background: #F0F7F4;
    color: #2D7A56;
    border-radius: 20px;
    padding: 2px 10px;
    font-size: 0.75rem;
    font-weight: 500;
    margin-right: 4px;
}

.timer-badge {
    background: #1A1A2E;
    color: #4CAF82;
    border-radius: 8px;
    padding: 0.4rem 1rem;
    font-size: 0.85rem;
    font-weight: 600;
    display: inline-block;
    margin-top: 0.5rem;
}

.rda-bar-wrap {
    background: #EEF;
    border-radius: 4px;
    height: 8px;
    width: 100%;
    margin-top: 2px;
}
.rda-bar-fill {
    background: #4CAF82;
    border-radius: 4px;
    height: 8px;
}
.rda-bar-over { background: #E07070; }

.section-header {
    font-family: 'DM Serif Display', serif;
    font-size: 1.4rem;
    color: #1A1A2E;
    margin: 1.5rem 0 0.8rem 0;
    padding-bottom: 0.3rem;
    border-bottom: 2px solid #E8F5EE;
}

.persona-tag {
    background: #E8F5EE;
    color: #1A6640;
    border-radius: 6px;
    padding: 0.15rem 0.6rem;
    font-size: 0.78rem;
    font-weight: 500;
}
</style>
""", unsafe_allow_html=True)

# ── Constants ─────────────────────────────────────────────────
NUTRIENT_COLS = ["calories_kcal","protein_g","carbs_g","fat_g",
                 "fiber_g","calcium_mg","iron_mg","sodium_mg",
                 "potassium_mg","zinc_mg","vitamin_d_ug","vitamin_b12_ug"]

DAILY_RDA = {
    "calories_kcal": 2000, "protein_g": 50,   "carbs_g": 275,
    "fat_g": 78,           "fiber_g": 28,      "calcium_mg": 1000,
    "iron_mg": 18,         "vitamin_d_ug": 15, "vitamin_b12_ug": 2.4,
    "zinc_mg": 8,
}

SLOT_PROMPT_OPTIONS = {
    "breakfast": [
        "eggs scrambled omelette frittata protein morning",
        "oats porridge cereal granola fruit yogurt morning",
        "bread toast whole grain muffin baked breakfast",
        "smoothie fruit juice dairy milk morning drink",
    ],
    "lunch": [
        "salad vegetables greens legumes beans healthy lunch",
        "soup stew broth vegetable chicken lunch warm",
        "sandwich wrap grain rice bowl midday meal",
        "fish seafood tuna salmon light healthy lunch",
    ],
    "dinner": [
        "beef steak lamb red meat protein evening dinner",
        "chicken poultry turkey baked roasted dinner",
        "fish salmon cod tilapia seafood evening meal",
        "lentils beans legumes vegan plant protein dinner",
        "pasta rice grain vegetable casserole dinner",
    ],
}

def get_slot_prompt(slot, day):
    options = SLOT_PROMPT_OPTIONS.get(slot, ["healthy nutritious meal"])
    return options[(day - 1) % len(options)]

SLOT_CAL_MIN   = {"breakfast": 200, "lunch": 300, "dinner": 300}
SLOT_CAL_SPLIT = {"breakfast": 0.25, "lunch": 0.35, "dinner": 0.35}

SLOT_BLACKLIST = {
    "breakfast": ["Snacks","Sweets","Fats and Oils","Spices and Herbs","Beverages","Baby Foods"],
    "lunch":     ["Snacks","Sweets","Breakfast Cereals","Baby Foods","Fats and Oils","Spices and Herbs"],
    "dinner":    ["Snacks","Sweets","Breakfast Cereals","Baby Foods","Beverages","Fats and Oils","Spices and Herbs"],
}

SLOT_PREFERRED = {
    "breakfast": ["Dairy and Egg Products","Breakfast Cereals","Cereal Grains and Pasta",
                  "Fruits and Fruit Juices","Baked Products"],
    "lunch":     ["Legumes and Legume Products","Soups, Sauces, and Gravies",
                  "Meals, Entrees, and Side Dishes","Vegetables and Vegetable Products"],
    "dinner":    ["Beef Products","Pork Products","Poultry Products",
                  "Finfish and Shellfish Products","Legumes and Legume Products",
                  "Vegetables and Vegetable Products","Cereal Grains and Pasta"],
}

ALLERGEN_TAGS = ["gluten","dairy","nuts","shellfish","soy","eggs","peanuts","fish","tree_nuts"]

# ── Load assets (cached) ──────────────────────────────────────
@st.cache_resource(show_spinner="Loading NutriAI models…")
def load_assets():
    conn  = sqlite3.connect("data/nutriai_foods.db")
    df    = pd.read_sql("SELECT * FROM foods", conn)
    conn.close()

    with open("data/bloom_filters.pkl","rb") as f:
        bloom = pickle.load(f)

    index   = faiss.read_index("data/nutriai.index")
    col_max = np.load("data/nutrient_col_max.npy")
    model   = SentenceTransformer("all-MiniLM-L6-v2")
    return df, bloom, index, col_max, model

# ── Filter helpers ────────────────────────────────────────────
def bloom_exclude(fdc_ids, allergens, bloom):
    return {fid for tag in allergens if tag in bloom
            for fid in fdc_ids if fid in bloom[tag]}

def clinical_filter(df, conditions):
    mask = pd.Series([True]*len(df), index=df.index)
    log  = {}
    def lg(fid, r): log.setdefault(int(fid),[]).append(r)
    if "IBS" in conditions:
        m = df["is_high_fodmap"]==0
        [lg(r.fdc_id,"High-FODMAP food") for _,r in df[~m].iterrows()]; mask &= m
    if "GERD" in conditions:
        m = (df["is_gerd_trigger"]==0)&(df["fat_g"]<=20)
        [lg(r.fdc_id,"GERD trigger / high fat") for _,r in df[~m].iterrows()]; mask &= m
    if "T2D" in conditions:
        m = (df["gi_estimate"]<=70)&(df["total_sugars_g"]<=15)
        [lg(r.fdc_id,"High GI or sugar") for _,r in df[~m].iterrows()]; mask &= m
    if "Hypertension" in conditions:
        m = df["sodium_mg"]<=600
        [lg(r.fdc_id,"High sodium (DASH limit)") for _,r in df[~m].iterrows()]; mask &= m
    return df[mask].copy(), log

def diet_filter(df, diet):
    log = {}
    def lg(fid, r): log.setdefault(int(fid),[]).append(r)
    if   diet=="Vegan":        m=df["is_vegan"]==1
    elif diet=="Vegetarian":   m=df["is_vegetarian"]==1
    elif diet=="Pescatarian":  m=df["is_pescatarian"]==1
    elif diet=="Eggs":         m=df["description"].str.lower().str.contains("egg|omelet|frittata|quiche", na=False)
    elif diet=="Red Meat":     m=df["description"].str.lower().str.contains("beef|steak|lamb|pork|veal|bison|venison", na=False)
    elif diet=="Dairy":        m=df["description"].str.lower().str.contains("milk|cheese|yogurt|butter|cream|whey", na=False)
    elif diet=="Protein":      m=df["protein_g"] >= 15
    else: return df.copy(), log
    [lg(r.fdc_id,"Not suitable for diet") for _,r in df[~m].iterrows()]
    return df[m].copy(), log

# ── FAISS retrieval ───────────────────────────────────────────
def slot_query_vec(prompt, model, col_max, meal_targets):
    text_vec = model.encode([prompt])[0].astype(np.float32)
    nut_vec  = np.zeros(len(NUTRIENT_COLS), dtype=np.float32)
    for i,col in enumerate(NUTRIENT_COLS):
        nut_vec[i] = meal_targets.get(col,0)/(col_max[i] if col_max[i]>0 else 1)
    nut_pad = np.pad(nut_vec,(0,384-len(NUTRIENT_COLS)),mode="constant")
    combined = (0.60*text_vec + 0.40*nut_pad).astype(np.float32)
    faiss.normalize_L2(combined.reshape(1,-1))
    return combined.reshape(1,-1)

def retrieve(qvec, index, pool_df, safe_ids, full_df, k=80):
    D, I = index.search(qvec, min(k*5, index.ntotal))
    rows, scores = [], []
    for sc, idx in zip(D[0], I[0]):
        if idx<0 or idx>=len(full_df): continue
        row = full_df.iloc[idx]
        if int(row["fdc_id"]) not in safe_ids: continue
        if (row.get("calories_kcal") or 0) < 10: continue
        rows.append(row); scores.append(float(sc))
        if len(rows)>=k: break
    return (pd.DataFrame(rows), scores) if rows else (pd.DataFrame(), [])

# ── Ranking ───────────────────────────────────────────────────
def nutrient_gap_score(row, targets):
    score = 0.0
    W = {"calories_kcal":0.25,"protein_g":0.25,"carbs_g":0.15,
         "fat_g":0.10,"fiber_g":0.10,"calcium_mg":0.05,"iron_mg":0.05}
    for col,w in W.items():
        t = targets.get(col,0)
        if t<=0: continue
        ratio = (row.get(col,0) or 0)/t
        if 0.2<=ratio<=0.8:   score+=w
        elif ratio<0.2:       score+=w*(ratio/0.2)*0.5
        else:                 score+=w*max(0,1-(ratio-0.8)/0.8)
    return score

def rank(cands, scores, targets, profile):
    rows = cands.copy()
    rows["faiss_sim"]    = scores
    rows["nutrient_gap"] = rows.apply(lambda r: nutrient_gap_score(r.to_dict(),targets),axis=1)
    rows["preference"]   = 0.5
    if "T2D"          in profile.get("conditions",[]): rows["preference"]+=(70-rows["gi_estimate"].clip(0,70))/70*0.3
    if "Hypertension" in profile.get("conditions",[]): rows["preference"]+=(600-rows["sodium_mg"].clip(0,600))/600*0.3
    if "IBS"          in profile.get("conditions",[]): rows["preference"]+=(1-rows["is_high_fodmap"])*0.2
    rows["preference"] = rows["preference"].clip(0,1)
    rows["score"] = 0.50*rows["nutrient_gap"]+0.30*rows["faiss_sim"]+0.20*rows["preference"]
    return rows.sort_values("score",ascending=False)

# ── Diversity engine ──────────────────────────────────────────
class Diversity:
    def __init__(self):
        self.used   = set()
        self.hist   = {}
    def pick(self, ranked, slot):
        # Block any category already used in this slot (whole week)
        used_cats = set(self.hist.get(slot, []))
        # First pass: find food from a completely new category
        for _, row in ranked.iterrows():
            fid = int(row["fdc_id"])
            cat = str(row["category"])
            if fid in self.used: continue
            if cat not in used_cats:
                self.used.add(fid)
                self.hist.setdefault(slot, []).append(cat)
                return row
        # Second pass: allow repeated category if no new ones available
        for _, row in ranked.iterrows():
            fid = int(row["fdc_id"])
            if fid not in self.used:
                cat = str(row["category"])
                self.used.add(fid)
                self.hist.setdefault(slot, []).append(cat)
                return row
        return None
    def score(self):
        cats = [c for cs in self.hist.values() for c in cs]
        return round(len(set(cats))/len(cats),3) if cats else 0.0

# ── Core generator ────────────────────────────────────────────
def generate_plan(profile, df, bloom, index, col_max, model):
    t0     = time.perf_counter()
    exc_log = {}

    fdc_ids = df["fdc_id"].astype(int).tolist()
    exc_ids = bloom_exclude(fdc_ids, profile.get("allergens",[]), bloom)
    for fid in exc_ids:
        exc_log.setdefault(int(fid),[]).append(f"Allergen: {', '.join(profile['allergens'])}")

    safe_df = df[~df["fdc_id"].astype(int).isin(exc_ids)].copy()
    if profile.get("conditions"):
        safe_df, cl = clinical_filter(safe_df, profile["conditions"]); exc_log.update(cl)
    if profile.get("diet","None") not in ("None","none",""):
        safe_df, dl = diet_filter(safe_df, profile["diet"]); exc_log.update(dl)

    safe_ids = set(safe_df["fdc_id"].astype(int).tolist())
    cal_target = profile.get("calories", 2000)
    daily = {k: v*(cal_target/2000) for k,v in DAILY_RDA.items()}

    div  = Diversity()
    plan = []

    for day in range(1,8):
        remaining = daily.copy()
        for slot in ["breakfast","lunch","dinner"]:
            cal_min   = SLOT_CAL_MIN[slot]
            blacklist = SLOT_BLACKLIST[slot]
            preferred = SLOT_PREFERRED[slot]
            meal_tgt  = {k:v*SLOT_CAL_SPLIT[slot] for k,v in remaining.items()}
            meal_tgt["calories_kcal"] = max(meal_tgt.get("calories_kcal",0), cal_target*SLOT_CAL_SPLIT[slot])

            _prompt_options = SLOT_PROMPT_OPTIONS.get(slot, ["healthy nutritious meal"])
            _prompt = _prompt_options[(day - 1) % len(_prompt_options)]
            qvec = slot_query_vec(_prompt, model, col_max, meal_targets)

            pool = safe_df[
                (safe_df["meal_slot"].isin([slot,"any"])) &
                (safe_df["calories_kcal"]>=cal_min) &
                (~safe_df["category"].isin(blacklist))
            ].copy()
            if len(pool)<30:
                pool = safe_df[(safe_df["calories_kcal"]>=cal_min)&(~safe_df["category"].isin(blacklist))].copy()
            if len(pool)<15:
                pool = safe_df[safe_df["calories_kcal"]>=cal_min].copy()

            pool["cat_boost"] = pool["category"].isin(preferred).astype(float)*0.15
            cands, scores = retrieve(qvec, index, pool, safe_ids, df, k=80)
            if cands.empty: continue

            ranked = rank(cands, scores, meal_tgt, profile)
            if "cat_boost" in ranked.columns:
                ranked = ranked.copy()
                ranked["score"] = (ranked["score"]+ranked["cat_boost"]).clip(0,1)
                ranked = ranked.sort_values("score",ascending=False)

            pick = div.pick(ranked, slot)
            if pick is None: continue

            food = pick.to_dict() if hasattr(pick,"to_dict") else dict(pick)
            plan.append({
                "day":   day, "slot": slot,
                "fdc_id": int(food["fdc_id"]),
                "name":   food["description"],
                "cal":    round(food.get("calories_kcal",0),1),
                "prot":   round(food.get("protein_g",0),1),
                "carbs":  round(food.get("carbs_g",0),1),
                "fat":    round(food.get("fat_g",0),1),
                "fiber":  round(food.get("fiber_g",0),1),
                "ca":     round(food.get("calcium_mg",0),1),
                "fe":     round(food.get("iron_mg",0),1),
                "vd":     round(food.get("vitamin_d_ug",0),2),
                "b12":    round(food.get("vitamin_b12_ug",0),2),
                "zn":     round(food.get("zinc_mg",0),2),
                "score":  round(food.get("score",0),4),
                "category": food.get("category",""),
            })
            for k in remaining:
                remaining[k] = max(0, remaining[k]-(food.get(k,0) or 0))

    elapsed = (time.perf_counter()-t0)*1000
    return plan, exc_log, elapsed, div.score()

# ── Nutrient analysis ─────────────────────────────────────────
def analyse_plan(plan, cal_target):
    days = {}
    for m in plan:
        d = m["day"]
        if d not in days:
            days[d] = {k:0.0 for k in ["cal","prot","carbs","fat","fiber","ca","fe","vd","b12","zn"]}
        for k in days[d]: days[d][k] += m.get(k,0)

    rda = {
        "cal": cal_target, "prot":50, "carbs":275, "fat":78, "fiber":28,
        "ca":1000, "fe":18, "vd":15, "b12":2.4, "zn":8
    }
    labels = {
        "cal":"Calories (kcal)","prot":"Protein (g)","carbs":"Carbs (g)",
        "fat":"Fat (g)","fiber":"Fibre (g)","ca":"Calcium (mg)",
        "fe":"Iron (mg)","vd":"Vitamin D (µg)","b12":"Vitamin B12 (µg)","zn":"Zinc (mg)"
    }
    return days, rda, labels

# ── PDF export ────────────────────────────────────────────────
def make_pdf(plan, profile_summary):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Helvetica","B",16)
    pdf.cell(0,10,"NutriAI - 7-Day Meal Plan",ln=True)
    pdf.set_font("Helvetica","",10)
    pdf.cell(0,6,f"Profile: {profile_summary}",ln=True)
    pdf.cell(0,6,f"Generated: {time.strftime('%Y-%m-%d %H:%M')}",ln=True)
    pdf.ln(4)

    for day in range(1,8):
        pdf.set_font("Helvetica","B",12)
        pdf.set_fill_color(240,247,244)
        pdf.cell(0,8,f"Day {day}",ln=True,fill=True)
        pdf.set_font("Helvetica","",9)
        for m in [x for x in plan if x["day"]==day]:
            pdf.cell(25,6,m["slot"].capitalize(),border="B")
            pdf.cell(110,6,m["name"][:70],border="B")
            pdf.cell(25,6,f"{m['cal']:.0f} kcal",border="B")
            pdf.cell(0,6,f"P:{m['prot']}g C:{m['carbs']}g F:{m['fat']}g",border="B",ln=True)
        pdf.ln(2)

    buf = io.BytesIO()
    buf.write(pdf.output())
    buf.seek(0)
    return buf

# ── CSV export ────────────────────────────────────────────────
def make_csv(plan):
    df_out = pd.DataFrame(plan)[["day","slot","name","cal","prot","carbs","fat","fiber","ca","fe","vd","b12","zn"]]
    df_out.columns = ["Day","Slot","Food","Calories","Protein_g","Carbs_g","Fat_g","Fiber_g",
                      "Calcium_mg","Iron_mg","VitD_ug","VitB12_ug","Zinc_mg"]
    return df_out.to_csv(index=False).encode()

# ══════════════════════════════════════════════════════════════
# UI
# ══════════════════════════════════════════════════════════════

st.markdown("<h1 style='font-family:DM Serif Display,serif;font-size:2.4rem;color:#1A1A2E;margin-bottom:0'>🥗 NutriAI</h1>", unsafe_allow_html=True)
st.markdown("<p style='color:#666;margin-top:0;margin-bottom:1.5rem;font-size:1rem'>Personalised 7-day meal plans · clinically tailored · generated in seconds</p>", unsafe_allow_html=True)

# ── Sidebar: user profile ─────────────────────────────────────
with st.sidebar:
    st.markdown("### Your Profile")

    name_input = st.text_input("Name (optional)", placeholder="e.g. Doe")
    age        = st.slider("Age", 18, 80, 18)
    sex        = st.selectbox("Sex", ["Female","Male","Other"])
    weight_kg  = st.number_input("Weight (kg)", 40, 200, 40)
    cal_target = st.slider("Daily calorie target (kcal)", 1400, 3000, 2000, step=50)

    st.markdown("---")
    st.markdown("**Clinical conditions**")
    conditions = st.multiselect("Select all that apply",
        ["IBS","GERD","T2D","Hypertension"], default=[])

    st.markdown("**Dietary preference**")
    diet = st.selectbox("Diet type", ["None","Protein","Vegan","Vegetarian","Pescatarian","Eggs","Red Meat","Dairy"])

    st.markdown("**Allergens to exclude**")
    allergens = st.multiselect("Select allergens",
        ["Gluten","Dairy","Nuts","Shellfish","Soy","Eggs","Peanuts","Fish","Tree Nuts","Red Meat"],
        default=[])

    st.markdown("---")
    generate_btn = st.button("🚀 Generate My Plan", use_container_width=True)

# ── Main area ─────────────────────────────────────────────────
df_foods, bloom, index, col_max, model = load_assets()

if generate_btn:
    profile = {
        "allergens":  allergens,
        "conditions": conditions,
        "diet":       diet if diet != "None" else "none",
        "calories":   cal_target,
    }

    with st.spinner("Building your personalised plan…"):
        plan, exc_log, elapsed_ms, div_score = generate_plan(
            profile, df_foods, bloom, index, col_max, model
        )

    st.session_state["plan"]      = plan
    st.session_state["exc_log"]   = exc_log
    st.session_state["elapsed"]   = elapsed_ms
    st.session_state["div_score"] = div_score
    st.session_state["profile"]   = profile
    st.session_state["cal_target"]= cal_target

if "plan" in st.session_state:
    plan      = st.session_state["plan"]
    exc_log   = st.session_state["exc_log"]
    elapsed   = st.session_state["elapsed"]
    div_score = st.session_state["div_score"]
    cal_target= st.session_state["cal_target"]
    profile   = st.session_state["profile"]

    # ── Header metrics ────────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Meals generated", f"{len(plan)} / 21")
    c2.metric("Generation time", f"{elapsed:.0f} ms")
    c3.metric("Diversity score", f"{div_score:.2f}")
    c4.metric("Foods excluded", len(exc_log))

    st.markdown(f'<div class="timer-badge">⚡ Generated in {elapsed:.0f} ms — well under the 60s target</div>',
                unsafe_allow_html=True)

    # ── Tabs ──────────────────────────────────────────────────
    tab1, tab2, tab3 = st.tabs(["📅 7-Day Plan", "📊 Nutrient Analysis", "❌ Why Excluded"])

    # ── TAB 1: Meal Plan ──────────────────────────────────────
    with tab1:
        day_cols = st.columns(7)
        for day in range(1,8):
            with day_cols[day-1]:
                st.markdown(f"**Day {day}**")
                for slot in ["breakfast","lunch","dinner"]:
                    meals = [m for m in plan if m["day"]==day and m["slot"]==slot]
                    if not meals:
                        st.markdown(f"<div class='meal-card'><p style='color:#aaa'>{slot.capitalize()}: —</p></div>",
                                    unsafe_allow_html=True)
                        continue
                    m = meals[0]
                    slot_emoji = {"breakfast":"🌅","lunch":"☀️","dinner":"🌙"}.get(slot,"🍽")
                    st.markdown(f"""
                    <div class='meal-card'>
                        <h4>{slot_emoji} {slot.capitalize()}</h4>
                        <p style='font-weight:500;color:#1A1A2E;margin-bottom:4px'>{m['name'][:50]}</p>
                        <span class='macro-pill'>{m['cal']:.0f} kcal</span>
                        <span class='macro-pill'>P {m['prot']}g</span>
                        <span class='macro-pill'>C {m['carbs']}g</span>
                        <span class='macro-pill'>F {m['fat']}g</span>
                    </div>""", unsafe_allow_html=True)

        # Export buttons
        st.markdown("---")
        ec1, ec2 = st.columns(2)
        profile_str = f"{diet} | {', '.join(conditions) if conditions else 'No conditions'} | Allergens: {', '.join(allergens) if allergens else 'None'}"
        with ec1:
            pdf_buf = make_pdf(plan, profile_str)
            st.download_button("📄 Download PDF", pdf_buf, "nutriai_plan.pdf", "application/pdf", use_container_width=True)
        with ec2:
            csv_bytes = make_csv(plan)
            st.download_button("📊 Download CSV", csv_bytes, "nutriai_plan.csv", "text/csv", use_container_width=True)

    # ── TAB 2: Nutrient Analysis ──────────────────────────────
    with tab2:
        days_data, rda, labels = analyse_plan(plan, cal_target)

        st.markdown("<div class='section-header'>Daily Nutrient Totals vs RDA</div>", unsafe_allow_html=True)

        # Summary table
        rows = []
        for k, label in labels.items():
            vals = [days_data[d].get(k,0) for d in range(1,8)]
            avg  = sum(vals)/len(vals)
            pct  = min(avg/rda[k]*100, 200) if rda[k]>0 else 0
            flag = "✅" if 80<=pct<=150 else ("⚠️" if pct<80 else "🔴")
            rows.append({"Nutrient":label,"Avg/day":f"{avg:.1f}","RDA":f"{rda[k]}",
                         "% RDA":f"{pct:.0f}%","Status":flag})
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        # Per-day calorie chart
        st.markdown("<div class='section-header'>Calories per Day</div>", unsafe_allow_html=True)
        cal_data = pd.DataFrame({
            "Day":   [f"Day {d}" for d in range(1,8)],
            "Calories": [days_data[d]["cal"] for d in range(1,8)]
        })
        st.bar_chart(cal_data.set_index("Day"))

        # Macro breakdown per day
        st.markdown("<div class='section-header'>Macro Breakdown</div>", unsafe_allow_html=True)
        macro_data = pd.DataFrame({
            "Day":   [f"Day {d}" for d in range(1,8)],
            "Protein (g)":  [days_data[d]["prot"] for d in range(1,8)],
            "Carbs (g)":    [days_data[d]["carbs"] for d in range(1,8)],
            "Fat (g)":      [days_data[d]["fat"] for d in range(1,8)],
            "Fibre (g)":    [days_data[d]["fiber"] for d in range(1,8)],
        })
        st.line_chart(macro_data.set_index("Day"))

    # ── TAB 3: Why Excluded ───────────────────────────────────
    with tab3:
        st.markdown("<div class='section-header'>Exclusion Log</div>", unsafe_allow_html=True)
        st.caption(f"{len(exc_log)} foods excluded from your plan")

        if exc_log:
            exc_rows = []
            for fid, reasons in list(exc_log.items())[:100]:
                food_name = df_foods.loc[df_foods["fdc_id"]==fid,"description"].values
                name_str  = food_name[0][:60] if len(food_name)>0 else f"FDC #{fid}"
                exc_rows.append({"Food": name_str, "Reason(s)": " | ".join(set(reasons))})
            st.dataframe(pd.DataFrame(exc_rows), use_container_width=True, hide_index=True)
        else:
            st.info("No foods were excluded for your profile.")

else:
    # Landing state
    st.markdown("""
    <div style='text-align:center;padding:3rem 2rem;background:white;border-radius:16px;margin-top:1rem'>
        <div style='font-size:3rem;margin-bottom:1rem'>🥗</div>
        <h2 style='font-family:DM Serif Display,serif;color:#1A1A2E;margin-bottom:0.5rem'>
            Your personalised meal plan awaits
        </h2>
        <p style='color:#888;max-width:480px;margin:0 auto 1.5rem auto'>
            Fill in your profile on the left — clinical conditions, allergens, dietary 
            preference — then hit Generate. Your 7-day plan builds in under 60 seconds.
        </p>
        <div style='display:flex;justify-content:center;gap:2rem;flex-wrap:wrap;margin-top:1.5rem'>
            <div style='text-align:center'>
                <div style='font-size:1.5rem'>🔬</div>
                <div style='font-size:0.8rem;color:#666;margin-top:4px'>Clinical filtering</div>
            </div>
            <div style='text-align:center'>
                <div style='font-size:1.5rem'>⚡</div>
                <div style='font-size:0.8rem;color:#666;margin-top:4px'>Sub-60s generation</div>
            </div>
            <div style='text-align:center'>
                <div style='font-size:1.5rem'>📊</div>
                <div style='font-size:0.8rem;color:#666;margin-top:4px'>Nutrient analysis</div>
            </div>
            <div style='text-align:center'>
                <div style='font-size:1.5rem'>📄</div>
                <div style='font-size:0.8rem;color:#666;margin-top:4px'>PDF + CSV export</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
