import os, asyncio, uuid, time, json, requests
from pathlib import Path
from dotenv import load_dotenv
from fastapi import FastAPI, Query
from fastapi.responses import HTMLResponse, JSONResponse

# Load .env file if it exists (local dev fallback)
load_dotenv()
# On Railway, API keys come from dashboard environment variables
# Required: DEEPSEEK_API_KEY, GROQ_API_KEY, GOOGLE_API_KEY

MISSING_KEYS = [k for k in ("DEEPSEEK_API_KEY", "GROQ_API_KEY", "GOOGLE_API_KEY") if not os.environ.get(k)]
if MISSING_KEYS:
    print(f"WARNING: Missing env vars: {', '.join(MISSING_KEYS)}")

analysis_results = {}
app = FastAPI(title="TradingAgents — AI Multi-Agent Trading Analysis Platform")

STAGES = [
    "جمع البيانات وتحليل السوق",
    "تحليل الأخبار والمؤشرات",
    "مناقشة فريق المحللين",
    "تقييم المخاطر",
    "اتخاذ القرار النهائي",
    "ترجمة التقارير للعربية",
]

PROVIDER_MODELS = {
    "groq":      {"deep": "llama-3.3-70b-versatile",   "quick": "llama-3.3-70b-versatile"},
    "ollama":    {"deep": "llama3.2:1b",             "quick": "llama3.2:1b"},
    "google":    {"deep": "gemini-3.1-pro-preview",  "quick": "gemini-3.5-flash"},
    "openai":    {"deep": "gpt-5.5",                 "quick": "gpt-5.4-mini"},
    "deepseek":  {"deep": "deepseek-v4-pro",          "quick": "deepseek-v4-flash"},
}

AGENT_FILTER_MAP = {
    "all": ["market_report", "sentiment_report", "news_report", "fundamentals_report", "debate", "risk_debate", "trader_investment_plan", "investment_plan", "final_trade_decision"],
    "technical": ["market_report"],
    "sentiment": ["sentiment_report"],
    "news": ["news_report"],
    "fundamental": ["fundamentals_report"],
    "risk": ["risk_debate"],
    "debate": ["debate"],
}

@app.get("/", response_class=HTMLResponse)
async def index():
    path = Path(__file__).parent / "templates" / "index.html"
    return HTMLResponse(content=path.read_text(encoding="utf-8"))

@app.post("/analyze")
async def analyze(
    ticker: str = Query(...),
    llm_provider: str = Query("ollama"),
    trade_date: str = Query(None),
    agents: str = Query("all"),
):
    if not trade_date:
        from datetime import datetime
        trade_date = datetime.now().strftime("%Y-%m-%d")
    task_id = uuid.uuid4().hex[:8]
    start = time.time()
    analysis_results[task_id] = {
        "status": "running", "result": None,
        "start_time": start, "stage": 0,
        "progress": 0, "log": [],
        "provider": llm_provider,
        "agents": agents,
    }
    loop = asyncio.get_event_loop()
    loop.run_in_executor(None, _run_task, task_id, ticker, llm_provider, trade_date, agents)
    return JSONResponse({"task_id": task_id, "status": "running"})

@app.get("/result/{task_id}")
async def get_result(task_id: str):
    data = analysis_results.get(task_id, {"status": "not_found", "result": None})
    if data.get("status") == "done":
        data["elapsed_seconds"] = round(time.time() - data.get("start_time", time.time()), 1)
        data["progress"] = 100
    elif data.get("status") == "translating":
        data["elapsed_seconds"] = round(time.time() - data.get("start_time", time.time()), 1)
        data["stage"] = len(STAGES) - 1
        data["progress"] = min(98, data.get("progress", 90))
    elif data.get("status") == "running":
        data["elapsed_seconds"] = round(time.time() - data.get("start_time", time.time()), 1)
        elapsed = data["elapsed_seconds"]
        est_total = {"deepseek": 120, "groq": 180, "google": 150, "openai": 200, "ollama": 1800}
        est = est_total.get(data.get("provider", "deepseek"), 180)
        data["progress"] = min(88, int((elapsed / est) * 100))
        stg = min(len(STAGES) - 2, int((elapsed / est) * (len(STAGES) - 1)))
        data["stage"] = stg
    return JSONResponse(data)

def _filter_by_agents(final_state, agents_param):
    """Filter final_state fields based on selected agents."""
    if agents_param == "all" or not agents_param:
        return final_state
    
    selected = agents_param.split(",")
    allowed_keys = set()
    for a in selected:
        for key in AGENT_FILTER_MAP.get(a, []):
            allowed_keys.add(key)
    
    # Always include these core fields
    allowed_keys.update(["instrument_context", "company_of_interest"])
    
    filtered = {}
    for key in allowed_keys:
        if key in final_state:
            filtered[key] = final_state[key]
    
    # For debate/risk, include the full state objects
    if "debate" in allowed_keys and "investment_debate_state" in final_state:
        filtered["investment_debate_state"] = final_state["investment_debate_state"]
    if "risk_debate" in allowed_keys and "risk_debate_state" in final_state:
        filtered["risk_debate_state"] = final_state["risk_debate_state"]
    
    return filtered

# ============================================
# TRANSLATION ENGINE
# ============================================
TRANSLATION_SYSTEM_PROMPT = """أنت مترجم مالي محترف. ترجم النص التالي من الإنجليزية إلى العربية بدقة عالية.
قواعد الترجمة:
- حافظ على نفس التنسيق والهيكل (العناوين، القوائم، الجداول، الأرقام)
- حافظ على المصطلحات المالية الفنية مع ترجمتها (مثل: RSI = مؤشر القوة النسبية، MACD = الماكد، Moving Average = المتوسط المتحرك)
- لا تغير الأرقام أو الرموز المالية أو أسماء الأزواج
- حافظ على علامات الماركداون (# ## ### ** | )
- اكتب الترجمة فقط بدون أي مقدمات أو تعليقات
- إذا كان النص يحتوي بالفعل على عربية، أبقه كما هو"""

def _translate_text(text: str, max_retries: int = 2) -> str:
    """Translate text to Arabic using Groq API (primary) or DeepSeek (fallback)."""
    if not text or not text.strip():
        return text
    
    # Skip if text is already mostly Arabic
    arabic_chars = sum(1 for c in text if '\u0600' <= c <= '\u06FF')
    if arabic_chars > len(text) * 0.4:
        return text
    
    # Truncate very long texts to avoid token limits
    max_chars = 12000
    if len(text) > max_chars:
        text = text[:max_chars] + "\n\n[... تم اختصار باقي النص ...]" 
    
    providers = []
    
    # Primary: Groq (fast)
    groq_key = os.environ.get("GROQ_API_KEY")
    if groq_key:
        providers.append({
            "url": "https://api.groq.com/openai/v1/chat/completions",
            "key": groq_key,
            "model": "llama-3.3-70b-versatile",
            "name": "Groq",
        })
    
    # Fallback: DeepSeek
    ds_key = os.environ.get("DEEPSEEK_API_KEY")
    if ds_key:
        providers.append({
            "url": "https://api.deepseek.com/v1/chat/completions",
            "key": ds_key,
            "model": "deepseek-chat",
            "name": "DeepSeek",
        })
    
    # Fallback: Google Gemini
    google_key = os.environ.get("GOOGLE_API_KEY")
    if google_key:
        providers.append({
            "url": f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={google_key}",
            "key": google_key,
            "model": "gemini",
            "name": "Google",
        })
    
    for provider in providers:
        for attempt in range(max_retries):
            try:
                if provider["name"] == "Google":
                    # Google Gemini API format
                    resp = requests.post(
                        provider["url"],
                        headers={"Content-Type": "application/json"},
                        json={
                            "contents": [{
                                "parts": [{"text": f"{TRANSLATION_SYSTEM_PROMPT}\n\n---\n\n{text}"}]
                            }],
                            "generationConfig": {"temperature": 0.2}
                        },
                        timeout=120
                    )
                    if resp.status_code == 200:
                        data = resp.json()
                        translated = data["candidates"][0]["content"]["parts"][0]["text"]
                        return translated.strip()
                else:
                    # OpenAI-compatible API format (Groq, DeepSeek)
                    resp = requests.post(
                        provider["url"],
                        headers={
                            "Authorization": f"Bearer {provider['key']}",
                            "Content-Type": "application/json",
                        },
                        json={
                            "model": provider["model"],
                            "messages": [
                                {"role": "system", "content": TRANSLATION_SYSTEM_PROMPT},
                                {"role": "user", "content": text}
                            ],
                            "temperature": 0.2,
                            "max_tokens": 8000,
                        },
                        timeout=120
                    )
                    if resp.status_code == 200:
                        data = resp.json()
                        translated = data["choices"][0]["message"]["content"]
                        return translated.strip()
                    
                print(f"Translation attempt {attempt+1} with {provider['name']} failed: {resp.status_code} - {resp.text[:200]}")
            except Exception as e:
                print(f"Translation error with {provider['name']}: {e}")
        
    print("WARNING: All translation providers failed, returning original text")
    return text


SUMMARY_SYSTEM_PROMPT = """أنت محلل مالي محترف ومتداول خبير بخبرة تزيد عن 15 سنة في الأسواق المالية. قم بتحليل تقرير التحليل المالي التالي للرمز {ticker} وإنتاج ملخص تداول احترافي عالي الدقة.

يجب أن يحتوي الملخص بدقة على الأقسام التالية باللغة العربية الفصحى وبتنسيق Markdown:

## 🎯 التوصية والاتجاه
- **التوصية**: (شراء قوي ✅ / شراء ✅ / انتظار ومراقبة ⏸️ / بيع 🔴 / بيع قوي 🔴🔴)
- **الاتجاه العام**: (صاعد / هابط / عرضي)
- **الإطار الزمني**: (سكالبينج / يومي / سوينج / متوسط المدى)
- **جملة واحدة توضح السبب الرئيسي للتوصية**

## 📊 نسبة الثقة
- **نسبة الثقة**: [رقم من 0 إلى 100]%
- **تقييم قوة الإشارة**: (ضعيفة / متوسطة / قوية / قوية جداً)
- **عدد المؤشرات المتوافقة**: [عدد] من [إجمالي] مؤشر
- يجب أن تكون نسبة الثقة مبنية على توافق المؤشرات الفنية + الأساسية + المعنويات + الأخبار. إذا تتوافق 3 من 4 عوامل = 75%، إذا تتوافق كل العوامل = 85-95%، إذا عاملين فقط = 50-65%.

## 💰 مستويات التداول الدقيقة
- **سعر الدخول المقترح**: [السعر الدقيق أو نطاق ضيق]
- **وقف الخسارة (Stop Loss)**: [السعر الدقيق] — [المسافة بالنقاط والنسبة المئوية من الدخول]
- **الهدف الأول (TP1)**: [السعر] — [المسافة بالنقاط والنسبة المئوية]
- **الهدف الثاني (TP2)**: [السعر] — [المسافة بالنقاط والنسبة المئوية]
- **الهدف الثالث (TP3)**: [السعر] — [المسافة بالنقاط والنسبة المئوية]
- **نسبة المخاطرة إلى العائد (R:R)**: [النسبة بناءً على TP2 / المسافة لوقف الخسارة]

## 🔬 التحليل الفني المتعدد الأطر الزمنية
- **الإطار اليومي (D1)**: [اتجاه + أهم مستوى دعم ومقاومة]
- **إطار 4 ساعات (H4)**: [اتجاه + إشارات]
- **إطار الساعة (H1)**: [اتجاه + نقطة الدخول المحتملة]
- **المؤشرات الرئيسية**: RSI=[قيمة] | MACD=[إشارة] | المتوسطات المتحركة=[اتجاه]

## ✅ شروط الدخول المثالية
- [شرط 1 يجب تحققه قبل الدخول]
- [شرط 2 يجب تحققه قبل الدخول]
- [شرط 3 يجب تحققه قبل الدخول]
- **شرط إبطال الصفقة**: [ما الذي يُبطل هذا التحليل بالكامل]

## 📰 أهم العوامل المؤثرة
- [نقطة 1: عامل فني أو أساسي مؤثر]
- [نقطة 2: خبر أو حدث اقتصادي مؤثر]
- [نقطة 3: معنويات السوق]

قواعد صارمة وحاسمة:
- استخرج مستويات التداول (الدخول، وقف الخسارة، الأهداف) بدقة من نص التقرير. إذا لم تُذكر صراحةً، احسبها بناءً على أقرب مستويات الدعم والمقاومة والسعر الحالي.
- نسبة الثقة يجب أن تكون رقماً محدداً (مثل 78%) وليست نطاقاً.
- نسبة R:R يجب أن تكون محسوبة فعلياً (مثل 1:2.5).
- لا تكتب مقدمات أو ترحيبات. ابدأ مباشرة.
- الأرقام يجب أن تكون واقعية ومنطقية بناءً على تحركات السعر الفعلية.
"""

def _summarize_and_format_report(text: str, ticker: str) -> str:
    """Summarize and format the trade decision to be concise, useful, and in Arabic."""
    if not text or not text.strip():
        return text
    
    prompt = SUMMARY_SYSTEM_PROMPT.format(ticker=ticker)
    
    providers = []
    
    # Primary: Groq (fast)
    groq_key = os.environ.get("GROQ_API_KEY")
    if groq_key:
        providers.append({
            "url": "https://api.groq.com/openai/v1/chat/completions",
            "key": groq_key,
            "model": "llama-3.3-70b-versatile",
            "name": "Groq",
        })
    
    # Fallback: DeepSeek
    ds_key = os.environ.get("DEEPSEEK_API_KEY")
    if ds_key:
        providers.append({
            "url": "https://api.deepseek.com/v1/chat/completions",
            "key": ds_key,
            "model": "deepseek-chat",
            "name": "DeepSeek",
        })
        
    # Fallback: Google Gemini
    google_key = os.environ.get("GOOGLE_API_KEY")
    if google_key:
        providers.append({
            "url": f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={google_key}",
            "key": google_key,
            "model": "gemini",
            "name": "Google",
        })
        
    for provider in providers:
        for attempt in range(2):
            try:
                if provider["name"] == "Google":
                    resp = requests.post(
                        provider["url"],
                        headers={"Content-Type": "application/json"},
                        json={
                            "contents": [{
                                "parts": [{"text": f"{prompt}\n\n---\n\n{text}"}]
                            }],
                            "generationConfig": {"temperature": 0.2}
                        },
                        timeout=120
                    )
                    if resp.status_code == 200:
                        data = resp.json()
                        return data["candidates"][0]["content"]["parts"][0]["text"].strip()
                else:
                    resp = requests.post(
                        provider["url"],
                        headers={
                            "Authorization": f"Bearer {provider['key']}",
                            "Content-Type": "application/json",
                        },
                        json={
                            "model": provider["model"],
                            "messages": [
                                {"role": "system", "content": prompt},
                                {"role": "user", "content": text}
                            ],
                            "temperature": 0.2,
                            "max_tokens": 4000,
                        },
                        timeout=120
                    )
                    if resp.status_code == 200:
                        data = resp.json()
                        return data["choices"][0]["message"]["content"].strip()
                print(f"Summarization attempt {attempt+1} with {provider['name']} failed: {resp.status_code}")
            except Exception as e:
                print(f"Summarization error with {provider['name']}: {e}")
                
    return text


def _extract_trading_levels(summary_text: str) -> dict:
    """Extract structured trading levels from the Arabic summary report."""
    import re
    levels = {
        "entry": "—",
        "stop_loss": "—",
        "tp1": "—",
        "tp2": "—",
        "tp3": "—",
        "rr_ratio": "—",
        "confidence": "—",
        "confidence_num": 0,
        "signal_strength": "—",
        "timeframe": "—",
        "direction": "—",
        "entry_conditions": [],
        "invalidation": "—",
    }
    if not summary_text:
        return levels

    lines = summary_text.split("\n")
    for line in lines:
        line_lower = line.strip()
        
        # Extract entry price
        if any(k in line_lower for k in ["سعر الدخول", "entry", "منطقة الدخول"]):
            nums = re.findall(r'\d+[\.,]?\d*', line_lower)
            if nums:
                levels["entry"] = nums[0].replace(',', '.')
        
        # Extract stop loss
        if any(k in line_lower for k in ["وقف الخسارة", "stop loss", "stop_loss"]):
            nums = re.findall(r'\d+[\.,]?\d*', line_lower)
            if nums:
                levels["stop_loss"] = nums[0].replace(',', '.')
        
        # Extract TP1
        if any(k in line_lower for k in ["الهدف الأول", "tp1", "take profit 1"]):
            nums = re.findall(r'\d+[\.,]?\d*', line_lower)
            if nums:
                levels["tp1"] = nums[0].replace(',', '.')
        
        # Extract TP2
        if any(k in line_lower for k in ["الهدف الثاني", "tp2", "take profit 2"]):
            nums = re.findall(r'\d+[\.,]?\d*', line_lower)
            if nums:
                levels["tp2"] = nums[0].replace(',', '.')
        
        # Extract TP3
        if any(k in line_lower for k in ["الهدف الثالث", "tp3", "take profit 3"]):
            nums = re.findall(r'\d+[\.,]?\d*', line_lower)
            if nums:
                levels["tp3"] = nums[0].replace(',', '.')
        
        # Extract R:R ratio
        if any(k in line_lower for k in ["مخاطرة", "r:r", "r/r", "risk", "عائد"]):
            rr_match = re.findall(r'1\s*[:/]\s*(\d+[\.,]?\d*)', line_lower)
            if rr_match:
                levels["rr_ratio"] = f"1:{rr_match[0].replace(',', '.')}"
            else:
                nums = re.findall(r'\d+[\.,]?\d*', line_lower)
                if len(nums) >= 2:
                    levels["rr_ratio"] = f"1:{nums[-1].replace(',', '.')}"
        
        # Extract confidence
        if any(k in line_lower for k in ["نسبة الثقة", "confidence", "الثقة"]):
            pct_match = re.findall(r'(\d+)\s*%', line_lower)
            if pct_match:
                levels["confidence"] = f"{pct_match[0]}%"
                levels["confidence_num"] = int(pct_match[0])
        
        # Extract signal strength
        if any(k in line_lower for k in ["قوة الإشارة", "signal"]):
            if "قوية جداً" in line_lower or "very strong" in line_lower.lower():
                levels["signal_strength"] = "قوية جداً"
            elif "قوية" in line_lower or "strong" in line_lower.lower():
                levels["signal_strength"] = "قوية"
            elif "متوسطة" in line_lower or "medium" in line_lower.lower():
                levels["signal_strength"] = "متوسطة"
            elif "ضعيفة" in line_lower or "weak" in line_lower.lower():
                levels["signal_strength"] = "ضعيفة"
        
        # Extract timeframe
        if any(k in line_lower for k in ["الإطار الزمني", "timeframe"]):
            if "سكالبينج" in line_lower or "scalp" in line_lower.lower():
                levels["timeframe"] = "سكالبينج"
            elif "يومي" in line_lower or "daily" in line_lower.lower() or "intraday" in line_lower.lower():
                levels["timeframe"] = "يومي"
            elif "سوينج" in line_lower or "swing" in line_lower.lower():
                levels["timeframe"] = "سوينج"
            elif "متوسط" in line_lower or "medium" in line_lower.lower():
                levels["timeframe"] = "متوسط المدى"
        
        # Extract direction
        if any(k in line_lower for k in ["الاتجاه العام", "direction", "الاتجاه"]):
            if "صاعد" in line_lower or "bullish" in line_lower.lower():
                levels["direction"] = "صاعد 📈"
            elif "هابط" in line_lower or "bearish" in line_lower.lower():
                levels["direction"] = "هابط 📉"
            elif "عرضي" in line_lower or "sideways" in line_lower.lower():
                levels["direction"] = "عرضي ↔️"
        
        # Extract entry conditions
        if any(k in line_lower for k in ["شروط الدخول", "entry condition"]):
            # Start collecting conditions from next lines
            pass
        if line_lower.startswith("- ") and any(k in line_lower for k in ["كسر", "اختراق", "ارتداد", "تأكيد", "إغلاق", "يجب"]):
            condition = line_lower.lstrip("- ").strip()
            if condition and len(condition) > 5:
                levels["entry_conditions"].append(condition)
        
        # Extract invalidation
        if any(k in line_lower for k in ["إبطال", "invalidat"]):
            invalidation_text = line_lower
            for prefix in ["**شرط إبطال الصفقة**:", "شرط إبطال الصفقة:", "- "]:
                invalidation_text = invalidation_text.replace(prefix, "")
            invalidation_text = invalidation_text.strip().strip("*").strip()
            if invalidation_text and len(invalidation_text) > 5:
                levels["invalidation"] = invalidation_text
    
    # Limit entry conditions to 4
    levels["entry_conditions"] = levels["entry_conditions"][:4]
    
    return levels


def _translate_results(result: dict, task_id: str) -> dict:
    """Translate all report fields in the result to Arabic."""
    if not result or "final_state" not in result:
        return result
    
    fs = result["final_state"]
    ticker = result.get("ticker", "")
    
    # First, generate the concise summary decision from final_trade_decision
    final_decision = fs.get("final_trade_decision")
    if final_decision and isinstance(final_decision, str) and final_decision.strip():
        try:
            print(f"[{task_id}] Generating concise Arabic summary from final decision...")
            fs["summary_decision"] = _summarize_and_format_report(final_decision, ticker)
            print(f"[{task_id}] ✓ Concise summary generated successfully!")
        except Exception as se:
            print(f"[{task_id}] WARNING: Summary generation failed: {se}")
            fs["summary_decision"] = ""
    
    # Extract structured trading levels from the summary
    try:
        trading_levels = _extract_trading_levels(fs.get("summary_decision", ""))
        fs["trading_levels"] = trading_levels
        print(f"[{task_id}] ✓ Trading levels extracted: confidence={trading_levels.get('confidence', '—')}")
    except Exception as tle:
        print(f"[{task_id}] WARNING: Trading levels extraction failed: {tle}")
        fs["trading_levels"] = {}

    # Fields to translate
    text_fields = [
        "market_report", "sentiment_report", "news_report",
        "fundamentals_report", "trader_investment_plan",
        "investment_plan", "final_trade_decision",
    ]
    
    total = sum(1 for f in text_fields if fs.get(f))
    translated_count = 0
    
    for field in text_fields:
        content = fs.get(field)
        if content and isinstance(content, str) and content.strip():
            print(f"[{task_id}] Translating {field} ({len(content)} chars)...")
            fs[field] = _translate_text(content)
            translated_count += 1
            print(f"[{task_id}] ✓ Translated {field} ({translated_count}/{total})")
    
    # Translate debate fields
    if fs.get("debate") and isinstance(fs["debate"], dict):
        for key in ["bull", "bear", "judge"]:
            val = fs["debate"].get(key)
            if val and isinstance(val, str) and val.strip():
                print(f"[{task_id}] Translating debate.{key}...")
                fs["debate"][key] = _translate_text(val)
    
    # Translate risk debate fields
    if fs.get("risk_debate") and isinstance(fs["risk_debate"], dict):
        for key in ["conservative", "neutral", "aggressive", "judge"]:
            val = fs["risk_debate"].get(key)
            if val and isinstance(val, str) and val.strip():
                print(f"[{task_id}] Translating risk.{key}...")
                fs["risk_debate"][key] = _translate_text(val)
    
    result["final_state"] = fs
    return result


def _run_task(task_id: str, ticker: str, llm_provider: str, trade_date: str, agents: str):
    try:
        task = analysis_results.get(task_id, {})
        task["provider"] = llm_provider
        task["agents"] = agents
        
        from tradingagents.graph.trading_graph import TradingAgentsGraph
        from tradingagents.default_config import DEFAULT_CONFIG
        config = DEFAULT_CONFIG.copy()
        config["llm_provider"] = llm_provider
        models = PROVIDER_MODELS.get(llm_provider, PROVIDER_MODELS["deepseek"])
        config["deep_think_llm"] = models["deep"]
        config["quick_think_llm"] = models["quick"]
        
        ta = TradingAgentsGraph(debug=False, config=config)
        final_state, decision = ta.propagate(ticker, trade_date)
        
        # Guard: propagate() must return dict; log type if not
        if not isinstance(final_state, dict):
            raise TypeError(
                f"ta.propagate() returned {type(final_state).__name__} "
                f"instead of dict: {str(final_state)[:500]}"
            )
        
        # Apply agent filtering
        filtered_state = _filter_by_agents(final_state, agents)
        
        def _safe_get(d, key, default=""):
            v = d.get(key) if isinstance(d, dict) else default
            return v if v is not None else default
        
        def _safe_nested(d, *keys):
            for k in keys:
                if isinstance(d, dict):
                    d = d.get(k)
                else:
                    return ""
            return d if d is not None else ""
        
        # Build the result
        result_data = {
            "action": decision,
            "ticker": ticker,
            "trade_date": trade_date,
            "company": _safe_get(final_state, "company_of_interest", ticker),
            "agents": agents,
            "final_state": {
                "market_report": _safe_get(filtered_state, "market_report"),
                "sentiment_report": _safe_get(filtered_state, "sentiment_report"),
                "news_report": _safe_get(filtered_state, "news_report"),
                "fundamentals_report": _safe_get(filtered_state, "fundamentals_report"),
                "trader_investment_plan": _safe_get(filtered_state, "trader_investment_plan"),
                "investment_plan": _safe_get(filtered_state, "investment_plan"),
                "final_trade_decision": _safe_get(filtered_state, "final_trade_decision"),
                "summary_decision": "",  # Will be generated during processing
                "trading_levels": {},  # Will be extracted during processing
                "current_price": _safe_nested(filtered_state, "instrument_context", "price"),
                "debate": {
                    "bull": _safe_nested(filtered_state, "investment_debate_state", "bull_history"),
                    "bear": _safe_nested(filtered_state, "investment_debate_state", "bear_history"),
                    "judge": _safe_nested(filtered_state, "investment_debate_state", "judge_decision"),
                    "history": _safe_nested(filtered_state, "investment_debate_state", "history"),
                },
                "risk_debate": {
                    "aggressive": _safe_nested(filtered_state, "risk_debate_state", "aggressive_history"),
                    "conservative": _safe_nested(filtered_state, "risk_debate_state", "conservative_history"),
                    "neutral": _safe_nested(filtered_state, "risk_debate_state", "neutral_history"),
                    "judge": _safe_nested(filtered_state, "risk_debate_state", "judge_decision"),
                },
            }
        }
        
        # === TRANSLATION STAGE ===
        analysis_results[task_id]["stage"] = len(STAGES) - 1  # Translation stage
        analysis_results[task_id]["progress"] = 90
        analysis_results[task_id]["status"] = "translating"
        print(f"[{task_id}] Starting Arabic translation...")
        
        try:
            result_data = _translate_results(result_data, task_id)
            print(f"[{task_id}] ✓ Translation complete!")
        except Exception as te:
            print(f"[{task_id}] WARNING: Translation failed: {te}")
            # Continue with untranslated results
        
        # === DONE ===
        analysis_results[task_id] = {
            "status": "done",
            "result": result_data,
            "start_time": task.get("start_time", time.time()),
            "stage": len(STAGES) - 1, "progress": 100,
            "provider": llm_provider,
            "agents": agents,
        }
    except Exception as e:
        import traceback
        error_detail = str(e)
        tb = traceback.format_exc()
        analysis_results[task_id] = {
            "status": "error", 
            "result": f"{error_detail}\n\n{tb}",
            "start_time": task.get("start_time", time.time()) if task else time.time(),
            "stage": 0, "progress": 0,
        }

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run(app, host="0.0.0.0", port=port)
