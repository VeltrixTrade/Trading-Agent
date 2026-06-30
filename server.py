import os, asyncio, uuid, time
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
    elif data.get("status") == "running":
        data["elapsed_seconds"] = round(time.time() - data.get("start_time", time.time()), 1)
        elapsed = data["elapsed_seconds"]
        est_total = {"deepseek": 120, "groq": 180, "google": 150, "openai": 200, "ollama": 1800}
        est = est_total.get(data.get("provider", "deepseek"), 180)
        data["progress"] = min(95, int((elapsed / est) * 100))
        stg = min(len(STAGES) - 1, int((elapsed / est) * len(STAGES)))
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
        
        analysis_results[task_id] = {
            "status": "done",
            "result": {
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
            },
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
