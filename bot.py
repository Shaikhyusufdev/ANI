import os
import re
import time
import random
import asyncio
import html
import threading
from collections import deque

import httpx
from flask import Flask
from bson import ObjectId
from motor.motor_asyncio import AsyncIOMotorClient
from telegram import Update, InlineKeyboardButton as Btn, InlineKeyboardMarkup as Kb
from telegram.constants import ChatAction, ParseMode
from telegram.ext import (
    Application, CommandHandler, CallbackQueryHandler, MessageHandler,
    ContextTypes, filters,
)

BOT_TOKEN = os.environ["BOT_TOKEN"]
MONGO_URI = os.environ["MONGO_URI"]
GROUP_ID = int(os.environ["GROUP_ID"])
SUPER_ADMIN_ID = int(os.environ["SUPER_ADMIN_ID"])
EXTRA_ADMINS = {int(x) for x in os.environ.get("ADMIN_IDS", "").split(",") if x.strip()}
BOT_NAME = os.environ.get("BOT_NAME", "Ani")
MAX_TOKENS = 120
HIST_LEN = 14

HTML = ParseMode.HTML
db = AsyncIOMotorClient(MONGO_URI)["anibot"]
apis_c, cfg_c, hist_c = db["apis"], db["cfg"], db["history"]

# preset -> (type, base_url)
PRESETS = {
    "openai": ("openai", "https://api.openai.com/v1"),
    "groq": ("openai", "https://api.groq.com/openai/v1"),
    "openrouter": ("openai", "https://openrouter.ai/api/v1"),
    "deepseek": ("openai", "https://api.deepseek.com/v1"),
    "mistral": ("openai", "https://api.mistral.ai/v1"),
    "together": ("openai", "https://api.together.xyz/v1"),
    "gemini": ("openai", "https://generativelanguage.googleapis.com/v1beta/openai"),
    "anthropic": ("anthropic", "https://api.anthropic.com"),
}

DEFAULT_STYLE = """Tum Telegram group me ek real ladki-dost ki tarah baat karti ho, bot ki tarah nahi.
Rules:
- Sirf Hinglish (Hindi + English, Roman script).
- Short natural reply: 1-2 lines, kabhi kabhi 3. Casual slang, 1-3 emojis har reply me.
- Context dekh ke reply do. "hi/hello" pe bhi har baar alag jawab do, same line repeat mat karo.
- Kabhi kabhi shararati/naughty, tease ya sass, par flirty/romantic mat bano.
- Koi sexy ya gandi baat kare to explicit mat bano aur baat aage mat badhao. Hasi-mazaak me witty tareeke se tal do ("arre arre, control karo 😂") ya topic badal do.
- Gaali mat do. Koi insult kare to halke sass me jawab do.
- Bot/AI hone ka zikr mat karo jab tak koi seriously na puche. Puche to honestly bata do.
- Sirf reply likho, naam ya prefix mat lagao."""

MOODS = [
    "sweet aur caring", "thodi shararati aur naughty", "sassy aur witty",
    "chill aur funny", "excited aur energetic", "naughty aur teasing",
]

CFG = {"enabled": True, "chance": 0.25, "style": DEFAULT_STYLE}
APIS: list[dict] = []
COOL: dict[str, float] = {}
LAST_ERR: dict[str, str] = {}
HIST: deque = deque(maxlen=HIST_LEN)
ALERT_AT = 0.0
http: httpx.AsyncClient | None = None

GREET = re.compile(r"^\s*(hi+|hey+|hello+|helo+|hlo|yo|sup|namaste|gm|gn|good\s*(morning|night|evening))\b", re.I)


# ---------- Helpers ----------
def is_staff(uid: int) -> bool:
    return uid == SUPER_ADMIN_ID or uid in EXTRA_ADMINS


def esc(s) -> str:
    return html.escape(str(s))


def mask(k: str) -> str:
    return k[:6] + "…" + k[-4:] if len(k) > 12 else "***"


async def reload_apis():
    global APIS
    APIS = await apis_c.find().sort("prio", 1).to_list(100)


def get_api(aid: str):
    return next((a for a in APIS if str(a["_id"]) == aid), None)


async def load_cfg():
    doc = await cfg_c.find_one({"_id": "cfg"}) or {}
    doc.pop("_id", None)
    CFG.update(doc)


async def save_cfg(**kw):
    CFG.update(kw)
    await cfg_c.update_one({"_id": "cfg"}, {"$set": kw}, upsert=True)


async def _save_hist(line: str):
    try:
        await hist_c.update_one(
            {"_id": GROUP_ID},
            {"$push": {"m": {"$each": [line], "$slice": -HIST_LEN}}},
            upsert=True,
        )
    except Exception as e:
        print("hist save error:", repr(e))


def remember(line: str):
    HIST.append(line)
    asyncio.create_task(_save_hist(line))


def build_system() -> str:
    return f"Tumhara naam {BOT_NAME} hai.\n{CFG['style']}\nAbhi tumhara mood: {random.choice(MOODS)}."


# ---------- LLM layer (multi-API + failover) ----------
async def call_api(a: dict, system: str, prompt: str) -> str:
    if a["type"] == "anthropic":
        r = await http.post(
            f"{a['base']}/v1/messages",
            headers={"x-api-key": a["key"], "anthropic-version": "2023-06-01"},
            json={"model": a["model"], "max_tokens": MAX_TOKENS, "temperature": 1.0,
                  "system": system, "messages": [{"role": "user", "content": prompt}]},
        )
        r.raise_for_status()
        d = r.json()
        return "".join(b.get("text", "") for b in d["content"] if b.get("type") == "text").strip()
    r = await http.post(
        f"{a['base']}/chat/completions",
        headers={"Authorization": f"Bearer {a['key']}"},
        json={"model": a["model"], "max_tokens": MAX_TOKENS, "temperature": 1.0,
              "messages": [{"role": "system", "content": system},
                           {"role": "user", "content": prompt}]},
    )
    r.raise_for_status()
    return (r.json()["choices"][0]["message"]["content"] or "").strip()


async def ask_llm(prompt: str, bot) -> str | None:
    global ALERT_AT
    system = build_system()
    now = time.time()
    for a in APIS:
        aid = str(a["_id"])
        if not a.get("enabled", True) or COOL.get(aid, 0) > now:
            continue
        try:
            t = await call_api(a, system, prompt)
            if t:
                return t
        except httpx.HTTPStatusError as e:
            code = e.response.status_code
            body = e.response.text.lower()
            if code in (401, 402, 403) or any(w in body for w in ("quota", "insufficient", "billing", "exhausted")):
                COOL[aid] = now + 3600
            elif code == 429:
                COOL[aid] = now + 60
            else:
                COOL[aid] = now + 30
            LAST_ERR[aid] = f"HTTP {code} {e.response.text[:80]}"
        except Exception as e:
            COOL[aid] = now + 5
            LAST_ERR[aid] = type(e).__name__
    # sab fail: super admin ko 10 min me max 1 alert
    if time.time() - ALERT_AT > 600:
        ALERT_AT = time.time()
        try:
            await bot.send_message(SUPER_ADMIN_ID, "⚠️ Saare APIs fail/cooldown me hain. /admin se check karo.")
        except Exception:
            pass
    return None


async def test_api(a: dict):
    try:
        t = await call_api(a, "Reply in one short line.", "Say hi in Hinglish with an emoji")
        return True, t
    except httpx.HTTPStatusError as e:
        return False, f"{e.response.status_code}: {e.response.text[:150]}"
    except Exception as e:
        return False, repr(e)[:150]


# ---------- Group chat ----------
async def group_msg(update: Update, context: ContextTypes.DEFAULT_TYPE):
    msg = update.message
    if not msg or not msg.text or not msg.from_user or msg.from_user.is_bot:
        return
    if update.effective_chat.id != GROUP_ID:
        return

    bot = context.bot
    text = msg.text.strip()
    line = f"{msg.from_user.first_name or 'User'}: {text}"
    low = text.lower()

    if not CFG["enabled"]:
        remember(line)
        return

    rt = msg.reply_to_message
    to_me = bool(rt and rt.from_user and rt.from_user.id == bot.id)
    called = BOT_NAME.lower() in low or (bot.username and f"@{bot.username.lower()}" in low)

    if rt and not to_me and not called:
        remember(line)  # kisi aur se baat ho rahi hai, beech me nahi kudna
        return

    greet = len(text) < 25 and bool(GREET.match(text)) and random.random() < 0.8
    if not (to_me or called or greet or random.random() < CFG["chance"]):
        remember(line)
        return

    asyncio.create_task(bot.send_chat_action(GROUP_ID, ChatAction.TYPING))
    convo = "\n".join(HIST)
    prompt = f"Group ki recent baatcheet:\n{convo}\n\nAb is latest msg ka reply do:\n{line}"
    reply = await ask_llm(prompt, bot)
    remember(line)
    if not reply:
        return
    if reply.lower().startswith(f"{BOT_NAME.lower()}:"):
        reply = reply.split(":", 1)[1].strip()
    reply = reply[:500]
    remember(f"{BOT_NAME}: {reply}")
    await msg.reply_text(reply)


# ---------- Private: sirf admin panel ----------
def main_kb() -> Kb:
    return Kb([
        [Btn("📡 APIs", callback_data="m:apis"), Btn("➕ Add API", callback_data="m:add")],
        [Btn("📢 Broadcast", callback_data="m:bc"), Btn("🎭 Style", callback_data="m:style")],
        [Btn("⚙️ Settings", callback_data="m:set")],
    ])


BACK = Kb([[Btn("🔙 Back", callback_data="m:home")]])


async def safe_edit(q, text: str, kb=None):
    try:
        await q.edit_message_text(text, parse_mode=HTML, reply_markup=kb)
    except Exception:
        pass


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if is_staff(update.effective_user.id):
        await update.message.reply_text("🛠 <b>Ani Admin Panel</b>", parse_mode=HTML, reply_markup=main_kb())
    else:
        await update.message.reply_text("Ye bot sirf group me chalta hai 🙂")


async def admin_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not is_staff(update.effective_user.id):
        return
    context.user_data.pop("st", None)
    await update.message.reply_text("🛠 <b>Ani Admin Panel</b>", parse_mode=HTML, reply_markup=main_kb())


async def cancel(update: Update, context: ContextTypes.DEFAULT_TYPE):
    context.user_data.pop("st", None)
    if is_staff(update.effective_user.id):
        await update.message.reply_text("Cancel ho gaya. /admin se panel kholo.")


def api_status(a: dict) -> str:
    aid = str(a["_id"])
    if not a.get("enabled", True):
        return "OFF ⛔"
    left = COOL.get(aid, 0) - time.time()
    if left > 0:
        return f"cooldown {int(left)}s ({LAST_ERR.get(aid, '?')}) ⏳"
    return "OK ✅"


async def show_apis(q):
    await reload_apis()
    if not APIS:
        await safe_edit(q, "📡 <b>APIs</b>\n\nKoi API nahi hai. ➕ Add API se jodo.", BACK)
        return
    text = "📡 <b>APIs</b> (upar wala pehle use hoga)\n\n"
    kb = []
    for i, a in enumerate(APIS, 1):
        text += f"{i}. <b>{esc(a['label'])}</b> — {api_status(a)}\n"
        kb.append([Btn(f"{i}. {a['label']}", callback_data=f"a:v:{a['_id']}")])
    kb.append([Btn("🔙 Back", callback_data="m:home")])
    await safe_edit(q, text, Kb(kb))


async def show_api(q, a: dict):
    aid = str(a["_id"])
    text = (
        f"📡 <b>{esc(a['label'])}</b>\n"
        f"Type: <code>{a['type']}</code>\nModel: <code>{esc(a['model'])}</code>\n"
        f"Key: <code>{esc(mask(a['key']))}</code>\nStatus: {api_status(a)}"
    )
    kb = Kb([
        [Btn("🔕 Turn OFF" if a.get("enabled", True) else "🔔 Turn ON", callback_data=f"a:t:{aid}"),
         Btn("🧪 Test", callback_data=f"a:x:{aid}")],
        [Btn("⬆️ Priority up", callback_data=f"a:u:{aid}"),
         Btn("🗑 Delete", callback_data=f"a:d:{aid}")],
        [Btn("🔙 Back", callback_data="m:apis")],
    ])
    await safe_edit(q, text, kb)


async def show_settings(q):
    kb = Kb([
        [Btn("🔕 Bot OFF" if CFG["enabled"] else "🔔 Bot ON", callback_data="s:on")],
        [Btn(f"{'• ' if abs(CFG['chance'] - p / 100) < 0.01 else ''}{p}%", callback_data=f"s:c:{p}")
         for p in (10, 25, 40, 60)],
        [Btn("🔙 Back", callback_data="m:home")],
    ])
    await safe_edit(
        q,
        f"⚙️ <b>Settings</b>\n\nBot: {'ON ✅' if CFG['enabled'] else 'OFF ❌'}\n"
        f"Random reply chance: <b>{int(CFG['chance'] * 100)}%</b>\n"
        f"(reply-to-bot, naam lene, hi/hello pe hamesha reply)",
        kb,
    )


async def panel_cb(update: Update, context: ContextTypes.DEFAULT_TYPE):
    q = update.callback_query
    await q.answer()
    if not is_staff(q.from_user.id):
        return
    d = q.data.split(":")
    k = d[0] + ":" + d[1]

    if k == "m:home":
        context.user_data.pop("st", None)
        await safe_edit(q, "🛠 <b>Ani Admin Panel</b>", main_kb())

    elif k == "m:apis":
        await show_apis(q)

    elif k == "m:add":
        context.user_data["st"] = "add"
        presets = ", ".join(PRESETS)
        await safe_edit(
            q,
            "➕ <b>Add API</b>\n\nEk msg me bhejo:\n<code>preset model key</code>\n\n"
            "Example:\n<code>groq llama-3.3-70b-versatile gsk_xxxxx</code>\n\n"
            f"Presets: {presets}\n"
            "Custom (OpenAI-compatible): <code>custom model key base_url</code>\n"
            "Model ka naam provider ki site se check kar lena.\n"
            "Key wala msg main delete kar dunga.\nCancel: /cancel",
            BACK,
        )

    elif k == "m:bc":
        context.user_data["st"] = "bc"
        await safe_edit(q, "📢 <b>Broadcast</b>\n\nJo bhejna hai group me wo yahan bhejo "
                           "(text/photo/video sab chalega).\nCancel: /cancel", BACK)

    elif k == "m:style":
        context.user_data["st"] = None
        kb = Kb([[Btn("✏️ Set Style", callback_data="y:set"), Btn("🔄 Reset", callback_data="y:reset")],
                 [Btn("🔙 Back", callback_data="m:home")]])
        await safe_edit(q, f"🎭 <b>Style / Persona</b>\n\n<i>Current:</i>\n{esc(CFG['style'])[:3000]}", kb)

    elif k == "y:set":
        context.user_data["st"] = "style"
        await safe_edit(q, "✏️ Naya style/persona prompt bhejo (Hinglish rules, mood, limits sab isme likho).\n"
                           "Naam apne aap add ho jayega.\nCancel: /cancel", BACK)

    elif k == "y:reset":
        await save_cfg(style=DEFAULT_STYLE)
        await safe_edit(q, "✅ Style reset ho gaya.", BACK)

    elif k == "m:set":
        await show_settings(q)

    elif k == "s:on":
        await save_cfg(enabled=not CFG["enabled"])
        await show_settings(q)

    elif k == "s:c":
        await save_cfg(chance=int(d[2]) / 100)
        await show_settings(q)

    elif d[0] == "a":
        await reload_apis()
        a = get_api(d[2])
        if not a:
            await show_apis(q)
            return
        aid = d[2]
        if d[1] == "v":
            await show_api(q, a)
        elif d[1] == "t":
            await apis_c.update_one({"_id": a["_id"]}, {"$set": {"enabled": not a.get("enabled", True)}})
            COOL.pop(aid, None)
            await reload_apis()
            await show_api(q, get_api(aid))
        elif d[1] == "x":
            await safe_edit(q, "🧪 Testing…")
            ok, out = await test_api(a)
            if ok:
                COOL.pop(aid, None)
            await safe_edit(q, f"{'✅ Chal raha hai' if ok else '❌ Fail'}\n\n<code>{esc(out)}</code>",
                            Kb([[Btn("🔙 Back", callback_data=f"a:v:{aid}")]]))
        elif d[1] == "u":
            ids = [str(x["_id"]) for x in APIS]
            i = ids.index(aid)
            if i > 0:
                b = APIS[i - 1]
                await apis_c.update_one({"_id": a["_id"]}, {"$set": {"prio": b["prio"]}})
                await apis_c.update_one({"_id": b["_id"]}, {"$set": {"prio": a["prio"]}})
            await show_apis(q)
            await reload_apis()
        elif d[1] == "d":
            await apis_c.delete_one({"_id": a["_id"]})
            COOL.pop(aid, None)
            await show_apis(q)


async def private_in(update: Update, context: ContextTypes.DEFAULT_TYPE):
    msg = update.message
    uid = update.effective_user.id
    if not is_staff(uid):
        return  # normal user ko kuch nahi, token bachao
    st = context.user_data.get("st")
    if not st:
        return

    if st == "bc":
        try:
            await msg.copy(GROUP_ID)
            await msg.reply_text("✅ Group me bhej diya.", reply_markup=main_kb())
        except Exception as e:
            await msg.reply_text(f"❌ Nahi gaya: {e}")
        context.user_data.pop("st", None)

    elif st == "style" and msg.text:
        await save_cfg(style=msg.text.strip())
        await msg.reply_text("✅ Style save ho gaya.", reply_markup=main_kb())
        context.user_data.pop("st", None)

    elif st == "add" and msg.text:
        parts = msg.text.split()
        try:
            await msg.delete()  # key chat me na dikhe
        except Exception:
            pass
        if len(parts) < 3:
            await context.bot.send_message(uid, "Format: preset model key (ya cancel: /cancel)")
            return
        preset, model, key = parts[0].lower(), parts[1], parts[2]
        if preset == "custom":
            if len(parts) < 4:
                await context.bot.send_message(uid, "Custom me base_url bhi do: custom model key base_url")
                return
            typ, base = "openai", parts[3].rstrip("/")
        elif preset in PRESETS:
            typ, base = PRESETS[preset]
        else:
            await context.bot.send_message(uid, f"Preset galat. Ye use karo: {', '.join(PRESETS)}, custom")
            return
        last = await apis_c.find_one(sort=[("prio", -1)])
        await apis_c.insert_one({
            "label": f"{preset}/{model[:20]}", "type": typ, "base": base, "model": model,
            "key": key, "enabled": True, "prio": (last["prio"] + 1) if last else 1,
        })
        await reload_apis()
        await context.bot.send_message(uid, "✅ API add ho gayi. 🧪 Test se check kar lo.", reply_markup=main_kb())
        context.user_data.pop("st", None)


# ---------- Keep-alive web server ----------
web = Flask(__name__)


@web.route("/")
def home():
    return "Ani is alive"


def run_web():
    web.run(host="0.0.0.0", port=int(os.environ.get("PORT", 10000)))


async def post_init(app: Application):
    global http
    http = httpx.AsyncClient(timeout=httpx.Timeout(25.0, connect=5.0))
    await load_cfg()
    await reload_apis()
    doc = await hist_c.find_one({"_id": GROUP_ID})
    if doc:
        HIST.extend(doc.get("m", []))


def main():
    threading.Thread(target=run_web, daemon=True).start()
    app = Application.builder().token(BOT_TOKEN).concurrent_updates(True).post_init(post_init).build()
    app.add_handler(CommandHandler("start", start, filters=filters.ChatType.PRIVATE))
    app.add_handler(CommandHandler("admin", admin_cmd, filters=filters.ChatType.PRIVATE))
    app.add_handler(CommandHandler("cancel", cancel, filters=filters.ChatType.PRIVATE))
    app.add_handler(CallbackQueryHandler(panel_cb, pattern=r"^(m|a|s|y):"))
    app.add_handler(MessageHandler(filters.ChatType.PRIVATE & ~filters.COMMAND, private_in))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND & filters.ChatType.GROUPS, group_msg))
    app.run_polling()


if __name__ == "__main__":
    main()
