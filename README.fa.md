# multi-agent-repo-reviewer (راهنمای کوتاه فارسی)

یک ابزار آموزشی و خودمیزبان برای بازبینی **فقط‌خواندنیِ** مخزن‌های پایتون با چند عامل (agent) تخصصی:
بازبین کد، تحلیلگر وابستگی‌ها و بازبین مستندات. هماهنگی عامل‌ها با **LangGraph** انجام می‌شود،
همهٔ فراخوانی‌های واقعی مدل از **LiteLLM Proxy** عبور می‌کنند، مدل محلی پیش‌فرض **Ollama** است و
**DeepSeek** فقط به‌صورت اختیاری و با انتخاب صریح استفاده می‌شود.

> توضیح کامل، معماری، مدل امنیتی و عیب‌یابی در [README اصلی (انگلیسی)](README.md) آمده است.

## نصب و اجرای سریع (حالت MOCK)

پیش‌نیاز: پایتون ۳٫۱۲ یا بالاتر و ابزار [uv](https://docs.astral.sh/uv/). برای حالت MOCK به Docker،
کلید API یا اینترنت نیاز نیست.

```bash
git clone <آدرس-مخزن-شما> multi-agent-repo-reviewer
cd multi-agent-repo-reviewer
uv sync
uv run python -m repo_reviewer review ./sample_project --mode mock
```

فقط بازبینی مستندات:

```bash
uv run python -m repo_reviewer review ./sample_project --agents documentation --mode mock
```

> حالت MOCK مدل زبانی واقعی نیست؛ یک شبیه‌ساز قطعی است که همان گراف، ابزارها و گزارش‌گیری را اجرا می‌کند.

## اجرای محلی با Ollama

۱. Ollama را از <https://ollama.com/download> نصب و اجرا کنید.
۲. مدلی را دانلود کنید که از فراخوانی ابزار (tool calling) پشتیبانی می‌کند و قابلیت `tools` را بررسی کنید:

```bash
ollama pull qwen3:8b
ollama show qwen3:8b
```

۳. فایل `.env.example` را به `.env` کپی کنید و مقدار `LITELLM_MASTER_KEY` را با کلید دلخواه خودتان پر کنید
(مقدار پیش‌فرض وجود ندارد).
۴. دروازهٔ LiteLLM را اجرا کنید:

```bash
docker compose up -d
```

۵. آزمون دود و سپس بازبینی:

```bash
uv run python -m repo_reviewer smoke --mode local
uv run python -m repo_reviewer review ./sample_project --mode local
```

نکته: در لینوکس، Ollama باید از داخل کانتینر در دسترس باشد؛ جزئیات در بخش «Networking notes» در README اصلی آمده است.

## DeepSeek (اختیاری)

با `--mode deepseek` بخش‌هایی از **کد منبع شما به یک سرویس خارجی ارسال می‌شود**. قبل از ارسال هشدار و
درخواست تأیید نمایش داده می‌شود (`--yes` برای رد کردن تأیید). کلید را فقط از طریق متغیر محیطی
`DEEPSEEK_API_KEY` بدهید و هرگز در مخزن ذخیره نکنید.

## نکتهٔ صادقانه

یافته‌های مدل‌ها «سرنخ» هستند، نه حقیقت قطعی. «تأییدشده» یعنی فایل، شماره خط و متن نقل‌شده واقعاً وجود دارند،
نه اینکه نتیجه‌گیری مدل درست است. مدل‌های کوچک ممکن است ابزار را درست صدا نزنند؛ ابتدا `smoke` را اجرا کنید.

مجوز: MIT
