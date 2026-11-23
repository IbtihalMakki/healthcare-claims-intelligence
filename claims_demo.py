import argparse
import webbrowser
from html import escape
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import parse_qs, urlsplit

import joblib

from claims_rules import review_claim
from common import ROOT, read_csv, read_json, read_jsonl
from intent_model import MODEL_PATH as INTENT_PATH
from ner_model import MODEL_PATH as NER_PATH, extract

REASONS = {
    "missing_documentation": "مستندات مطلوبة غير مرفقة",
    "missing_prior_authorization": "الموافقة المسبقة ناقصة",
}
DOCUMENTS = {"referral": "الإحالة", "clinical_note": "الملاحظة السريرية", "physician_order": "طلب الطبيب"}
ENTITY_TYPES = {"DIAGNOSIS": "تشخيص", "PROCEDURE": "إجراء"}
ASSERTIONS = {
    "present": "قائم",
    "historical": "سابق",
    "negated": "منفي",
    "suspected": "مشتبه به",
    "ordered": "مطلوب",
}
INTENTS = {
    "appeal_request": "طلب اعتراض",
    "billing_question": "استفسار عن الفاتورة",
    "claim_status": "استفسار عن حالة المطالبة",
    "coverage_inquiry": "استفسار عن التغطية",
    "denial_reason": "استفسار عن سبب الرفض",
    "document_requirement": "استفسار عن المستندات",
    "eligibility_check": "استفسار عن سريان التأمين",
    "prior_authorization": "استفسار عن الموافقة المسبقة",
}


def highlight_note(text, entities):
    # نستخدم إزاحات الموديل على النص الأصلي حتى يظهر كل كيان في مكانه.
    parts = []
    end = 0
    for entity in sorted(entities, key=lambda item: item["start"]):
        start, stop = entity["start"], entity["end"]
        parts.append(escape(text[end:start]))
        css = "diagnosis" if entity["label"] == "DIAGNOSIS" else "procedure"
        parts.append(f'<mark class="{css}">{escape(text[start:stop])}</mark>')
        end = stop
    parts.append(escape(text[end:]))
    return "".join(parts)


def render(claim, note, entities, review, intent):
    status = "تحتاج مراجعة" if review["review_required"] else "لا توجد نواقص حسب السياسة التجريبية"
    tone = "warning" if review["review_required"] else "clear"
    reasons = "".join(f"<li>{escape(REASONS.get(code, code))}</li>" for code in review["review_reasons"])
    docs = "، ".join(DOCUMENTS.get(code, code) for code in review["missing_documents"]) or "لا يوجد"
    rows = "".join(
        f'<tr><td dir="auto">{escape(item["text"])}</td>'
        f'<td>{escape(ENTITY_TYPES.get(item["label"], item["label"]))}</td>'
        f'<td>{escape(ASSERTIONS.get(item["assertion"], item["assertion"]))}</td>'
        f'<td><bdi>{item["start"]}–{item["end"]}</bdi></td></tr>'
        for item in entities
    )
    empty_row = '<tr><td colspan="4">لم يستخرج النموذج كيانات من هذه الملاحظة.</td></tr>'
    note_text = highlight_note(note["text"], entities)
    status_detail = f"<ul>{reasons}</ul>" if reasons else ""
    inquiry = ""
    if intent:
        label = escape(INTENTS.get(intent["intent"], intent["intent"]))
        inquiry = (
            f'<section><h2>تصنيف الاستفسار</h2><p dir="auto">{escape(intent["text"])}</p>'
            f'<p>{label} <code dir="ltr">{escape(intent["intent"])}</code></p></section>'
        )
    return f"""
<div class="grid">
  <section>
    <h2>الملاحظة السريرية</h2>
    <p class="note" dir="auto">{note_text}</p>
    <table>
      <thead><tr><th>الكيان</th><th>النوع</th><th>الحالة</th><th>الإزاحات</th></tr></thead>
      <tbody>{rows or empty_row}</tbody>
    </table>
  </section>
  <section>
    <h2>نتيجة محرك القواعد</h2>
    <p><strong class="badge {tone}">{status}</strong></p>
    {status_detail}
    <p><strong>المستندات الناقصة</strong> · {escape(docs)}</p>
  </section>
</div>
{inquiry}
"""

def page(content, claim=None, controls=""):
    heading = "مراجعة المطالبات"
    caption = "بيانات اصطناعية"
    if claim:
        heading = f'مراجعة المطالبة <bdi>{escape(claim["claim_id"])}</bdi>'
        caption += f' · سياسة <bdi>{escape(claim["policy_id"])}</bdi>'
    return f"""<!doctype html>
<html lang="ar" dir="rtl">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>ديمو مراجعة المطالبات</title>
<style>
* {{ box-sizing: border-box }}
body {{ margin:0; background:#f0f5f6; color:#102b35; font:16px/1.65 Tahoma,Arial,sans-serif }}
main {{ width:min(1100px,100% - 32px); margin:36px auto }}
header {{ background:#123942; color:white; padding:28px; border-radius:18px; margin-bottom:18px }}
h1,h2,p {{ margin:0 0 10px }}
header p {{ margin:0; color:#d1e4e7 }}
h2 {{ font-size:20px }}
section {{ background:white; border:1px solid #d9e6e9; border-radius:16px; padding:20px; margin-bottom:16px; min-width:0 }}
.grid {{ display:grid; grid-template-columns:1.75fr 1fr; gap:16px }}
.note {{ background:#f3f7f8; border-radius:10px; padding:14px; line-height:2; white-space:pre-wrap }}
mark {{ border-radius:5px; padding:2px 4px; color:inherit }}
.diagnosis {{ background:#eee4fb }}
.procedure {{ background:#cceee9 }}
.badge {{ display:inline-block; border-radius:999px; padding:4px 12px }}
.warning {{ background:#fff0cf; color:#734607 }}
.clear {{ background:#d7f3e9; color:#164e41 }}
table {{ width:100%; border-collapse:collapse; font-size:14px }}
th,td {{ padding:9px 7px; border-bottom:1px solid #e4ecee; text-align:start }}
code {{ display:inline-block; background:#edf3f5; border-radius:5px; padding:0 6px }}
@media (max-width:760px) {{ .grid {{ grid-template-columns:1fr; gap:0 }} }}

form {{ display:grid; grid-template-columns: minmax(130px, 0.65fr) minmax(180px, 2fr) auto; gap:12px; align-items:end }}
label {{ display:block; font-weight:bold; margin-bottom:4px }}
input,textarea,button {{ font:inherit; border-radius:9px; padding:10px 12px; min-height:46px }}
input,textarea {{ width:100%; border:1px solid #9fb6bd; background:#fbfdfd; color:inherit }}
textarea {{ display:block; resize:vertical }}
button {{ background:#0d726d; border:1px solid #0d726d; color:white; font-weight:bold; cursor:pointer }}
button:hover {{ background:#095c58 }}
:focus-visible {{ outline:3px solid #268f88; outline-offset:2px }}
.error {{ margin:14px 0 0; padding:10px 12px; border-radius:8px; background:#fdefeb; color:#8d2b22 }}
@media (max-width:620px) {{ form {{ grid-template-columns:1fr }} }}
</style>
</head>
<body>
<main>
<header><h1>{heading}</h1><p>{caption}</p></header>
{controls}
{content}
</main>
</body>
</html>
"""


def controls_for(claim_id, question, error=""):
    feedback = f'<p class="error" role="alert">{escape(error)}</p>' if error else ""
    return f"""
<section aria-label="مدخلات الديمو">
  <form method="post" action="/">
    <div>
      <label for="claim-id">رقم المطالبة</label>
      <input id="claim-id" name="claim_id" dir="ltr" value="{escape(claim_id)}"
             placeholder="CLM-00053" maxlength="20" required>
    </div>
    <div>
      <label for="question">الاستفسار (اختياري)</label>
      <textarea id="question" name="question" dir="auto" rows="1" maxlength="500"
                placeholder="خلصتوا مراجعة مطالبة جلسة العلاج الطبيعي؟">{escape(question)}</textarea>
    </div>
    <button type="submit">تشغيل</button>
  </form>
  {feedback}
</section>
"""


def render_report(claim, notes, policies, ner_models, intent_model, question):
    # نحلّل ملاحظة المطالبة والسياسة، ونصنّف الاستفسار عند إدخاله.
    note = notes[claim["note_id"]]
    entities = extract(note["text"], *ner_models)
    review = review_claim(claim, policies[claim["policy_id"]])
    intent = None
    if question:
        intent = {"text": question, "intent": str(intent_model.predict([question])[0])}
    return render(claim, note, entities, review, intent)


def serve(claims, notes, policies, ner_models, intent_model, initial_claim, no_open):
    class Handler(BaseHTTPRequestHandler):
        def send_page(self, document, status=200):
            body = document.encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def show_result(self, claim_id, question):
            claim_id = claim_id.strip().upper()
            question = question.strip()
            claim = claims.get(claim_id)
            error = ""
            if claim is None:
                error = "رقم المطالبة غير موجود في البيانات. جرّبي CLM-00053."
            elif len(question) > 500:
                error = "اكتبي استفسارًا لا يتجاوز 500 حرف."
            elif question and intent_model is None:
                error = "شغّلي intent_model.py لتدريب مصنّف الاستفسارات أولًا."
            if error:
                self.send_page(page("", controls=controls_for(claim_id, question, error)), 400)
                return
            content = render_report(claim, notes, policies, ner_models, intent_model, question)
            self.send_page(page(content, claim, controls_for(claim_id, question)))

        def do_GET(self):
            if urlsplit(self.path).path != "/":
                self.send_error(404)
                return
            self.show_result(initial_claim, "")

        def do_POST(self):
            if urlsplit(self.path).path != "/":
                self.send_error(404)
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= 8192:
                    raise ValueError("Invalid form length.")
                fields = parse_qs(self.rfile.read(length).decode("utf-8"), max_num_fields=8)
            except (UnicodeError, ValueError):
                self.send_error(400, "Invalid form.")
                return
            self.show_result(fields.get("claim_id", [""])[0], fields.get("question", [""])[0])

    with HTTPServer(("127.0.0.1", 0), Handler) as server:
        url = f"http://127.0.0.1:{server.server_port}/"
        print(f"Claims demo: {url}", flush=True)
        if not no_open:
            webbrowser.open(url)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass


def main():
    parser = argparse.ArgumentParser(description="Open the claims review demo in your browser.")
    parser.add_argument("--claim-id", default="CLM-00053")
    parser.add_argument("--question", help="Optionally classify an insurance question.")
    parser.add_argument("--serve", action="store_true", help="Start the interactive demo on this device.")
    parser.add_argument("--no-open", action="store_true", help="Skip automatically opening the browser.")
    args = parser.parse_args()
    claim_id = args.claim_id.strip().upper()
    claims = {row["claim_id"]: row for row in read_csv("claims.csv")}
    claim = claims.get(claim_id)
    if claim is None:
        parser.error(f"Claim not found: {claim_id}")
    if not NER_PATH.is_file():
        parser.error("Run ner_model.py to train the NER model first.")
    if args.question is not None and (not args.question.strip() or not INTENT_PATH.is_file()):
        parser.error("Use a non-empty question and train the intent model with intent_model.py first.")

    notes = {row["note_id"]: row for row in read_jsonl("clinical_ner.jsonl")}
    policies = {row["policy_id"]: row for row in read_json("policies.json")}
    ner_models = joblib.load(NER_PATH)
    intent_model = joblib.load(INTENT_PATH) if INTENT_PATH.is_file() and (args.serve or args.question) else None
    if args.serve:
        serve(claims, notes, policies, ner_models, intent_model, claim_id, args.no_open)
        return

    content = render_report(claim, notes, policies, ner_models, intent_model, args.question)

    output = ROOT / "artifacts" / f"{claim['claim_id']}_demo.html"
    output.parent.mkdir(exist_ok=True)
    output.write_text(page(content, claim), encoding="utf-8")
    print(f"Demo saved: {output}")
    if not args.no_open:
        webbrowser.open(output.as_uri())


if __name__ == "__main__":
    main()
