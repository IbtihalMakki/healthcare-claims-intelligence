import joblib  # يحمّل الأوزان الناتجة من التدريب المحلي.
from sklearn.metrics import f1_score  # يحسب Macro-F1 لتصنيف النوايا.

from intent_model import MODEL_PATH as INTENT_PATH  # يحدد ملف نموذج النوايا.
from ner_model import MODEL_PATH as NER_PATH, evaluate, extract  # يعيد استخدام تقييم NER والاستدلال المعتاد.
from training_data import challenge_intents, challenge_notes  # يحمّل الأمثلة الجديدة المخصصة للتقييم.


def intent_metrics(intent_model, messages):  # يحسب نتيجة النوايا لمجموعة رسائل واحدة.
    predictions = intent_model.predict([row["text"] for row in messages])  # يتنبأ بالنوايا من الرسائل.
    gold = [row["intent"] for row in messages]  # يقرأ النوايا المرجعية للتقييم فقط.
    correct = sum(expected == actual for expected, actual in zip(gold, predictions))  # يعد الرسائل المصنفة بشكل صحيح.
    macro_f1 = f1_score(gold, predictions, labels=sorted(set(gold)), average="macro", zero_division=0)  # يعطي النوايا الثماني وزنًا متساويًا.
    return macro_f1, correct, len(gold)  # يعيد الدرجة وعدد الإجابات الصحيحة وحجم المجموعة.


def score(intent_model, extract_entities):  # يقيّم أي نسخة من النموذجين على الأمثلة نفسها.
    intents, notes = list(challenge_intents()), list(challenge_notes())  # يحمّل الاستفسارات والملاحظات المرجعية.
    for language in ("ar", "en"):  # يعرض نتائج كل لغة منفصلة.
        messages = [row for row in intents if row["language"] == language]  # يحدد رسائل اللغة الحالية.
        macro_f1, correct, total = intent_metrics(intent_model, messages)  # يقيس تصنيف رسائل اللغة الحالية.
        selected_notes = [row for row in notes if row["language"] == language]  # يحدد ملاحظات اللغة الحالية.
        pairs = [(note, extract_entities(note["text"])) for note in selected_notes]  # يستخرج الكيانات اعتمادًا على النص فقط.
        print(f"{language}: Intent Macro-F1={macro_f1:.3f} ({correct}/{total} correct)")  # يطبع نتيجة تصنيف النوايا وعدد الإجابات الصحيحة.
        print(f"{language}: NER {evaluate(pairs)} | NER + assertion {evaluate(pairs, joint=True)} | {len(pairs)} notes")  # يطبع نتيجة الحدود والحالة مع حجم الاختبار.


if __name__ == "__main__":  # يشغّل التقييم الإضافي عند استدعاء الملف مباشرة.
    if not INTENT_PATH.is_file() or not NER_PATH.is_file():  # يتأكد أن المستخدم شغّل تدريبي النوايا وNER أولًا.
        raise SystemExit("Train both models before running evaluate_challenge.py.")  # يوضح المتطلب قبل الاستدلال.
    intent_model = joblib.load(INTENT_PATH)  # يقرأ مصنف النوايا المحفوظ.
    ner_models = joblib.load(NER_PATH)  # يقرأ المصنفين وقاموس تصحيح الحدود.
    score(intent_model, lambda text: extract(text, *ner_models))  # يقيّم النموذجين على الأمثلة الجديدة.
    probe = list(challenge_intents("intent_probe.txt"))  # يقرأ مجموعة الاختبار العربية الصغيرة.
    macro_f1, correct, total = intent_metrics(intent_model, probe)  # يقيّم الأمثلة الإضافية بمعزل عن التدريب.
    print(f"ar probe: Intent Macro-F1={macro_f1:.3f} ({correct}/{total} correct)")  # يطبع نتيجة المجموعة الجديدة.
