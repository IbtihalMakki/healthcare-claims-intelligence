import argparse  # يقرأ خيار النص من الطرفية.
import json  # يطبع النتائج بصيغة JSON.
import re  # يقطّع النص ويطابق العبارات.
from functools import lru_cache  # يعيد استخدام نمط مطابقة الباء بين الملاحظات.

import joblib  # يحفظ النموذجين المدربين ويحمّلهما.
from sklearn.feature_extraction import DictVectorizer  # يحوّل سمات التوكنات إلى مصفوفة.
from sklearn.feature_extraction.text import TfidfVectorizer  # يحوّل سياق التشخيص إلى أوزان محارف.
from sklearn.linear_model import LogisticRegression  # ينشئ مصنّف الكيانات ومصنّف الحالة.
from sklearn.pipeline import make_pipeline  # يربط تجهيز المدخلات بالمصنّف.

from common import MODELS, read_jsonl  # يستورد مسار الحفظ وقارئ الملاحظات.
from training_data import extra_notes  # يضيف صيغًا تدريبية لحالات التشخيص.

MODEL_PATH = MODELS / "ner_v2.joblib"  # يحدد اسم ملف الموديل المدرّب.
TOKEN = re.compile(r"[بلكوف]ـ|\w+|[^\w\s]")  # يفصل الحروف المطوّلة ويلتقط الكلمات والرموز.

ARABIC_CUES = {  # يربط كل حالة بعبارات عربية تدل عليها.
    "negated": r"استبعاد|استبعد(?:وا)?|(?:ما|لم) (?:ثبت|يثبت|يدعم|تثبت)|ما (?:عنده|عندها|لقينا)|لا (?:يوجد|توجد) (?:دليل|دلائل)|غير (?:مثبت|مدعوم)|ينفي|لا يتوافق",  # قرائن النفي.
    "suspected": r"اشتباه|يشتبه|يشك|شاك|(?:ال)?احتمال(?:ات)?|محتمل(?:ه|ة)?|ما تاكد|غير مؤكد",  # قرائن الاشتباه.
    "historical": r"سابقا|تاريخ مرضي|تاريخ سابق|السجل القديم|ملف(?:ه)? القديم|زيارة سابقة|سبق تشخيص|من سنوات|في الماضي",  # قرائن التاريخ المرضي.
    "present": r"التشخيص الحالي|التشخيص المثبت|تشخيص مؤكد|حالة قائمة|ثبت(?: التشخيص)?",  # قرائن التشخيص الحالي.
}  # تنتهي قائمة القرائن، ويرجع الكود إلى المصنّف إذا غابت قرينة واضحة.
ARABIC_DIACRITICS = re.compile(r"[\u064b-\u065f\u0670ـ]")  # يزيل التشكيل والتطويل من سياق تحديد الحالة فقط.


@lru_cache(maxsize=4)  # يحفظ النمط حتى لا يُعاد بناؤه لكل ملاحظة.
def attached_b_pattern(terms):  # يبني نمطًا للباء الملتصقة قبل تشخيص عربي معروف.
    alternatives = "|".join(re.escape(term) for term in terms)  # يحوّل المصطلحات إلى بدائل مطابقة حرفية.
    if not alternatives:  # يتحقق من وجود مصطلحات عربية في القاموس.
        return re.compile(r"(?!)")  # يرجع نمطًا لا يطابق أي نص إذا كان القاموس فارغًا.
    return re.compile(r"(?<!\w)ب(?P<diagnosis>" + alternatives + r")(?!\w)")  # يطابق الباء والتشخيص عند حدود الكلمات.


def tokenize(text, attached_starts=()):  # يقطّع النص ويحافظ على إزاحات المحارف الأصلية.
    starts = set(attached_starts)  # يجمع مواضع الكلمات التي تبدأ بباء قبل تشخيص معروف.
    tokens = []  # يجهّز قائمة التوكنات.
    for match in TOKEN.finditer(text):  # يمر على الكلمات والرموز بترتيبها.
        word, start, end = match.group(), match.start(), match.end()  # يقرأ التوكن وبدايته ونهايته.
        if start in starts:  # يفصل الباء في المواضع التي طابقت القاموس.
            tokens.append(("بـ", start, start + 1))  # يعطي المصنّف شكل الباء المألوف مع حفظ طولها الأصلي.
            tokens.append((word[1:], start + 1, end))  # يبدأ توكن التشخيص من الحرف التالي للباء.
        else:  # يطبق التقطيع المعتاد على بقية النص.
            tokens.append((word, start, end))  # يحفظ التوكن كاملًا مع إزاحاته.
    return tokens  # يرجع التوكنات بترتيبها.


def features(tokens):  # يبني سمات كل توكن والكلمات القريبة منه.
    words = [token[0].casefold() for token in tokens]  # يوحّد حالة الحروف الإنجليزية في السمات.
    rows = []  # ينشئ قائمة لقواميس السمات.
    for i, word in enumerate(words):  # يمر على التوكنات بالتسلسل.
        row = {"word": word, "is_digit": word.isdigit()}  # يسجّل الكلمة وهل تتكون من أرقام.
        for size in (2, 3, 4):  # يستخدم ثلاثة أطوال للبادئات واللواحق.
            row[f"prefix{size}"] = word[:size]  # يضيف أول محارف الكلمة.
            row[f"suffix{size}"] = word[-size:]  # يضيف آخر محارف الكلمة.
        for shift in (-2, -1, 1, 2):  # ينظر إلى أقرب توكنين قبل الكلمة وبعدها.
            nearby = i + shift  # يحسب موضع التوكن المجاور.
            row[f"context{shift}"] = words[nearby] if 0 <= nearby < len(words) else "<boundary>"  # يمثل حدود النص بعلامة خاصة.
        rows.append(row)  # يضيف سمات التوكن الحالي.
    return rows  # يرجع السمات بالترتيب نفسه.


def gold_tags(note, tokens):  # يحوّل الكيانات المعلّمة بالإزاحات إلى وسوم تدريب BIO.
    tags = ["O"] * len(tokens)  # يبدأ باعتبار كل التوكنات خارج الكيانات.
    for span in note["entities"]:  # يمر على كل كيان مرجعي.
        indices = [i for i, (_, start, end) in enumerate(tokens) if start < span["end"] and end > span["start"]]  # يجد التوكنات المتقاطعة مع الكيان.
        if not indices or tokens[indices[0]][1] != span["start"] or tokens[indices[-1]][2] != span["end"]:  # يتحقق من تطابق حدود الكيان مع التقطيع.
            raise ValueError(f"Token alignment failed in {note['note_id']}: {span}")  # يوضح موضع خلل المحاذاة.
        for position, i in enumerate(indices):  # يوسم توكنات الكيان بالترتيب.
            if tags[i] != "O":  # يتحقق من عدم وسم التوكن لكيان آخر.
                raise ValueError(f"Overlapping entities in {note['note_id']}")  # يرفض الوسوم المتداخلة.
            prefix = "B" if position == 0 else "I"  # يضع B للبداية وI للاستمرار.
            tags[i] = f"{prefix}-{span['label']}"  # يضيف نوع الكيان للوسم.
    return tags  # يرجع الوسم المرجعي لكل توكن.


def decode(text, tokens, tags):  # يعيد بناء الكيانات من وسوم BIO المتوقعة.
    spans = []  # ينشئ قائمة الكيانات الناتجة.
    current = None  # يتتبّع الكيان الجاري.
    for (_, start, end), tag in zip(tokens, tags):  # يربط إزاحات كل توكن بوسمه.
        if tag == "O":  # يتعامل مع توكن خارج أي كيان.
            current = None  # ينهي الكيان الجاري.
            continue  # ينتقل للتوكن التالي.
        prefix, label = tag.split("-", 1)  # يفصل وسم الموضع عن نوع الكيان.
        if prefix == "I" and current is not None and current["label"] == label:  # يتحقق من إمكانية تمديد الكيان الجاري.
            current["end"] = end  # يمد نهاية الكيان لنهاية التوكن الحالي.
        else:  # يبدأ كيانًا عند B أو عند I بلا بداية متوافقة.
            current = {"start": start, "end": end, "label": label}  # يحفظ نوع الكيان وإزاحاته.
            spans.append(current)  # يضيف الكيان لقائمة النتائج.
    for span in spans:  # يمر على الكيانات بعد اكتمال حدودها.
        span["text"] = text[span["start"]:span["end"]]  # يأخذ العبارة من النص الأصلي باستخدام الإزاحات.
    return spans  # يرجع الكيانات بالترتيب.


def assertion_context(text, span):  # يأخذ السياق القريب من التشخيص لتحديد حالته.
    before = re.split(r"[.!?؟؛;،,\n]", text[:span["start"]])[-1].split()  # يأخذ كلمات العبارة السابقة.
    after = re.split(r"[.!?؟؛;،,\n]", text[span["end"]:])[0].split()  # يأخذ كلمات العبارة اللاحقة.
    return " ".join([*before[-8:], "[DX]", *after[:8]])  # يستبدل التشخيص بعلامة ويبقي الكلمات الأقرب إليه.


def predict_assertion(text, span, model):  # يحدّد حالة التشخيص من القرائن أو المصنّف.
    context = assertion_context(text, span)  # يبني سياق التشخيص الحالي.
    if re.search(r"[ء-ي]", span["text"]):  # يفعّل القرائن عند وجود حروف عربية في التشخيص.
        normalized = ARABIC_DIACRITICS.sub("", context).translate(str.maketrans("أإآ", "ااا"))  # يوحّد التشكيل والتطويل والألف في السياق.
        before, _, after = normalized.partition("[DX]")  # يفصل السياق السابق واللاحق للكيان.
        cues = []  # يجمع الحالات المرشحة ومسافة كل قرينة.
        for label, pattern in ARABIC_CUES.items():  # يفحص الأنماط الخاصة بكل حالة.
            matches_before = re.finditer(r"\b(?:" + pattern + r")\b", before)  # يجد القرائن السابقة للتشخيص.
            matches_after = re.finditer(r"\b(?:" + pattern + r")\b", after)  # يجد القرائن اللاحقة للتشخيص.
            cues.extend((len(before[match.end():].split()), -len(match.group()), label) for match in matches_before)  # يسجّل مسافة وطول القرائن السابقة.
            cues.extend((len(after[:match.start()].split()), -len(match.group()), label) for match in matches_after)  # يسجّل مسافة وطول القرائن اللاحقة.
        nearby = sorted(cue for cue in cues if cue[0] <= 5)  # يرتّب الأقرب ثم الأطول، مثل «ما ثبت» قبل «ثبت».
        if nearby and len({cue[2] for cue in nearby if cue[:2] == nearby[0][:2]}) == 1:  # يعتمد القرينة الأعلى ترتيبًا إذا لم تتعارض مع قرينة مكافئة.
            return nearby[0][2]  # يرجع الحالة التي تدل عليها القرينة.
    return str(model.predict([context])[0])  # يرجع إلى المصنّف عندما لا توجد قرينة حاسمة.


def lexicon_matches(text, lexicon):  # يبحث عن مصطلحات قاموس التدريب داخل النص.
    return [{"start": match.start(), "end": match.end(), "text": match.group(), "label": lexicon["labels"][match.group().casefold()]} for match in lexicon["pattern"].finditer(text)]  # يرجع المطابقات بإزاحاتها الأصلية.


def recover_boundaries(spans, matches):  # يستعيد حدود المصطلحات المعروفة إذا جزّأها المصنّف.
    for match in matches:  # يعالج كل مطابقة من القاموس.
        overlaps = [span for span in spans if span["start"] < match["end"] and match["start"] < span["end"]]  # يجد التوقعات المتداخلة مع المطابقة.
        aligned = any(span["label"] == match["label"] and span["start"] == match["start"] for span in overlaps)  # يشترط اتفاق البداية والنوع.
        contained = all(match["start"] <= span["start"] and span["end"] <= match["end"] for span in overlaps)  # يمنع قص توقع أطول من المطابقة.
        if aligned and contained:  # يصحح الحدود إذا تحقق الشرطان.
            spans = [span for span in spans if span not in overlaps] + [match]  # يستبدل الأجزاء المتداخلة بالعبارة الكاملة.
    return sorted(spans, key=lambda span: span["start"])  # يرتّب الكيانات حسب مواضعها في النص.


def extract(text, tagger, assertion_model, lexicon):  # يستخرج الكيانات وحالاتها من ملاحظة واحدة.
    terms = tuple(sorted((term for term, label in lexicon["labels"].items() if label == "DIAGNOSIS" and re.search(r"[ء-ي]", term)), key=lambda term: (-len(term), term)))  # يرتّب التشخيصات العربية المعروفة من الأطول للأقصر.
    prefixed = list(attached_b_pattern(terms).finditer(text))  # يجد مواضع التشخيصات المعروفة المسبوقة بباء ملتصقة.
    tokens = tokenize(text, (match.start() for match in prefixed))  # يفصل الباء في التقطيع ويحتفظ بالنص الأصلي.
    if not tokens:  # يتحقق من وجود توكنات للتحليل.
        return []  # يرجع قائمة فارغة للنص الفارغ.
    spans = decode(text, tokens, tagger.predict(features(tokens)))  # يتوقع وسوم BIO ويحوّلها إلى كيانات.
    matches = lexicon_matches(text, lexicon)  # يجلب المطابقات المعتادة من القاموس.
    matches.extend({"start": match.start("diagnosis"), "end": match.end("diagnosis"), "text": match.group("diagnosis"), "label": "DIAGNOSIS"} for match in prefixed)  # يضيف التشخيصات المسبوقة بالباء مع حدود تستثني الباء.
    spans = recover_boundaries(spans, matches)  # يكمل حدود التشخيص عندما يلتقط الموديل بدايته ونوعه.
    for span in spans:  # يحدّد حالة كل كيان مستخرج.
        if span["label"] == "DIAGNOSIS":  # يعالج التشخيصات باستخدام السياق.
            span["assertion"] = predict_assertion(text, span, assertion_model)  # يحدد إن كان التشخيص قائمًا أو منفيًا أو تاريخيًا أو مشتبهًا به.
        else:  # يعالج كيانات الإجراءات.
            span["assertion"] = "ordered"  # يسند حالة الإجراء المعتمدة في الداتا.
    return spans  # يرجع الكيانات وحدودها وأنواعها وحالاتها.


def keys(spans, joint):  # يحوّل الكيانات إلى مفاتيح للمقارنة الدقيقة.
    fields = ("start", "end", "label", "assertion") if joint else ("start", "end", "label")  # يضمّن الحالة عندما يكون التقييم مشتركًا.
    return {tuple(span[field] for field in fields) for span in spans}  # ينشئ مجموعة من مفاتيح الكيانات.


def evaluate(pairs, joint=False):  # يحسب Precision وRecall وF1 على مستوى الكيان.
    matched = expected_count = predicted_count = 0  # يصفّر عدادات المطابقة.
    for note, prediction in pairs:  # يقارن التوقعات بالمرجع ملاحظة بملاحظة.
        expected = keys(note["entities"], joint)  # يبني مفاتيح الكيانات المرجعية.
        actual = keys(prediction, joint)  # يبني مفاتيح الكيانات المتوقعة.
        matched += len(expected & actual)  # يعد الكيانات المتطابقة داخل الملاحظة.
        expected_count += len(expected)  # يجمع عدد الكيانات المرجعية.
        predicted_count += len(actual)  # يجمع عدد الكيانات المتوقعة.
    precision = matched / predicted_count if predicted_count else 0.0  # يقيس نسبة الصحيح من التوقعات.
    recall = matched / expected_count if expected_count else 0.0  # يقيس نسبة الكيانات المرجعية التي استُخرجت.
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0  # يحسب المتوسط التوافقي للمقياسين.
    return f"P={precision:.3f} R={recall:.3f} F1={f1:.3f}"  # ينسّق المقاييس للعرض.


def train():  # يدرّب مستخرج الكيانات ومصنّف حالة التشخيص.
    notes = read_jsonl("clinical_ner.jsonl")  # يقرأ الملاحظات والكيانات المرجعية.
    token_features, token_labels = [], []  # يهيّئ مدخلات ووسوم مصنّف BIO.
    contexts, assertions = [], []  # يهيّئ سياقات ووسوم مصنّف الحالة.
    original_training = [note for note in notes if note["split"] == "train"]  # يختار الملاحظات المخصصة للتدريب.
    training = [*original_training, *extra_notes(original_training)]  # يضيف ملاحظات التدريب الاصطناعية.
    labels = {span["text"].casefold(): span["label"] for note in original_training for span in note["entities"]}  # يبني القاموس من التدريب الأصلي.
    terms = "|".join(re.escape(term) for term in sorted(labels, key=lambda term: (-len(term), term)))  # يرتّب العبارات الأطول أولًا ويحمي رموزها.
    lexicon = {"labels": labels, "pattern": re.compile(r"(?<![^\Wـ])(?:" + terms + r")(?!\w)", flags=re.IGNORECASE)}  # يطابق المصطلح الكامل ويسمح بالبادئة المطوّلة.

    for note in training:  # يمر على الملاحظات التدريبية.
        text = note["text"]  # يأخذ النص الأصلي للحفاظ على الإزاحات.
        tokens = tokenize(text)  # يقطّع ملاحظات التدريب بالطريقة الأساسية.
        token_features.extend(features(tokens))  # يجمع سمات التوكنات.
        token_labels.extend(gold_tags(note, tokens))  # يجمع وسوم BIO المرجعية.
        for span in note["entities"]:  # يمر على الكيانات المعلّمة.
            if span["label"] == "DIAGNOSIS":  # يستخدم التشخيصات لتدريب مصنّف الحالة.
                contexts.append(assertion_context(text, span))  # يضيف سياق التشخيص بعد إخفاء اسمه.
                assertions.append(span["assertion"])  # يضيف الحالة المرجعية للسياق.

    tagger = make_pipeline(DictVectorizer(), LogisticRegression(max_iter=1000))  # يجهّز مسار تصنيف وسوم BIO.
    tagger.fit(token_features, token_labels)  # يدرّب المصنّف على سمات التوكنات ووسومها.
    context_vectors = TfidfVectorizer(analyzer="char", ngram_range=(2, 5), sublinear_tf=True)  # يمثّل السياق باستخدام مقاطع المحارف.
    assertion_model = make_pipeline(context_vectors, LogisticRegression(max_iter=1000, class_weight="balanced"))  # يبني مصنّف حالة التشخيص مع موازنة الفئات.
    assertion_model.fit(contexts, assertions)  # يدرّب المصنّف على السياقات وحالاتها.

    for split in ("validation", "test"):  # يقيّم على التحقق والاختبار.
        for language in ("ar", "en"):  # يحسب نتائج العربي والإنجليزي بصورة مستقلة.
            subset = [note for note in notes if note["split"] == split and note["language"] == language]  # يختار ملاحظات القسم واللغة.
            pairs = [(note, extract(note["text"], tagger, assertion_model, lexicon)) for note in subset]  # يشغّل الاستدلال ويقرن التوقع بالملاحظة المرجعية.
            print(f"{split} {language} | NER {evaluate(pairs)} | NER + assertion {evaluate(pairs, joint=True)}")  # يطبع نتيجة NER ثم النتيجة مع اشتراط صحة الحالة.

    MODELS.mkdir(exist_ok=True)  # ينشئ مجلد الحفظ إذا لزم.
    joblib.dump((tagger, assertion_model, lexicon), MODEL_PATH)  # يحفظ النموذجين والقاموس معًا.


if __name__ == "__main__":  # يشغّل واجهة الطرفية عند تشغيل الملف مباشرة.
    parser = argparse.ArgumentParser(description="Train or run bilingual NER.")  # يصف وظيفة البرنامج.
    parser.add_argument("--text", help="Extract entities from this clinical note.")  # يستقبل ملاحظة اختبار اختيارية.
    args = parser.parse_args()  # يقرأ خيارات الطرفية.
    if args.text is None:  # يبدأ التدريب إذا لم يُمرّر نص.
        train()  # يدرّب النموذجين ويقيّمهما ويحفظهما.
    elif not MODEL_PATH.is_file():  # يفحص وجود ملف الموديل قبل التنبؤ.
        parser.error("Run python ner_model.py first to train the models.")  # يوضح أمر التدريب إذا كان الملف مفقودًا.
    else:  # ينتقل إلى التنبؤ عند وجود نص وموديل مدرّب.
        tagger, assertion_model, lexicon = joblib.load(MODEL_PATH)  # يحمّل النموذجين والقاموس.
        spans = extract(args.text, tagger, assertion_model, lexicon)  # يشغّل استخراج الكيانات على النص.
        print(json.dumps(spans, ensure_ascii=False, indent=2))  # يعرض النتيجة بصيغة JSON ويحافظ على وضوح العربي.