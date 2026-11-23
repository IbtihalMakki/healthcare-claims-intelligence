import re  # يستخرج علامات الكيانات من صيغ التدريب والاختبار.
from collections import defaultdict  # يجمع التشخيص والإجراء لكل لغة.

from common import DATA  # يستخدم مسار الداتا المرفقة بالمشروع.

MARKUP = re.compile(r"\[([^:\]]+):([^:\]]+):([^\]]+)\]")  # يقرأ الوسم والحالة والنص بين أقواس مربعة.


def parse_note(marked):  # يفصل النص الصافي عن وسومه ويحسب إزاحات المحارف.
    parts, spans = [], []  # يجمع أجزاء النص والكيانات.
    offset = cursor = 0  # يبدأ من أول محرف في النص الأصلي والمعلّم.
    for match in MARKUP.finditer(marked):  # يمر على الكيانات بالترتيب.
        prefix = marked[cursor:match.start()]  # يأخذ النص السابق للكيان.
        label, assertion, mention = match.groups()  # يستخرج خصائص الكيان.
        start = offset + len(prefix)  # يحسب بداية العبارة في النص بعد إزالة الوسم.
        offset = start + len(mention)  # يحسب نهاية العبارة الحصرية.
        parts.extend((prefix, mention))  # يعيد إضافة النص والعبارة نفسها.
        spans.append({"start": start, "end": offset, "label": label, "assertion": assertion, "text": mention})  # يحفظ الإزاحات مع الوسم.
        cursor = match.end()  # يتجاوز أقواس الوسم.
    parts.append(marked[cursor:])  # يحفظ نهاية الجملة بعد آخر كيان.
    return "".join(parts), spans  # يعيد النص الصافي والكيانات الموثقة.


def extra_intents():  # يقرأ استفسارات إضافية مخصصة للتدريب.
    label = None  # يحتفظ بالنوايا من عناوين الأقسام.
    for line in (DATA / "train_intents_extra.txt").read_text(encoding="utf-8").splitlines():  # يقرأ أمثلة الملف بالترتيب.
        if line.startswith("["):  # يتعرف على بداية قسم النية.
            label = line[1:-1]  # يزيل الأقواس للحصول على اسم النية.
        elif line.strip():  # ينشئ سجلًا من كل مثال غير فارغ.
            language, text = line.split("|", 1)  # يفصل اللغة عن رسالة المستخدم.
            yield {"language": language, "text": text, "intent": label, "split": "train"}  # يحدد المثال كتدريب فقط.


def extra_notes(training):  # يربط صيغ الحالة بمصطلحات مأخوذة من التدريب الأصلي.
    pairs = defaultdict(set)  # يجمع أزواج التشخيص والإجراء بدون تكرار.
    for row in training:  # يستخدم الملاحظات التي سبق اختيارها للتدريب.
        terms = {span["label"]: span["text"] for span in row["entities"]}  # يستخرج نوع كل عبارة ونصها.
        pairs[row["language"]].add((terms["DIAGNOSIS"], terms["PROCEDURE"]))  # يسجل الزوج لكل لغة.
    forms = (DATA / "train_note_forms.txt").read_text(encoding="utf-8").splitlines()  # يقرأ صيغ الحالة الجديدة.
    for index, line in enumerate(forms):  # يعطي كل صيغة رقمًا ثابتًا.
        language, assertion, template = line.split("|", 2)  # يفصل اللغة والحالة والقالب.
        terms = sorted(pairs[language])  # يجعل اختيار المصطلحات قابلًا لإعادة التشغيل.
        for sample in range(min(8, len(terms))):  # يأخذ ثمانية أزواج متنوعة في كل صيغة.
            diagnosis, procedure = terms[(index * 7 + sample) % len(terms)]  # يوزع الأزواج عبر الصيغ.
            tagged = template.replace("[DX]", f"[DIAGNOSIS:{assertion}:{diagnosis}]")  # يدرج اسم التشخيص مع حالته.
            tagged = tagged.replace("[PROC]", f"[PROCEDURE:ordered:{procedure}]")  # يدرج الإجراء مع وسمه.
            text, spans = parse_note(tagged)  # يحسب الإزاحات بعد إدراج العبارتين.
            yield {"note_id": f"extra-{index}-{sample}", "language": language, "split": "train", "text": text, "entities": spans}  # يعيد سجل تدريب محليًا.


def challenge_intents(filename="challenge_intents.txt"):  # يحمّل رسائل التقييم الإضافي.
    for line in (DATA / filename).read_text(encoding="utf-8").splitlines():  # يقرأ الاستفسارات الجديدة.
        label, language, text = line.split("|", 2)  # يفصل الحقول حسب علامة الفصل.
        yield {"language": language, "text": text, "intent": label}  # يحفظ النية المرجعية للمقارنة.


def challenge_notes():  # يحمّل ملاحظات التقييم الإضافي بوسومها.
    for line in (DATA / "challenge_notes.txt").read_text(encoding="utf-8").splitlines():  # يمر على الملاحظات المعلّمة.
        language, marked = line.split("|", 1)  # يفصل اللغة عن الملاحظة.
        text, spans = parse_note(marked)  # يرجع الإزاحات المكافئة بعد إزالة الوسوم.
        yield {"language": language, "text": text, "entities": spans}  # يوفر النص والوسوم المرجعية للتقييم فقط.
