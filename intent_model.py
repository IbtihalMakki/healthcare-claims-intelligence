import argparse  # يحدد خيارات التدريب والتنبؤ في الطرفية.
import json  # يطبع نتيجة التنبؤ بصيغة قابلة للقراءة والمعالجة.

import joblib  # يحفظ نموذج scikit-learn ويسترجعه.
from sklearn.feature_extraction.text import TfidfVectorizer  # يحوّل النص إلى أوزان رقمية.
from sklearn.linear_model import LogisticRegression  # يتعلم التمييز بين النوايا الثماني.
from sklearn.metrics import classification_report, f1_score  # يلخص أداء التصنيف.
from sklearn.pipeline import make_pipeline  # يجمع تحويل النص والتصنيف في مسار واحد.

from common import MODELS, read_csv  # يستخدم مسارات المشروع وقارئ CSV المشترك.

from training_data import extra_intents  # يقرأ الصياغات الإضافية المخصصة للتدريب.

MODEL_PATH = MODELS / "intent_v3.joblib"  # يحدد اسم الملف الذي سيحفظ النموذج.


def train():  # يدرب النموذج على جزء التدريب ويحفظه.
    rows = read_csv("intents.csv")  # يحمّل الرسائل والنوايا واللغات والتقسيمات.
    training = [row for row in rows if row["split"] == "train"]  # يختار سجلات التدريب فقط.
    training.extend(extra_intents())  # يضيف استفسارات متنوعة عند التدريب فقط.
    vectorizer = TfidfVectorizer(analyzer="char", ngram_range=(2, 5), sublinear_tf=True)  # يمثل مقاطع المحارف لمراعاة تنوع الكتابة.
    classifier = LogisticRegression(C=4, max_iter=1000)  # يستخدم إعداد الانتظام الذي اختير على قسم التحقق.
    model = make_pipeline(vectorizer, classifier)  # يحفظ المعالجة والنموذج في مسار واحد.
    model.fit([row["text"] for row in training], [row["intent"] for row in training])  # يتعلم من الجمل والوسوم الصحيحة.
    labels = sorted({row["intent"] for row in training})  # يثبت ترتيب النوايا للتقارير.
    for split in ("validation", "test"):  # يقيّم النموذج على جزأين محفوظين خارج التدريب.
        subset = [row for row in rows if row["split"] == split]  # يحدد سجلات الجزء الحالي.
        gold = [row["intent"] for row in subset]  # يجمع النوايا المرجعية.
        predicted = model.predict([row["text"] for row in subset])  # يتنبأ بالنوايا من النصوص.
        print(f"\n{split} | {len(subset)} samples")  # يطبع اسم الجزء وعدد أمثلته.
        print(classification_report(gold, predicted, labels=labels, digits=3, zero_division=0))  # يعرض Precision وRecall وF1 لكل نية.
        for language in ("ar", "en"):  # يحسب مقياسًا منفصلًا لكل لغة.
            indices = [index for index, row in enumerate(subset) if row["language"] == language]  # يحدد مواقع أمثلة اللغة الحالية.
            score = f1_score([gold[i] for i in indices], predicted[indices], labels=labels, average="macro", zero_division=0)  # يعطي كل نية الوزن نفسه.
            print(f"{language} macro F1: {score:.3f}")  # يعرض نتيجة اللغة الحالية.
    MODELS.mkdir(exist_ok=True)  # ينشئ مجلد النماذج إذا كان مفقودًا.
    joblib.dump(model, MODEL_PATH)  # يحفظ المصنف وقاموس مقاطع النص معًا.


def predict(text):  # يتوقع نية استفسار واحد باستخدام النموذج المحفوظ.
    if not text.strip():  # يفحص وجود نص فعلي.
        raise ValueError("Enter a non-empty message.")  # يوقف التنبؤ عند الإدخال الفارغ.
    model = joblib.load(MODEL_PATH)  # يسترجع النموذج الناتج من تدريبك المحلي.
    intent = model.predict([text])[0]  # يأخذ أول نتيجة لأن الإدخال جملة واحدة.
    return {"text": text, "intent": str(intent)}  # يعيد النص والفئة المتوقعة.


if __name__ == "__main__":  # يشغّل واجهة الطرفية عند تشغيل الملف مباشرة.
    parser = argparse.ArgumentParser(description="Train or run bilingual intent classification.")  # ينشئ محلل خيارات الطرفية.
    parser.add_argument("--text", help="Predict the intent of this message.")  # يضيف خيار تجربة استفسار بعد التدريب.
    args = parser.parse_args()  # يقرأ الخيارات المرسلة للبرنامج.
    if args.text is None:  # عند غياب النص يبدأ التدريب والتقييم.
        train()  # ينفذ التدريب ويحفظ النموذج.
    elif not MODEL_PATH.is_file():  # يتحقق من أن النموذج مدرّب قبل تجربة نص.
        parser.error("Run python intent_model.py first to train the model.")  # يطبع أمر التدريب المطلوب.
    else:  # إذا توفر النص والنموذج ينتقل إلى التنبؤ.
        print(json.dumps(predict(args.text), ensure_ascii=False, indent=2))  # يطبع النتيجة ويحافظ على الحروف العربية.
