from pathlib import Path  # يبني مسارات تشتغل على ويندوز ولينكس.
import csv  # يقرأ جداول CSV مع دعم الفواصل داخل النص.
import json  # يقرأ ملفات JSON وJSONL.

ROOT = Path(__file__).resolve().parent  # يحدد مكان ملفات المشروع من موقع هذا الملف.
DATA = ROOT / "data"  # يحدد مجلد بيانات v0.2 المرفق مع المشروع.
MODELS = ROOT / "artifacts"  # يحدد مجلد النماذج الناتجة من التدريب.


def read_csv(filename):  # يحمّل ملف CSV على شكل قائمة من القواميس.
    with (DATA / filename).open(encoding="utf-8", newline="") as file:  # يفتح الملف بترميز يدعم العربي.
        return list(csv.DictReader(file))  # يستخدم أسماء الأعمدة كمفاتيح لكل سجل.


def read_json(filename):  # يحمّل وثيقة JSON واحدة مثل قائمة السياسات.
    return json.loads((DATA / filename).read_text(encoding="utf-8"))  # يقرأ النص ويفسره كـ JSON.


def read_jsonl(filename):  # يحمّل قائمة سجلات من ملف فيه سجل JSON بكل سطر.
    lines = (DATA / filename).read_text(encoding="utf-8").splitlines()  # يقرأ السطور بالترتيب.
    return [json.loads(line) for line in lines if line.strip()]  # يحول السطور غير الفارغة إلى قواميس.
