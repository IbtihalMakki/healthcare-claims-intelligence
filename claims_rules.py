import argparse  # يقرأ رقم المطالبة الاختياري من سطر الأوامر.
import json  # يطبع تقييم المطالبة بصيغة JSON.

from common import read_csv, read_json, read_jsonl  # يستورد قارئات ملفات الداتا الثلاثة.


def review_claim(claim, policy):  # يحسب أسباب المراجعة من المرفقات والسياسة.
    if claim["policy_id"] != policy["policy_id"] or claim["procedure_group"] != policy["procedure_group"]:  # يفحص توافق المطالبة والسياسة.
        raise ValueError(f"Policy mismatch for {claim['claim_id']}")  # يوقف المراجعة إذا مررت سياسة غير مطابقة.
    attached = set(claim["attached_documents"].split("|")) if claim["attached_documents"] else set()  # يحول خلية المرفقات إلى مجموعة أسماء.
    missing = [doc for doc in policy["required_documents"] if doc not in attached]  # يستخرج المستندات المطلوبة وغير المرفقة.
    reasons = ["missing_documentation"] if missing else []  # يضيف سبب نقص المستندات عند الحاجة.
    if policy["requires_prior_auth"]:  # يراجع الموافقة عندما تشترطها السياسة.
        status = claim["prior_authorization"]  # يقرأ قيمة الموافقة النصية من CSV.
        if status not in {"True", "False"}:  # يفحص أن الحقل يحتوي على قيمة معروفة.
            raise ValueError(f"Invalid authorization status for {claim['claim_id']}: {status!r}")  # يوضح الخلل في السجل.
        if status == "False":  # يتحقق من غياب الموافقة المطلوبة.
            reasons.append("missing_prior_authorization")  # يضيف غياب الموافقة إلى أسباب المراجعة.
    return {"review_required": bool(reasons), "review_reasons": reasons, "missing_documents": missing}  # يعيد النتيجة ومبرراتها.


def check_examples(claims, policies):  # يقارن القواعد بالنتائج المتوقعة في حالات الاختبار.
    cases = read_jsonl("evaluation_cases.jsonl")  # يحمّل السجلات المرجعية الـ32.
    mismatches = []  # يجمع اختلافات الاختبار ليعرضها معًا.
    for case in cases:  # يعالج كل حالة اختبار مستقلة.
        claim_id = case["claim_id"]  # يقرأ مفتاح الربط مع جدول المطالبات.
        claim = claims[claim_id]  # يسترجع المطالبة الأصلية.
        policy = policies[case["policy_id"]]  # يسترجع السياسة المحددة لحالة الاختبار.
        result = review_claim(claim, policy)  # يحسب النتيجة من المدخلات فقط.
        expected = {key: case[f"expected_{key}"] for key in result}  # يزيل بادئة expected لأجل المقارنة.
        if result != expected:  # يتحقق من تطابق المراجعة وأسبابها والمستندات الناقصة.
            mismatches.append({"claim_id": claim_id, "expected": expected, "actual": result})  # يحفظ تفاصيل الاختلاف.
    print(f"Claims rule checks: {len(cases) - len(mismatches)}/{len(cases)} passed")  # يطبع عدد الحالات المطابقة.
    if mismatches:  # يتعامل مع الاختلافات إن وجدت.
        print(json.dumps(mismatches, ensure_ascii=False, indent=2))  # يطبع النتائج المتوقعة والفعلية.
        raise SystemExit(1)  # يرجع حالة فشل لأدوات التشغيل والاختبار.


if __name__ == "__main__":  # ينفذ هذا القسم عند تشغيل الملف مباشرة.
    parser = argparse.ArgumentParser(description="Review one claim or check all reference cases.")  # يصف عمليتي الملف.
    parser.add_argument("--claim-id", help="Claim ID to evaluate, for example CLM-00001.")  # يتيح تجربة مطالبة محددة.
    args = parser.parse_args()  # يقرأ الخيارات.
    claims = {row["claim_id"]: row for row in read_csv("claims.csv")}  # يفهرس المطالبات حسب المعرّف.
    policies = {row["policy_id"]: row for row in read_json("policies.json")}  # يفهرس السياسات حسب المعرّف.
    if args.claim_id is None:  # يحدد أن المطلوب تشغيل جميع حالات الاختبار.
        check_examples(claims, policies)  # يقارن نتائج القواعد بالملف المرجعي.
    elif args.claim_id not in claims:  # يتحقق من أن رقم المطالبة معروف.
        parser.error(f"Unknown claim ID: {args.claim_id}")  # يوضح أن رقم المطالبة غير موجود.
    else:  # يراجع مطالبة واحدة إذا توفر معرّف صحيح.
        claim = claims[args.claim_id]  # يسترجع بيانات المطالبة المحددة.
        result = review_claim(claim, policies[claim["policy_id"]])  # يحدد سياستها ويحسب نتيجة المراجعة.
        print(json.dumps({"claim_id": args.claim_id, **result}, ensure_ascii=False, indent=2))  # يعرض الرقم والنتيجة في JSON.
