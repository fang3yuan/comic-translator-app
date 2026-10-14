# 📦 بناء التطبيق ونشره على GitHub

## بناء ملف التشغيل بنقرة واحدة

| النظام | الأمر |
|--------|-------|
| 🪟 ويندوز | انقر مرتين على **`build_windows.bat`** |
| 🍎/🐧 | `./build.sh` |

ينشئ بيئة مؤقتة، يثبّت المتطلبات وPyInstaller، ثم يبني:
- `dist/ComicTranslator/ComicTranslator.exe` (مجلد كامل، لا يحتاج بايثون عند المستخدم)
- `dist/ComicTranslator-windows.zip` (أو `.tar.gz`) جاهز للتوزيع

الحجم حوالي 400 ميغابايت (يشمل OCR وonnxruntime). نموذج LaMa **لا** يُضمَّن؛ يُنزَّل من نافذة **Setup** عند أول استخدام، وبدونه يعمل المسح بـ OpenCV.

> جُرّب البناء فعليًا على لينكس: يعمل، ويقرأ النص، وتفتح النافذة. بناء ويندوز/ماك يتم عبر GitHub Actions.

## رفع المشروع على GitHub
```bash
git init && git add . && git commit -m "Comic Translator"
git branch -M main
git remote add origin https://github.com/USER/REPO.git
git push -u origin main
```
عند كل push يعمل ملف `.github/workflows/build.yml`: يشغّل الاختبارات ثم يبني الثلاثة أنظمة ويحفظ الملفات في **Actions ▸ Artifacts**.

## إصدار Release
```bash
git tag v2.0.0 && git push origin v2.0.0
```
ينشئ GitHub صفحة Release تلقائيًا ويرفق بها `ComicTranslator-windows.zip` و`-macos.tar.gz` و`-linux.tar.gz`.

## ملاحظات
- المتغير `COMIC_TRANSLATOR_HOME` يغيّر مكان الإعدادات والنماذج والخطوط. افتراضيًا بجانب البرنامج، وإن لم تُسمح الكتابة فـ `~/ComicTranslator`.
- ماك: قد يطلب النظام السماح بتشغيل تطبيق غير موقّع (Right-click ▸ Open).
