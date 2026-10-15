# 🧽 محركات المسح: LaMa وMI-GAN

| المحرك | الإعداد | الملف | ملاحظة |
|--------|---------|-------|--------|
| OpenCV (Telea) | `telea` | لا يحتاج | الأخف والأقل جودة |
| LaMa | `lama` أو `auto` | `models/lama_fp32.onnx` | الأجود، يُنزَّل من Setup |
| **MI-GAN** (تجريبي) | `migan` | `models/migan_512.onnx` | مصمم للأجهزة الضعيفة، أخف من LaMa بحسب ورقته (ICCV 2023) |

## تجربة MI-GAN
1. نزّل نموذج MI-GAN 512 بصيغة ONNX (مثلًا `FreeHugsForRobots/ps-inpaint-migan` على Hugging Face، رخصة MIT) وسمّه `migan_512.onnx` وضعه في مجلد `models`.
   يجب أن يقبل مدخلًا واحدًا بشكل `[1,4,512,512]` (قناة القناع ثم الصورة) ويُخرج `[1,3,512,512]`.
2. قارن على صفحاتك:
   ```bash
   python tools/bench_inpaint.py page.png --out bench_out
   ```
   يطبع زمن كل محرك ويحفظ صورة (الأصل | Telea | LaMa | MI-GAN).
3. أعجبتك النتيجة؟ اختر `migan` من قائمة **Inpainting** في تبويب General (أو `--inpaint migan`).

> ملاحظة: التوصيل مختبر بنموذج بديل صغير فقط. جودة MI-GAN الفعلية على المانهوا لم تُقَس بعد.
