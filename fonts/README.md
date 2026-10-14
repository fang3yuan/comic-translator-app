# Arabic fonts

Put your Arabic `.ttf` / `.otf` files in this folder. They appear automatically in the
**Fonts & Style** tab, where each comic role can be linked to a font:

| Role | Typical choice |
|------|----------------|
| Dialogue / Speech | clean, readable (Cairo, Tajawal, Noto Naskh Arabic) |
| Narration Box | calmer, text-like (Noto Naskh, Amiri) |
| Scream / SFX | heavy display or brush style (Lalezar, Reem Kufi Bold, Aref Ruqaa Bold) |
| Whisper / Thoughts | light or hand-written (Harmattan, Mada Light, Handjet) |

If a role has no explicit mapping, the app first looks for a file whose name hints at the role
(e.g. `*Bold*`/`*Black*` for SFX, `*Light*`/`*Hand*` for whispers) and finally uses
`default_arabic.ttf`.

`default_arabic.ttf` shipped here is DejaVu Sans Bold (free licence, includes Arabic). It is only a
placeholder: replace it with a manga-style Arabic font for production use, keeping the same file
name, or change `fonts.default_font` in `config/settings.json`.

Free sources: Google Fonts (fonts.google.com, filter by the Arabic script).
Check each font's licence before publishing translated pages.
