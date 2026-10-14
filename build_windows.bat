@echo off
chcp 65001 >nul
cd /d "%~dp0"
echo ==== Building ComicTranslator.exe ====
where python >nul 2>nul || (echo Python 3.10+ is required: https://www.python.org/downloads/ ^(tick "Add to PATH"^) & pause & exit /b 1)
if not exist .venv-build python -m venv .venv-build
call .venv-build\Scripts\activate.bat
python -m pip install --upgrade pip
pip install -r requirements.txt pyinstaller || (echo Install failed & pause & exit /b 1)
pyinstaller comic_translator.spec --noconfirm --clean || (echo Build failed & pause & exit /b 1)
powershell -NoProfile -Command "Compress-Archive -Path dist\ComicTranslator -DestinationPath dist\ComicTranslator-windows.zip -Force"
echo.
echo Done:  dist\ComicTranslator\ComicTranslator.exe
echo Zip:   dist\ComicTranslator-windows.zip
pause
