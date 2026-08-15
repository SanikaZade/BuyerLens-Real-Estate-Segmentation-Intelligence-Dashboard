@echo off
echo ========================================================
echo   Starting Parcl Buyer Segmentation Dashboard...
echo ========================================================
echo.
echo Installing requirements...
python -m pip install -r requirements.txt
echo.
echo Launching Streamlit...
python -m streamlit run app.py
pause
