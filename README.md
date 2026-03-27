# Leitor de Questionários PDF — Render

Projeto pronto para deploy no Render.

## Deploy manual no Render
1. criar repositório no GitHub
2. carregar estes ficheiros
3. no Render, criar Web Service
4. ligar ao repositório
5. confirmar:
   - Environment: Python
   - Build Command: pip install -r requirements.txt
   - Start Command: gunicorn app:app

## Importante
A calibração em config.json é inicial. O mais provável é precisares de ajustar:
- page_zones
- mark_threshold
- uncertain_threshold
- cell_padding_x
- cell_padding_y
