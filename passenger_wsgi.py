# -*- coding: utf-8 -*-

import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from app import app
# from fastapi.responses import PlainTextResponse
from starlette.middleware.wsgi import WSGIMiddleware

# FastAPI — це ASGI, а Phusion Passenger працює через WSGI.
# Тому потрібен WSGI-адаптер.

# Поверни простий Hello world, якщо треба перевірити роботу
# def application(environ, start_response):
#   start_response('200 OK', [('Content-Type', 'text/plain')])
#   return [b"Hello from WSGI"]

# або запускай FastAPI
application = app