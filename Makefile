# Шляхи
FRONTEND_DIR=gui

## 🔧 Запускає фронтенд із HMR (порт 5173)
fe:
	cd $(FRONTEND_DIR) && npm run dev

## 🐍 Запускає Flask-бекенд (порт 5000)
be:
	py app.py

## 🚀 Запускає фронт і бек одночасно (неблокуюче)
dev:
	@echo "Starting frontend and backend..."
	$(MAKE) be & $(MAKE) fe

## 🏗️ Збірка фронтенду для продакшну
build:
	cd $(FRONTEND_DIR) && npm run build

## 🧹 Очистка збірки фронту
clean:
	rm -rf $(FRONTEND_DIR)/dist