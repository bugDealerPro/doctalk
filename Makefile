.PHONY: up down test dev env

up:
	docker compose up --build

down:
	docker compose down

test:
	pytest

dev:
	chainlit run app/main.py

env:
	cp -n .env.example .env
