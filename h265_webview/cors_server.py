#!/usr/bin/env python3
"""
CORS-enabled HTTP server для FFmpeg WASM H.265 Video Player
Поддерживает Cross-Origin Resource Sharing для работы с внешними ресурсами
"""

import http.server
import socketserver
import argparse
import sys
import os
from urllib.parse import urlparse


class CORSHTTPRequestHandler(http.server.SimpleHTTPRequestHandler):
    """HTTP Request Handler с поддержкой CORS"""

    def end_headers(self):
        # Добавляем CORS заголовки
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header(
            "Access-Control-Allow-Methods", "GET, POST, PUT, DELETE, OPTIONS"
        )
        self.send_header(
            "Access-Control-Allow-Headers",
            "Content-Type, Authorization, X-Requested-With",
        )
        self.send_header("Access-Control-Max-Age", "86400")  # 24 часа

        # Заголовки для безопасности SharedArrayBuffer (нужно для FFmpeg WASM)
        self.send_header("Cross-Origin-Embedder-Policy", "require-corp")
        self.send_header("Cross-Origin-Opener-Policy", "same-origin")

        # Кэширование для статических ресурсов
        if self.path.endswith((".js", ".css", ".wasm")):
            self.send_header("Cache-Control", "public, max-age=3600")

        super().end_headers()

    def do_OPTIONS(self):
        """Обработка preflight OPTIONS запросов"""
        self.send_response(200)
        self.end_headers()

    def log_message(self, format, *args):
        """Улучшенное логирование"""
        print(f"[{self.log_date_time_string()}] {format % args}")


def main():
    parser = argparse.ArgumentParser(
        description="CORS-enabled HTTP server для H.265 Video Player"
    )
    parser.add_argument(
        "--port", "-p", type=int, default=8000, help="Порт сервера (по умолчанию: 8000)"
    )
    parser.add_argument(
        "--host",
        "-H",
        type=str,
        default="127.0.0.1",
        help="Хост для привязки (по умолчанию: 127.0.0.1)",
    )
    parser.add_argument(
        "--directory",
        "-d",
        type=str,
        default=".",
        help="Директория для сервера (по умолчанию: текущая)",
    )

    args = parser.parse_args()

    # Проверяем существование основных файлов
    required_files = ["index.html", "script.js", "package.json"]
    missing_files = [f for f in required_files if not os.path.exists(f)]

    if missing_files:
        print(f"⚠️  Предупреждение: Не найдены файлы: {', '.join(missing_files)}")
        print("   Убедитесь, что запускаете сервер из корневой папки проекта")

    # Смена директории
    os.chdir(args.directory)

    try:
        with socketserver.TCPServer(
            (args.host, args.port), CORSHTTPRequestHandler
        ) as httpd:
            host_display = args.host if args.host != "0.0.0.0" else "localhost"

            print("=" * 60)
            print("🚀 FFmpeg WASM H.265 Video Player Server")
            print("=" * 60)
            print(f"📺 Сервер запущен: http://{host_display}:{args.port}")
            print(f"🔒 CORS включен для внешних ресурсов")
            print(f"🛡️  SharedArrayBuffer поддержка для FFmpeg WASM")
            print(f"📂 Директория: {os.getcwd()}")
            print("=" * 60)
            print("📋 Для остановки нажмите Ctrl+C")
            print()

            httpd.serve_forever()

    except KeyboardInterrupt:
        print("\n🛑 Сервер остановлен пользователем")
        sys.exit(0)
    except OSError as e:
        if e.errno == 98:  # Address already in use
            print(f"❌ Ошибка: Порт {args.port} уже используется")
            print(
                f"   Попробуйте другой порт: python3 {sys.argv[0]} --port {args.port + 1}"
            )
        else:
            print(f"❌ Ошибка запуска сервера: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Неожиданная ошибка: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
