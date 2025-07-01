import { FFmpeg } from 'https://unpkg.com/@ffmpeg/ffmpeg@0.12.10/dist/esm/index.js';
import { toBlobURL } from 'https://unpkg.com/@ffmpeg/util@0.12.1/dist/esm/index.js';

class VideoPlayer {
    constructor() {
        this.ffmpeg = new FFmpeg();
        this.isLoaded = false;
        this.currentVideo = null;
        this.isPlaying = false;
        this.isMuted = false;
        this.currentSource = '';
        this.videoFormat = '';
        
        // DOM элементы
        this.videoPlayer = document.getElementById('videoPlayer');
        this.videoCanvas = document.getElementById('videoCanvas');
        this.canvasWrapper = document.getElementById('canvasWrapper');
        this.playPauseBtn = document.getElementById('playPauseBtn');
        this.stopBtn = document.getElementById('stopBtn');
        this.muteBtn = document.getElementById('muteBtn');
        this.frameBtn = document.getElementById('frameBtn');
        this.progressContainer = document.getElementById('progressContainer');
        this.progressFill = document.getElementById('progressFill');
        
        // Статус элементы
        this.playerStatus = document.getElementById('playerStatus');
        this.ffmpegStatus = document.getElementById('ffmpegStatus');
        this.videoFormatEl = document.getElementById('videoFormat');
        this.currentSourceEl = document.getElementById('currentSource');
        this.logOutput = document.getElementById('logOutput');
        
        // Инициализация
        this.init();
        this.setupEventListeners();
    }
    
    async init() {
        try {
            this.log('Инициализация FFmpeg WASM...', 'info');
            this.updateFFmpegStatus('Загрузка...');
            
            // Настраиваем event listeners для FFmpeg
            this.ffmpeg.on('log', ({ message }) => {
                this.log(`FFmpeg: ${message}`, 'info');
            });
            
            this.ffmpeg.on('progress', ({ progress }) => {
                if (progress > 0) {
                    this.updateProgress(progress * 100);
                }
            });
            
            // Пробуем различные стратегии инициализации
            await this.initWithMultipleStrategies();
            
        } catch (error) {
            this.log(`Критическая ошибка инициализации FFmpeg: ${error}`, 'error');
            this.updateFFmpegStatus('Критическая ошибка');
            
            // Пробуем запустить без FFmpeg (только для нативно поддерживаемых форматов)
            this.log('Переходим в режим "только нативные форматы"', 'warn');
            this.isLoaded = false;
            this.updateFFmpegStatus('Только нативные форматы');
        }
    }
    
    async initWithMultipleStrategies() {
        const strategies = [
            () => this.initWithMainCDN(),
            () => this.initWithoutWorker()
        ];
        
        for (let i = 0; i < strategies.length; i++) {
            try {
                this.log(`Пробуем стратегию инициализации ${i + 1}/${strategies.length}...`, 'info');
                await strategies[i]();
                this.isLoaded = true;
                this.updateFFmpegStatus('Готов');
                this.log('FFmpeg WASM успешно инициализирован', 'success');
                return;
            } catch (error) {
                this.log(`Стратегия ${i + 1} не удалась: ${error}`, 'warn');
                if (i === strategies.length - 1) {
                    throw error;
                }
            }
        }
    }
    
    async initWithMainCDN() {
        this.log('Попытка загрузки с основного CDN (unpkg.com)...', 'info');
        const baseURL = 'https://unpkg.com/@ffmpeg/core@0.12.10/dist/umd';
        
        await this.ffmpeg.load({
            coreURL: await toBlobURL(`${baseURL}/ffmpeg-core.js`, 'text/javascript'),
            wasmURL: await toBlobURL(`${baseURL}/ffmpeg-core.wasm`, 'application/wasm'),
            workerURL: await toBlobURL(`${baseURL}/ffmpeg-core.worker.js`, 'text/javascript'),
        });
    }
    
    
    
    async initWithoutWorker() {
        this.log('Попытка загрузки без worker (single-threaded версия)...', 'info');
        const baseURL = 'https://unpkg.com/@ffmpeg/core@0.12.10/dist/esm';
    
        await this.ffmpeg.load({
            coreURL: await toBlobURL(`${baseURL}/ffmpeg-core.js`, 'text/javascript'),
            wasmURL: await toBlobURL(`${baseURL}/ffmpeg-core.wasm`, 'application/wasm'),
        });
        
        this.log('FFmpeg WASM инициализирован в однопоточном режиме', 'success');
    }
    

    setupEventListeners() {
        // Обработчики для video элемента
        this.videoPlayer.addEventListener('loadedmetadata', () => {
            this.log('Метаданные видео загружены', 'info');
            this.enableControls();
            this.progressContainer.style.display = 'block';
        });
        
        this.videoPlayer.addEventListener('timeupdate', () => {
            if (this.videoPlayer.duration) {
                const progress = (this.videoPlayer.currentTime / this.videoPlayer.duration) * 100;
                this.updateProgress(progress);
            }
        });
        
        this.videoPlayer.addEventListener('ended', () => {
            this.isPlaying = false;
            this.updatePlayPauseBtn();
            this.log('Воспроизведение завершено', 'info');
        });
        
        this.videoPlayer.addEventListener('error', (e) => {
            this.log(`Ошибка воспроизведения: ${e.message}`, 'error');
        });
        
      
        // Обработчик для URL - автоматическая загрузка при вводе
        const videoUrlInput = document.getElementById('videoUrl');
        if (videoUrlInput) {
            let timeoutId;
            videoUrlInput.addEventListener('input', () => {
                // Очищаем предыдущий таймер
                clearTimeout(timeoutId);
                
                // Устанавливаем новый таймер с задержкой 1 секунда
                timeoutId = setTimeout(() => {
                    const url = videoUrlInput.value.trim();
                    if (url && this.isValidUrl(url)) {
                        this.loadVideo();
                    }
                }, 1000);
            });
        }
    }
    
    async loadVideo() {
        this.updatePlayerStatus('Загрузка...');
        this.disableControls();
        
        const videoUrl = document.getElementById('videoUrl').value.trim();
        
        if (!videoUrl) {
            this.log('Введите URL видео', 'warn');
            this.updatePlayerStatus('Ошибка: URL не указан');
            return;
        }
        
        try {
            await this.loadFromUrl(videoUrl);
        } catch (error) {
            this.log(`Ошибка загрузки видео: ${error}`, 'error');
            this.updatePlayerStatus('Ошибка загрузки');
        }
    }
    

    
    async loadFromUrl(url) {
        this.log(`Загрузка по URL: ${url}`, 'info');
        this.currentSource = url;
        this.updateCurrentSource(this.currentSource);
        
        try {
            // Попробуем загрузить напрямую
            await this.loadDirectly(url);
            
            // Определяем формат по URL
            const urlParts = url.split('.');
            const extension = urlParts[urlParts.length - 1].split('?')[0].toLowerCase();
            this.videoFormat = extension.toUpperCase();
            this.updateVideoFormat(this.videoFormat);
            
        } catch (error) {
            this.log('Прямая загрузка не удалась, пробуем через FFmpeg', 'warn');
            
            // Проверяем доступность FFmpeg для конвертации
            if (!this.isLoaded) {
                const extension = url.split('.').pop().split('?')[0].toLowerCase();
                if (this.isNativelySupported(extension)) {
                    throw new Error(`Не удалось загрузить ${extension.toUpperCase()} файл. Возможно проблема с CORS или доступностью файла.`);
                } else {
                    throw new Error(`Формат ${extension.toUpperCase()} не поддерживается напрямую браузером, а FFmpeg WASM не загружен. Попробуйте перезагрузить страницу или используйте поддерживаемые форматы: MP4, WebM, OGG.`);
                }
            }
            
            try {
                
                await this.convertAndLoad(url);
            } catch (fetchError) {
                if (fetchError.name === 'TypeError' && fetchError.message.includes('CORS')) {
                    throw new Error('CORS ошибка: Убедитесь что открываете приложение через http://localhost:8000, а не из файловой системы');
                } else if (fetchError.message.includes('cannot be accessed from origin')) {
                    throw new Error('CORS ошибка: Запустите CORS сервер (python3 cors_server.py) и откройте http://localhost:8000');
                }
                throw fetchError;
            }
        }
    }
    
    async loadDirectly(url) {
        return new Promise((resolve, reject) => {
            const tempVideo = document.createElement('video');
            tempVideo.addEventListener('loadedmetadata', () => {
                this.videoPlayer.src = url;
                this.updatePlayerStatus('Готов к воспроизведению');
                this.log('Видео успешно загружено', 'success');
                resolve();
            });
            tempVideo.addEventListener('error', reject);
            tempVideo.src = url;
        });
    }
    
    async convertAndLoad(videoURL) {
        this.log(`Начинаем конвертацию через FFmpeg... ${videoURL}`, 'info');
        this.updatePlayerStatus('Конвертация...');
        
        // Проверяем, что FFmpeg загружен
        if (!this.isLoaded) {
            throw new Error('FFmpeg WASM не загружен. Попробуйте перезагрузить страницу или используйте нативно поддерживаемые форматы (MP4, WebM).');
        }
        
        const outputFile = 'output.webm';
        
        try {
            
            // Конвертируем в WebM (широко поддерживается)
            this.log('Начинаем конвертацию...', 'info');
            await this.ffmpeg.exec([
                
                '-protocol_whitelist','file,http,https,tcp,tls',
                '-i', videoURL,
                '-c:v', 'libvpx-vp9',
                '-c:a', 'libopus',
                '-crf', '30',
                '-b:v', '1M',
                '-f', 'webm',
                outputFile
            ]);
            
            // Проверяем, что выходной файл создан
            this.log('Читаем результат конвертации...', 'info');
            const data = await this.ffmpeg.readFile(outputFile);
            
            if (!data || data.length === 0) {
                throw new Error('Конвертация не создала выходной файл');
            }
            
            const blob = new Blob([data], { type: 'video/webm' });
            const url = URL.createObjectURL(blob);
            
            await this.loadDirectly(url);
            
            this.log('Конвертация завершена успешно', 'success');
            
        } catch (error) {
            this.log(`Детали ошибки конвертации: ${error}`, 'error');
            
            if (error.message && error.message.includes('FS error')) {
                throw new Error('Ошибка файловой системы FFmpeg. Попробуйте перезагрузить страницу или используйте файл меньшего размера.');
            } else if (error.message && error.message.includes('out of memory')) {
                throw new Error('Недостаточно памяти для конвертации. Попробуйте файл меньшего размера.');
            } else {
                throw new Error(`Ошибка конвертации: ${error.message || error}`);
            }
        } finally {
            // Очищаем временные файлы (независимо от результата)
            try {
                await this.ffmpeg.deleteFile(filename);
                this.log(`Удален временный файл: ${filename}`, 'info');
            } catch (e) {
                this.log(`Не удалось удалить файл ${filename}: ${e}`, 'warn');
            }
            
            try {
                await this.ffmpeg.deleteFile(outputFile);
                this.log(`Удален выходной файл: ${outputFile}`, 'info');
            } catch (e) {
                this.log(`Не удалось удалить файл ${outputFile}: ${e}`, 'warn');
            }
        }
    }
    
    isNativelySupported(extension) {
        const supported = ['mp4', 'webm', 'ogg', 'ogv', 'm4v', 'mov', 'avi'];
        return supported.includes(extension.toLowerCase());
    }
    
    isValidUrl(string) {
        try {
            const url = new URL(string);
            return url.protocol === 'http:' || url.protocol === 'https:';
        } catch (_) {
            return false;
        }
    }
    
    playPause() {
        this.log(`DEBUG: playPause вызвана. src=${this.videoPlayer.src}, disabled=${this.playPauseBtn.disabled}`, 'info');
        
        if (!this.videoPlayer.src) {
            this.log('Сначала загрузите видео', 'warn');
            return;
        }
        
        if (this.videoPlayer.paused) {
            this.videoPlayer.play().then(() => {
                this.isPlaying = true;
                this.log('Воспроизведение начато', 'info');
                this.updatePlayPauseBtn();
            }).catch(error => {
                this.log(`Ошибка воспроизведения: ${error.message}`, 'error');
            });
        } else {
            this.videoPlayer.pause();
            this.isPlaying = false;
            this.log('Воспроизведение приостановлено', 'info');
            this.updatePlayPauseBtn();
        }
    }
    
    stopVideo() {
        if (!this.videoPlayer.src) return;
        
        this.videoPlayer.pause();
        this.videoPlayer.currentTime = 0;
        this.isPlaying = false;
        this.updatePlayPauseBtn();
        this.updateProgress(0);
        this.log('Воспроизведение остановлено', 'info');
    }
    
    toggleMute() {
        if (!this.videoPlayer.src) return;
        
        this.videoPlayer.muted = !this.videoPlayer.muted;
        this.isMuted = this.videoPlayer.muted;
        
        const muteBtn = document.getElementById('muteBtn');
        muteBtn.textContent = this.isMuted ? '🔊 Включить звук' : '🔇 Выключить звук';
        
        this.log(this.isMuted ? 'Звук выключен' : 'Звук включен', 'info');
    }
    
    async extractFrame() {
        if (!this.videoPlayer.src) {
            this.log('Сначала загрузите видео', 'warn');
            return;
        }
        
        try {
            // Создаем canvas для захвата кадра
            const canvas = document.createElement('canvas');
            const ctx = canvas.getContext('2d');
            
            canvas.width = this.videoPlayer.videoWidth;
            canvas.height = this.videoPlayer.videoHeight;
            
            ctx.drawImage(this.videoPlayer, 0, 0, canvas.width, canvas.height);
            
            // Показываем кадр
            const frameCanvas = document.getElementById('videoCanvas');
            const frameCtx = frameCanvas.getContext('2d');
            
            frameCanvas.width = canvas.width;
            frameCanvas.height = canvas.height;
            
            frameCtx.drawImage(canvas, 0, 0);
            
            this.canvasWrapper.style.display = 'block';
            
            // Создаем ссылку для скачивания
            canvas.toBlob((blob) => {
                const url = URL.createObjectURL(blob);
                const a = document.createElement('a');
                a.href = url;
                a.download = `frame_${Date.now()}.png`;
                a.click();
                URL.revokeObjectURL(url);
            });
            
            this.log('Кадр извлечен и сохранен', 'success');
            
        } catch (error) {
            this.log(`Ошибка извлечения кадра: ${error.message}`, 'error');
        }
    }
    
    // Утилиты для UI
    enableControls() {
        this.playPauseBtn.disabled = false;
        this.stopBtn.disabled = false;
        this.muteBtn.disabled = false;
        this.frameBtn.disabled = false;
    }
    
    disableControls() {
        this.playPauseBtn.disabled = true;
        this.stopBtn.disabled = true;
        this.muteBtn.disabled = true;
        this.frameBtn.disabled = true;
    }
    
    updatePlayPauseBtn() {
        this.playPauseBtn.textContent = this.isPlaying ? '⏸️ Пауза' : '▶️ Воспроизвести';
    }
    
    updateProgress(percentage) {
        this.progressFill.style.width = `${Math.max(0, Math.min(100, percentage))}%`;
    }
    
    updatePlayerStatus(status) {
        this.playerStatus.textContent = status;
    }
    
    updateFFmpegStatus(status) {
        this.ffmpegStatus.textContent = status;
        
        // Добавляем индикатор статуса и кнопку повторной попытки
        const existingRetryBtn = document.getElementById('ffmpegRetryBtn');
        if (existingRetryBtn) {
            existingRetryBtn.remove();
        }
        
        if (status === 'Критическая ошибка' || status === 'Только нативные форматы') {
            const retryBtn = document.createElement('button');
            retryBtn.id = 'ffmpegRetryBtn';
            retryBtn.className = 'btn btn-primary btn-small';
            retryBtn.textContent = '🔄 Повторить загрузку FFmpeg';
            retryBtn.style.marginLeft = '10px';
            retryBtn.onclick = () => this.retryFFmpegInit();
            
            this.ffmpegStatus.parentNode.appendChild(retryBtn);
        }
    }
    
    async retryFFmpegInit() {
        this.log('Повторная попытка инициализации FFmpeg...', 'info');
        this.updateFFmpegStatus('Повторная загрузка...');
        
        // Сбрасываем состояние
        this.isLoaded = false;
        this.ffmpeg = new FFmpeg();
        
        // Настраиваем event listeners заново
        this.ffmpeg.on('log', ({ message }) => {
            this.log(`FFmpeg: ${message}`, 'info');
        });
        
        this.ffmpeg.on('progress', ({ progress }) => {
            if (progress > 0) {
                this.updateProgress(progress * 100);
            }
        });
        
        try {
            await this.initWithMultipleStrategies();
        } catch (error) {
            this.log(`Повторная инициализация не удалась: ${error}`, 'error');
            this.updateFFmpegStatus('Критическая ошибка');
        }
    }
    
    updateVideoFormat(format) {
        this.videoFormatEl.textContent = format;
    }
    
    updateCurrentSource(source) {
        this.currentSourceEl.textContent = source.length > 50 ? 
            source.substring(0, 47) + '...' : source;
    }
    
    log(message, level = 'info') {
        const timestamp = new Date().toLocaleTimeString();
        const logEntry = document.createElement('div');
        logEntry.className = 'log-entry';
        
        const levelClass = `log-level-${level}`;
        logEntry.innerHTML = `
            <span class="log-timestamp">[${timestamp}]</span>
            <span class="${levelClass}">[${level.toUpperCase()}]</span>
            ${message}
        `;
        
        this.logOutput.appendChild(logEntry);
        this.logOutput.scrollTop = this.logOutput.scrollHeight;
        
        // Ограничиваем количество логов
        while (this.logOutput.children.length > 100) {
            this.logOutput.removeChild(this.logOutput.firstChild);
        }
    }
    
    clearLogs() {
        this.logOutput.innerHTML = '';
        this.log('Логи очищены', 'info');
    }
}

// Глобальные функции для HTML
let player;

// Инициализация при загрузке страницы
window.addEventListener('DOMContentLoaded', () => {
    player = new VideoPlayer();
});

// Экспорт функций для использования в HTML
window.playPause = () => player?.playPause();
window.stopVideo = () => player?.stopVideo();
window.toggleMute = () => player?.toggleMute();
window.extractFrame = () => player?.extractFrame();
window.clearLogs = () => player?.clearLogs(); 
