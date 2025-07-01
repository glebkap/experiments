class H265HLSPlayer {
    constructor() {
        this.hls = null;
        this.goldPlayer = null;
        this.useGoldPlayer = false;
        this.isLoaded = false;
        this.isPlaying = false;
        this.isMuted = false;
        this.currentSource = '';
        this.videoFormat = '';
        this.hasH265Support = false;
        this.hasGoldPlayerSupport = false;
        
        // DOM элементы
        this.videoPlayer = document.getElementById('videoPlayer');
        this.playPauseBtn = document.getElementById('playPauseBtn');
        this.stopBtn = document.getElementById('stopBtn');
        this.muteBtn = document.getElementById('muteBtn');
        this.frameBtn = document.getElementById('frameBtn');
        this.progressContainer = document.getElementById('progressContainer');
        this.progressFill = document.getElementById('progressFill');
        
        // Статус элементы
        this.playerStatus = document.getElementById('playerStatus');
        this.decoderStatus = document.getElementById('decoderStatus');
        this.videoFormatEl = document.getElementById('videoFormat');
        this.currentSourceEl = document.getElementById('currentSource');
        this.logOutput = document.getElementById('logOutput');
        
        // Инициализация
        this.init();
        this.setupEventListeners();
    }
    
    async init() {
        try {
            this.log('Инициализация H.265 HLS плеера...', 'info');
            this.updateDecoderStatus('Проверка поддержки...');
            
            // Проверяем нативную поддержку H.265
            this.checkH265Support();
            
            // Проверяем поддержку goldvideo player
            this.checkGoldPlayerSupport();
            
            // Проверяем поддержку HLS.js
            if (Hls.isSupported()) {
                this.log('HLS.js поддерживается', 'success');
                this.initHLS();
            } else if (this.videoPlayer.canPlayType('application/vnd.apple.mpegurl')) {
                this.log('Нативная поддержка HLS в Safari', 'info');
            } else {
                throw new Error('HLS не поддерживается в этом браузере');
            }
            
            this.isLoaded = true;
            
            // Определяем статус декодера
            let decoderStatus = '';
            if (this.hasH265Support) {
                decoderStatus = 'H.265 поддерживается (нативно)';
            } else if (this.hasGoldPlayerSupport) {
                decoderStatus = 'H.265 поддерживается (goldvideo)';
            } else {
                decoderStatus = 'Только H.264';
            }
            
            this.updateDecoderStatus(decoderStatus);
            this.log('H.265 HLS плеер успешно инициализирован', 'success');
            
        } catch (error) {
            this.log(`Критическая ошибка инициализации: ${error}`, 'error');
            this.updateDecoderStatus('Ошибка');
        }
    }
    
    checkH265Support() {
        const video = document.createElement('video');
        
        // Проверяем различные форматы H.265/HEVC
        const h265Codecs = [
            'video/mp4; codecs="hev1.1.6.L93.B0"',
            'video/mp4; codecs="hvc1.1.6.L93.B0"',
            'video/mp4; codecs="hev1"',
            'video/mp4; codecs="hvc1"'
        ];
        
        for (const codec of h265Codecs) {
            const support = video.canPlayType(codec);
            if (support === 'probably' || support === 'maybe') {
                this.hasH265Support = true;
                this.log(`H.265 поддержка обнаружена: ${codec} (${support})`, 'success');
                break;
            }
        }
        
        if (!this.hasH265Support) {
            this.log('Нативная поддержка H.265 не найдена', 'warn');
            this.log('Плеер будет работать с H.264 контентом', 'info');
        }
    }
    
    checkGoldPlayerSupport() {
        try {
            // Проверяем доступность goldvideo player
            if (typeof GoldPlay !== 'undefined') {
                this.hasGoldPlayerSupport = true;
                this.log('goldvideo H.265 player доступен как fallback', 'success');
                
                // Debuging: log what's available in GoldPlay
                this.log('GoldPlay object keys: ' + Object.keys(GoldPlay).join(', '), 'info');
                if (GoldPlay.Events) {
                    this.log('GoldPlay.Events keys: ' + Object.keys(GoldPlay.Events).join(', '), 'info');
                } else {
                    this.log('GoldPlay.Events is undefined', 'warn');
                    // Check if events are defined differently
                    if (GoldPlay.prototype && GoldPlay.prototype.Events) {
                        this.log('Found Events in prototype: ' + Object.keys(GoldPlay.prototype.Events).join(', '), 'info');
                    }
                }
                
                // Если нет нативной поддержки H.265, используем goldvideo как fallback
                if (!this.hasH265Support) {
                    this.useGoldPlayer = true;
                    this.log('Будет использоваться goldvideo player для H.265 контента', 'info');
                }
            } else {
                this.log('goldvideo player не найден', 'warn');
                this.hasGoldPlayerSupport = false;
            }
        } catch (error) {
            this.log(`Ошибка проверки goldvideo player: ${error}`, 'error');
            this.hasGoldPlayerSupport = false;
        }
    }
    
    initHLS() {
        this.hls = new Hls({
            debug: false,
            enableWorker: true,
            lowLatencyMode: false,
            backBufferLength: 90,
            maxBufferLength: 30,
            maxMaxBufferLength: 600,
            capLevelToPlayerSize: true,
            startPosition: -1,
            enableSoftwareAES: true
        });
        
        this.setupHLSEvents();
        this.log('HLS.js инициализирован', 'success');
    }
    
    initGoldPlayer() {
        try {
            if (!this.hasGoldPlayerSupport) {
                throw new Error('goldvideo player не доступен');
            }
            
            // Скрываем стандартный video элемент
            this.videoPlayer.style.display = 'none';
            
            // Показываем canvas для goldvideo player
            const canvasWrapper = document.getElementById('canvasWrapper');
            const canvas = document.getElementById('videoCanvas');
            canvasWrapper.style.display = 'block';
            
            this.log('goldvideo player инициализирован', 'success');
            return true;
            
        } catch (error) {
            this.log(`Ошибка инициализации goldvideo player: ${error}`, 'error');
            return false;
        }
    }
    
    setupHLSEvents() {
        if (!this.hls) return;
        
        this.hls.on(Hls.Events.MEDIA_ATTACHED, () => {
            this.log('HLS медиа прикреплено', 'info');
        });
        
        this.hls.on(Hls.Events.MANIFEST_PARSED, (event, data) => {
            this.log(`HLS манифест загружен. Уровней: ${data.levels.length}`, 'info');
            
            // Анализируем доступные кодеки
            const levels = data.levels;
            let hasH265 = false;
            let hasH264 = false;
            
            levels.forEach((level, index) => {
                const videoCodec = level.videoCodec ? level.videoCodec.toLowerCase() : '';
                this.log(`Уровень ${index}: ${level.width}x${level.height}, битрейт: ${level.bitrate}, кодек: ${level.videoCodec}`, 'info');
                
                if (videoCodec.includes('hvc1') || videoCodec.includes('hev1')) {
                    hasH265 = true;
                } else if (videoCodec.includes('avc1')) {
                    hasH264 = true;
                }
            });
            
            // Определяем стратегию воспроизведения
            if (hasH265 && this.hasH265Support) {
                this.videoFormat = 'HLS (H.265 - поддерживается)';
                this.log('Используется H.265 с нативной поддержкой браузера', 'success');
            } else if (hasH265 && !this.hasH265Support) {
                this.videoFormat = 'HLS (H.265 - ограниченная поддержка)';
                this.log('H.265 контент - возможны проблемы воспроизведения', 'warn');
            } else if (hasH264) {
                this.videoFormat = 'HLS (H.264)';
                this.log('Используется H.264 контент', 'info');
            } else {
                this.videoFormat = 'HLS (неизвестный кодек)';
            }
            
            this.updateVideoFormat(this.videoFormat);
            this.enableControls();
        });
        
        this.hls.on(Hls.Events.LEVEL_SWITCHED, (event, data) => {
            const level = this.hls.levels[data.level];
            this.log(`Переключен на уровень ${data.level}: ${level.width}x${level.height}`, 'info');
        });
        
        this.hls.on(Hls.Events.FRAG_LOADED, (event, data) => {
            this.log(`Фрагмент загружен: #${data.frag.sn}`, 'info');
        });
        
        // Обработка ошибок HLS
        this.hls.on(Hls.Events.ERROR, (event, data) => {
            this.log(`HLS ошибка: ${data.type} - ${data.details}`, data.fatal ? 'error' : 'warn');
            
            if (data.fatal) {
                switch(data.type) {
                    case Hls.ErrorTypes.NETWORK_ERROR:
                        this.log('Критическая сетевая ошибка, попытка восстановления...', 'error');
                        this.hls.startLoad();
                        break;
                        
                    case Hls.ErrorTypes.MEDIA_ERROR:
                        this.log('Критическая медиа ошибка, попытка восстановления...', 'error');
                        
                        // Обработка ошибок кодеков
                        if (data.details === 'bufferAddCodecError' || 
                            data.details === 'bufferIncompatibleCodecsError' ||
                            data.details === 'bufferAppendError') {
                            
                            this.handleCodecError(data);
                        } else {
                            this.hls.recoverMediaError();
                        }
                        break;
                        
                    default:
                        this.log(`Критическая ошибка HLS: ${data.details}`, 'error');
                        this.handleFatalError();
                        break;
                }
            }
        });
    }
    
    handleCodecError(errorData) {
        this.log('Обнаружена ошибка кодека - возможно H.265 не поддерживается', 'error');
        
        if (!this.hasH265Support) {
            this.log('H.265 не поддерживается этим браузером', 'error');
            this.updatePlayerStatus('Ошибка: H.265 не поддерживается');
            this.suggestAlternatives();
        } else {
            this.log('Попытка восстановления...', 'info');
            this.hls.recoverMediaError();
        }
    }
    
    suggestAlternatives() {
        this.log('Рекомендации для решения проблемы:', 'info');
        this.log('1. Попробуйте H.264 HLS поток', 'info');
        this.log('2. Используйте Safari для лучшей поддержки H.265', 'info');
        this.log('3. Проверьте что манифест содержит совместимые кодеки', 'info');
        
        // Примеры рабочих потоков
        const examples = [
            'https://test-streams.mux.dev/x36xhzz/x36xhzz.m3u8 (H.264)',
            'https://bitdash-a.akamaihd.net/content/sintel/hls/playlist.m3u8 (H.264)'
        ];
        
        this.log('Примеры совместимых потоков:', 'info');
        examples.forEach(url => this.log(url, 'info'));
    }
    
    handleFatalError() {
        this.log('Критическая ошибка HLS', 'error');
        this.updatePlayerStatus('Критическая ошибка');
        this.disableControls();
        
        if (this.hls) {
            this.hls.destroy();
            this.hls = null;
        }
    }

    setupEventListeners() {
        // Обработчики video элемента
        this.videoPlayer.addEventListener('loadedmetadata', () => {
            this.log('Метаданные видео загружены', 'info');
            this.log(`Разрешение: ${this.videoPlayer.videoWidth}x${this.videoPlayer.videoHeight}`, 'info');
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
            const error = e.target.error;
            if (error) {
                this.log(`Ошибка video: код ${error.code}`, 'error');
                
                switch(error.code) {
                    case MediaError.MEDIA_ERR_DECODE:
                        this.log('Ошибка декодирования - возможно неподдерживаемый кодек', 'error');
                        this.suggestAlternatives();
                        break;
                    case MediaError.MEDIA_ERR_SRC_NOT_SUPPORTED:
                        this.log('Источник не поддерживается', 'error');
                        this.suggestAlternatives();
                        break;
                    case MediaError.MEDIA_ERR_NETWORK:
                        this.log('Сетевая ошибка', 'error');
                        break;
                }
            }
        });
        
        // Автозагрузка при вводе URL
        const videoUrlInput = document.getElementById('videoUrl');
        if (videoUrlInput) {
            let timeoutId;
            videoUrlInput.addEventListener('input', () => {
                clearTimeout(timeoutId);
                timeoutId = setTimeout(() => {
                    const url = videoUrlInput.value.trim();
                    if (url && this.isValidUrl(url)) {
                        this.loadVideo();
                    }
                }, 1000);
            });
            
            // Добавляем пример URL
            if (!videoUrlInput.value) {
                videoUrlInput.placeholder = 'https://test-streams.mux.dev/x36xhzz/x36xhzz.m3u8';
            }
        }
    }
    
    async loadVideo() {
        this.updatePlayerStatus('Загрузка...');
        this.disableControls();
        
        const videoUrl = document.getElementById('videoUrl').value.trim();
        
        if (!videoUrl) {
            this.log('Введите URL HLS плейлиста', 'warn');
            this.updatePlayerStatus('Ошибка: URL не указан');
            return;
        }

        try {
            await this.loadHLSStream(videoUrl);
        } catch (error) {
            this.log(`Ошибка загрузки: ${error}`, 'error');
            this.updatePlayerStatus('Ошибка загрузки');
        }
    }
    
    async loadHLSStream(url) {
        this.log(`Загрузка HLS: ${url}`, 'info');
        this.currentSource = url;
        this.updateCurrentSource(url);
        
        try {
            // Сначала проверяем через HLS.js какие кодеки доступны
            if (this.useGoldPlayer && this.hasGoldPlayerSupport) {
                await this.loadWithGoldPlayer(url);
            } else if (Hls.isSupported() && this.hls) {
                // Очищаем предыдущий источник
                this.hls.destroy();
                this.initHLS();
                
                this.hls.loadSource(url);
                this.hls.attachMedia(this.videoPlayer);
                
                this.hls.on(Hls.Events.MANIFEST_PARSED, (event, data) => {
                    this.log('Манифест загружен успешно', 'success');
                    
                    // Проверяем наличие H.265 контента
                    const hasH265Content = this.checkManifestForH265(data);
                    
                    if (hasH265Content && !this.hasH265Support && this.hasGoldPlayerSupport) {
                        this.log('Обнаружен H.265 контент, переключаемся на goldvideo player', 'info');
                        this.hls.destroy();
                        this.loadWithGoldPlayer(url);
                        return;
                    }
                    
                    this.updatePlayerStatus('Готов к воспроизведению');
                });
                
            } else if (this.videoPlayer.canPlayType('application/vnd.apple.mpegurl')) {
                // Нативная поддержка Safari
                this.videoPlayer.src = url;
                this.updatePlayerStatus('Готов (нативная поддержка)');
                this.log('Используется нативная поддержка HLS', 'info');
            }
            
        } catch (error) {
            this.log(`Ошибка загрузки HLS: ${error}`, 'error');
            this.updatePlayerStatus('Ошибка загрузки');
        }
    }
    
    checkManifestForH265(data) {
        const levels = data.levels;
        let hasH265 = false;
        
        levels.forEach(level => {
            const videoCodec = level.videoCodec ? level.videoCodec.toLowerCase() : '';
            if (videoCodec.includes('hvc1') || videoCodec.includes('hev1')) {
                hasH265 = true;
            }
        });
        
        return hasH265;
    }
    
    async loadWithGoldPlayer(url) {
        try {
            this.log(`Загрузка с помощью goldvideo player... ${url}`, 'info');
            
            if (!this.initGoldPlayer()) {
                throw new Error('Не удалось инициализировать goldvideo player');
            }
            
            // Уничтожаем предыдущий экземпляр если есть
            if (this.goldPlayer) {
                this.goldPlayer.destroy();
            }
            
            const canvas = document.getElementById('videoCanvas');
            const audioElement = this.videoPlayer; // Используем существующий audio element
            
            // Создаем goldvideo player
            this.goldPlayer = new GoldPlay(canvas.parentElement, {
                sourceURL: url,
                type: 'HLS',
                libPath: 'https://goldvideo.github.io/h265player/dist/lib',
                enableSkipFrame: false,
                bufferTime: 0,
                isShowStatistics: false
            });
            this.log('goldvideo player создан', 'success');
            // Настраиваем события goldvideo player
            this.setupGoldPlayerEvents();
            this.log('goldvideo player события настроены', 'success');

            this.updatePlayerStatus('Готов к воспроизведению (goldvideo)');
            this.updateVideoFormat('HLS (H.265 - goldvideo player)');
            this.log('goldvideo player загружен успешно', 'success');
            
        } catch (error) {
            this.log(`Ошибка загрузки goldvideo player: ${error}`, 'error');
            this.updatePlayerStatus('Ошибка загрузки goldvideo player');
            
            // Fallback к обычному HLS.js
            this.useGoldPlayer = false;
            this.loadHLSStream(url);
        }
    }
    
    setupGoldPlayerEvents() {
        if (!this.goldPlayer) return;

        this.log('goldvideo player события настраиваются', 'info');
        // Настраиваем события goldvideo player
        // Use string event names instead of GoldPlay.Events
        
        // Событие готовности плеера
        this.goldPlayer.on('ready', () => {
            this.log('goldvideo player готов', 'success');
            this.enableControls();
        });
        this.log('goldvideo player событие готовности настроено', 'success');
        
        // Событие начала воспроизведения
        this.goldPlayer.on('play', () => {
            this.isPlaying = true;
            this.updatePlayPauseBtn();
            this.log('goldvideo воспроизведение запущено', 'info');
        });
        this.log('goldvideo player событие воспроизведения настроено', 'success');
        
        // Событие паузы
        this.goldPlayer.on('pause', () => {
            this.isPlaying = false;
            this.updatePlayPauseBtn();
            this.log('goldvideo пауза', 'info');
        });
        this.log('goldvideo player событие паузы настроено', 'success');
        
        // Событие загрузки
        this.goldPlayer.on('loadstart', () => {
            this.log('goldvideo начало загрузки', 'info');
        });
        this.log('goldvideo player событие загрузки настроено', 'success');
        
        // Событие ошибки
        this.goldPlayer.on('error', (error) => {
            this.log(`goldvideo player ошибка: ${error}`, 'error');
        });
        this.log('goldvideo player событие ошибки настроено', 'success');
        
        // Событие информации о медиа
        this.goldPlayer.on('loadedmetadata', (event, data) => {
            this.log(`goldvideo медиа информация: ${JSON.stringify(data)}`, 'info');
        });
        this.log('goldvideo player событие информации о медиа настроено', 'success');
    }
    
    playPause() {
        if (this.useGoldPlayer && this.goldPlayer) {
            // Используем goldvideo player
            if (this.isPlaying) {
                this.goldPlayer.pause();
                this.isPlaying = false;
                this.log('Пауза (goldvideo)', 'info');
            } else {
                this.goldPlayer.play();
                this.isPlaying = true;
                this.log('Воспроизведение начато (goldvideo)', 'success');
            }
        } else {
            // Используем стандартный video элемент
            if (!this.videoPlayer.src && !this.hls) {
                this.log('Сначала загрузите видео', 'warn');
                return;
            }
            
            if (this.isPlaying) {
                this.videoPlayer.pause();
                this.isPlaying = false;
                this.log('Пауза', 'info');
            } else {
                this.videoPlayer.play().then(() => {
                    this.isPlaying = true;
                    this.log('Воспроизведение начато', 'success');
                }).catch(error => {
                    this.log(`Ошибка воспроизведения: ${error.message}`, 'error');
                    this.isPlaying = false;
                });
            }
        }
        
        this.updatePlayPauseBtn();
    }
    
    stopVideo() {
        if (this.useGoldPlayer && this.goldPlayer) {
            // Используем goldvideo player
            this.goldPlayer.pause();
            this.goldPlayer.currentTime = 0;
            this.isPlaying = false;
            this.log('Остановлено (goldvideo)', 'info');
        } else {
            // Используем стандартный video элемент
            this.videoPlayer.pause();
            this.videoPlayer.currentTime = 0;
            this.isPlaying = false;
            this.log('Остановлено', 'info');
        }
        
        this.updatePlayPauseBtn();
        this.updateProgress(0);
    }
    
    toggleMute() {
        this.isMuted = !this.isMuted;
        
        if (this.useGoldPlayer && this.goldPlayer) {
            // Используем goldvideo player
            this.goldPlayer.muted = this.isMuted;
            this.log(`Звук ${this.isMuted ? 'выключен' : 'включен'} (goldvideo)`, 'info');
        } else {
            // Используем стандартный video элемент
            this.videoPlayer.muted = this.isMuted;
            this.log(`Звук ${this.isMuted ? 'выключен' : 'включен'}`, 'info');
        }
        
        this.muteBtn.textContent = this.isMuted ? '🔇 Вкл звук' : '🔇 Выкл звук';
    }
    
    async extractFrame() {
        try {
            let sourceCanvas, sourceWidth, sourceHeight;
            
            if (this.useGoldPlayer && this.goldPlayer) {
                // Используем canvas из goldvideo player
                sourceCanvas = document.getElementById('videoCanvas');
                if (!sourceCanvas || sourceCanvas.width === 0) {
                    this.log('goldvideo canvas не готов', 'warn');
                    return;
                }
                sourceWidth = sourceCanvas.width;
                sourceHeight = sourceCanvas.height;
                this.log('Извлечение кадра из goldvideo player', 'info');
            } else {
                // Используем стандартный video элемент
                if (!this.videoPlayer.videoWidth) {
                    this.log('Видео не загружено', 'warn');
                    return;
                }
                sourceWidth = this.videoPlayer.videoWidth;
                sourceHeight = this.videoPlayer.videoHeight;
                this.log('Извлечение кадра из video элемента', 'info');
            }
            
            const canvas = document.createElement('canvas');
            const ctx = canvas.getContext('2d');
            
            canvas.width = sourceWidth;
            canvas.height = sourceHeight;
            
            if (this.useGoldPlayer && this.goldPlayer) {
                // Копируем данные из canvas goldvideo player
                const sourceCtx = sourceCanvas.getContext('2d');
                const imageData = sourceCtx.getImageData(0, 0, sourceWidth, sourceHeight);
                ctx.putImageData(imageData, 0, 0);
            } else {
                // Рисуем из video элемента
                ctx.drawImage(this.videoPlayer, 0, 0);
            }
            
            canvas.toBlob((blob) => {
                const url = URL.createObjectURL(blob);
                const a = document.createElement('a');
                a.href = url;
                a.download = `frame_${Date.now()}.png`;
                a.click();
                URL.revokeObjectURL(url);
                this.log('Кадр сохранен', 'success');
            });
            
        } catch (error) {
            this.log(`Ошибка сохранения кадра: ${error}`, 'error');
        }
    }
    
    isValidUrl(string) {
        try {
            new URL(string);
            return true;
        } catch (_) {
            return false;
        }
    }
    
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
        this.progressFill.style.width = `${percentage}%`;
    }
    
    updatePlayerStatus(status) {
        this.playerStatus.textContent = status;
    }
    
    updateDecoderStatus(status) {
        this.decoderStatus.textContent = status;
    }
    
    updateVideoFormat(format) {
        this.videoFormatEl.textContent = format;
    }
    
    updateCurrentSource(source) {
        const maxLength = 50;
        const displaySource = source.length > maxLength ? 
            source.substring(0, maxLength) + '...' : source;
        this.currentSourceEl.textContent = displaySource;
    }
    
    log(message, level = 'info') {
        const timestamp = new Date().toLocaleTimeString();
        const logEntry = document.createElement('div');
        logEntry.className = 'log-entry';
        
        const timestampSpan = document.createElement('span');
        timestampSpan.className = 'log-timestamp';
        timestampSpan.textContent = `[${timestamp}] `;
        
        const messageSpan = document.createElement('span');
        messageSpan.className = `log-level-${level}`;
        messageSpan.textContent = message;
        
        logEntry.appendChild(timestampSpan);
        logEntry.appendChild(messageSpan);
        
        this.logOutput.appendChild(logEntry);
        this.logOutput.scrollTop = this.logOutput.scrollHeight;
        
        // Ограничиваем количество логов
        while (this.logOutput.children.length > 100) {
            this.logOutput.removeChild(this.logOutput.firstChild);
        }
        
        console.log(`[${level.toUpperCase()}] ${message}`);
    }
    
    clearLogs() {
        this.logOutput.innerHTML = '';
        this.log('Логи очищены', 'info');
    }
}

// Создаем экземпляр плеера
let player;

// Инициализация при загрузке DOM
document.addEventListener('DOMContentLoaded', () => {
    player = new H265HLSPlayer();
});

// Глобальные функции для HTML кнопок
function playPause() {
    if (player) player.playPause();
}

function stopVideo() {
    if (player) player.stopVideo();
}

function toggleMute() {
    if (player) player.toggleMute();
}

function extractFrame() {
    if (player) player.extractFrame();
}

function clearLogs() {
    if (player) player.clearLogs();
} 
