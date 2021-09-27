# test-kelly

## Задание
**Регрессионная задача**. Изначально требуется провести анализ всех имеющихся признаков (признаки, начинающиеся на: **NUM** - вещественные; **CAT** - категориальные/бинарные) и выборки в целом. Процесс анализа полностью по пунктам описать. После обработки данных требуется обучить ML модель на обучающей выборке (см. поле “Разбивка” в файле) и предсказать целевую по данным тестовой выборки (см. поле “Разбивка”), целевая — **TARGET**. Данные полностью обезличены, так что придется ориентироваться только на статистики и связь с целевой. Особое внимание требуется уделить категориальным данным и их анализу с подробным описанием. Выборка заранее разбита нами на обучение и тест (см. поле “Разбивка”) <br>

Метрики качества: MAPE, собственная метрика. <br>

Собственная метрика: до 80 = ДОЛЯ(TARGET/PREDICT < 0,8); более 120 = ДОЛЯ(TARGET/PREDICT > 1,2); 80-120 = ДОЛЯ((TARGET/PREDICT >= 0,8)&&(TARGET/PREDICT <= 1,2)). Повышением качества модели является максимизация доли «80-120», минимизация доли «до 80» и минимизация MAPE.
