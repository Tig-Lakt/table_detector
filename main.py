import cv2
import pandas as pd
import argparse
from ultralytics import YOLO
from datetime import timedelta

# ─── КОНСТАНТЫ ────────────────────────────────────────────────────────────────

# Координаты зоны столика (x, y, ширина, высота) — получены через selectROI
ROI_X, ROI_Y, ROI_W, ROI_H = 899, 357, 412, 248

# Минимальная уверенность YOLO для принятия детекции
CONFIDENCE_THRESHOLD = 0.5

# Сколько кадров подряд нужно для фиксации смены состояния (дебаунсинг).
# Устраняет дёрганье состояния когда человек стоит на границе зоны.
DEBOUNCE_FRAMES = 15

# Минимальная доля перекрытия bbox человека с зоной столика.
# 0.25 = человек должен находиться минимум на 25% внутри зоны.
# Устраняет ложные срабатывания от людей за соседними столами.
IOU_THRESHOLD = 0.25

# Минимальное время присутствия у стола в секундах.
# Если человек пробыл меньше — считаем проходом мимо и убираем из отчёта.
# Устраняет срабатывания от официантов проходящих мимо.
MIN_OCCUPIED_SECONDS = 3.0

# Состояния столика
STATE_EMPTY    = "EMPTY"
STATE_OCCUPIED = "OCCUPIED"

# Цвета для отрисовки bbox столика (BGR — OpenCV использует BGR, не RGB!)
COLOR_EMPTY    = (0, 255, 0)   # Зелёный — стол пуст
COLOR_OCCUPIED = (0, 0, 255)   # Красный — стол занят


# ─── ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ ──────────────────────────────────────────────────

def is_person_in_roi(box, roi, iou_threshold=IOU_THRESHOLD):
    """
    Проверяет присутствие человека в зоне столика через IoU.

    Вместо простого пересечения считаем долю перекрытия:
    если человек лишь краем задел зону — игнорируем.

    Дополнительно: используем только нижние 2/3 bbox человека —
    ноги и торс точнее показывают где человек стоит,
    голова может нависать над столом даже если человек далеко.

    box:           (x1, y1, x2, y2) — bbox человека от YOLO
    roi:           (rx, ry, rw, rh) — зона столика
    iou_threshold: минимальная доля перекрытия (0.25 = 25%)
    """
    x1, y1, x2, y2 = box
    rx, ry, rw, rh = roi
    rx2, ry2 = rx + rw, ry + rh

    # Отрезаем верхнюю треть bbox (голову) — работаем только с телом
    body_height = y2 - y1
    y1 = y1 + int(body_height * 0.33)

    # Площадь пересечения
    inter_x1 = max(x1, rx)
    inter_y1 = max(y1, ry)
    inter_x2 = min(x2, rx2)
    inter_y2 = min(y2, ry2)

    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    if inter_area == 0:
        return False

    # Считаем IoU относительно площади человека:
    # какая доля тела человека попала в зону столика
    person_area = (x2 - x1) * (y2 - y1)
    iou = inter_area / person_area

    return iou >= iou_threshold


def frame_to_timestamp(frame_idx, fps):
    """
    Переводит номер кадра во временную метку (timedelta).
    Например: кадр 150 при fps=20 → 0:00:07.500000
    """
    seconds = frame_idx / fps
    return timedelta(seconds=round(seconds, 2))


def draw_roi(frame, roi, state):
    """
    Рисует прямоугольник зоны столика на кадре.
    Цвет зависит от текущего состояния: зелёный — пусто, красный — занято.
    """
    rx, ry, rw, rh = roi
    color = COLOR_EMPTY if state == STATE_EMPTY else COLOR_OCCUPIED

    # Прямоугольник зоны столика
    cv2.rectangle(frame, (rx, ry), (rx + rw, ry + rh), color, 2)

    # Подпись состояния над прямоугольником
    label = "EMPTY" if state == STATE_EMPTY else "OCCUPIED"
    cv2.putText(
        frame, label,
        (rx, ry - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7, color, 2
    )


def filter_short_events(df, min_seconds=MIN_OCCUPIED_SECONDS):
    """
    Постфильтрация таблицы событий: убирает пары EMPTY→OCCUPIED / OCCUPIED→EMPTY
    где стол был занят меньше min_seconds секунд (проходы мимо, случайные детекции).

    Работает над готовым DataFrame — не влияет на логику основного цикла.
    Это принципиально важно: цикл честно фиксирует всё,
    а фильтрация происходит отдельно и не ломает внутренние переменные состояния.
    """
    rows_to_drop = []
    i = 0

    while i < len(df) - 1:
        current_event = df.iloc[i]["event"]
        next_event    = df.iloc[i + 1]["event"]

        # Ищем пару: стол занят → стол пуст
        if current_event == "EMPTY → OCCUPIED" and next_event == "OCCUPIED → EMPTY":
            t_start  = pd.to_timedelta(df.iloc[i]["timestamp"])
            t_end    = pd.to_timedelta(df.iloc[i + 1]["timestamp"])
            duration = (t_end - t_start).total_seconds()

            if duration < min_seconds:
                # Слишком короткое присутствие — помечаем оба события на удаление
                rows_to_drop.extend([i, i + 1])
                print(f"[SKIP] проход мимо {df.iloc[i]['timestamp']} "
                      f"({duration:.1f}с) — убираем из отчёта")
                i += 2   # прыгаем через пару, она уже обработана
                continue
        i += 1

    df = df.drop(index=rows_to_drop).reset_index(drop=True)
    return df


def recalculate_delays(df):
    """
    Пересчитывает колонку delay_seconds после фильтрации.

    delay_seconds — время между уходом предыдущего гостя (OCCUPIED→EMPTY)
    и приходом следующего (EMPTY→OCCUPIED).

    Пересчёт нужен потому что после удаления строк старые значения устарели:
    некоторые пары событий были удалены и временные метки сдвинулись.
    """
    last_empty_ts = None

    for idx, row in df.iterrows():
        if row["event"] == "OCCUPIED → EMPTY":
            # Запоминаем момент когда стол опустел
            last_empty_ts = pd.to_timedelta(row["timestamp"])
            df.at[idx, "delay_seconds"] = None

        elif row["event"] == "EMPTY → OCCUPIED":
            if last_empty_ts is not None:
                # Считаем сколько времени стол простоял пустым
                delay = (pd.to_timedelta(row["timestamp"]) - last_empty_ts).total_seconds()
                df.at[idx, "delay_seconds"] = round(delay, 2)
            else:
                # Первое появление человека — предыдущего ухода не было
                df.at[idx, "delay_seconds"] = None

    return df


# ─── ГЛАВНАЯ ФУНКЦИЯ ──────────────────────────────────────────────────────────

def main(video_path):
    # Загружаем модель YOLO (веса скачаются автоматически при первом запуске)
    print("[INFO] Загружаем модель YOLO...")
    model = YOLO("yolov8n.pt")

    # Открываем видеофайл
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[ERROR] Не удалось открыть видео: {video_path}")
        return

    # Получаем параметры видео
    fps    = cap.get(cv2.CAP_PROP_FPS)
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"[INFO] Видео: {width}x{height}, {fps:.1f} fps, {total} кадров")

    # Настраиваем запись выходного видео
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out    = cv2.VideoWriter("output.mp4", fourcc, fps, (width, height))

    # ── Переменные состояния ──────────────────────────────────────────────────

    roi            = (ROI_X, ROI_Y, ROI_W, ROI_H)
    current_state  = STATE_EMPTY   # Текущее подтверждённое состояние
    pending_state  = STATE_EMPTY   # Кандидат на смену состояния
    debounce_count = 0             # Счётчик кадров для дебаунсинга
    frame_idx      = 0             # Номер текущего кадра

    # Временная метка когда стол последний раз стал пустым
    last_empty_time = None

    # Список событий — потом превратим в Pandas DataFrame
    events = []

    # ── Основной цикл обработки кадров ───────────────────────────────────────
    print("[INFO] Обрабатываем видео...")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1
        timestamp = frame_to_timestamp(frame_idx, fps)

        # ── Запускаем YOLO на текущем кадре ──────────────────────────────────
        # classes=[0] — детектируем только класс "person", игнорируем остальные
        # verbose=False — отключаем вывод YOLO в консоль на каждом кадре
        results = model(frame, verbose=False, classes=[0])

        # ── Проверяем: есть ли человек в зоне столика ────────────────────────
        person_in_roi = False

        for result in results:
            for box in result.boxes:
                confidence = float(box.conf[0])
                if confidence < CONFIDENCE_THRESHOLD:
                    continue

                x1, y1, x2, y2 = map(int, box.xyxy[0])

                if is_person_in_roi((x1, y1, x2, y2), roi):
                    person_in_roi = True
                    break

        # ── Дебаунсинг ───────────────────────────────────────────────────────
        # Состояние меняется только если оно стабильно DEBOUNCE_FRAMES кадров подряд.
        # Без этого состояние дёргалось бы каждый кадр на границе зоны.
        new_candidate = STATE_OCCUPIED if person_in_roi else STATE_EMPTY

        if new_candidate == pending_state:
            debounce_count += 1
        else:
            pending_state  = new_candidate
            debounce_count = 1

        # ── Фиксируем смену состояния если счётчик достиг порога ─────────────
        if debounce_count == DEBOUNCE_FRAMES and pending_state != current_state:
            previous_state = current_state
            current_state  = pending_state

            # Предварительно считаем задержку.
            # После постфильтрации она будет пересчитана заново.
            delay_seconds = None
            if current_state == STATE_OCCUPIED and last_empty_time is not None:
                delay_seconds = round(
                    (timestamp - last_empty_time).total_seconds(), 2
                )

            # Запоминаем когда стол стал пустым
            if current_state == STATE_EMPTY:
                last_empty_time = timestamp

            # Записываем событие в список
            events.append({
                "frame":         frame_idx,
                "timestamp":     str(timestamp),
                "event":         f"{previous_state} → {current_state}",
                "delay_seconds": delay_seconds
            })

            print(f"[EVENT] {timestamp} | {previous_state} → {current_state}"
                  + (f" | задержка: {delay_seconds}с" if delay_seconds else ""))

        # ── Рисуем визуализацию и записываем кадр ────────────────────────────
        draw_roi(frame, roi, current_state)

        if frame_idx % 100 == 0:
            print(f"[INFO] Обработано {frame_idx}/{total} кадров...")

        out.write(frame)

    # ── Завершение ────────────────────────────────────────────────────────────
    cap.release()
    out.release()
    print("[INFO] Обработка завершена. Видео сохранено в output.mp4")

    # ── Аналитика ─────────────────────────────────────────────────────────────
    if not events:
        print("[INFO] Событий не зафиксировано.")
        return

    df = pd.DataFrame(events)

    print("\n─── СОБЫТИЯ ДО ФИЛЬТРАЦИИ ─────────────────────────")
    print(df.to_string(index=False))

    # Шаг 1: убираем короткие присутствия (проходы мимо, случайные детекции)
    df = filter_short_events(df)

    # Шаг 2: пересчитываем задержки после удаления строк
    df = recalculate_delays(df)

    print("\n─── ТАБЛИЦА СОБЫТИЙ (после фильтрации) ────────────")
    print(df.to_string(index=False))

    delays = df["delay_seconds"].dropna()

    print("\n─── СТАТИСТИКА ────────────────────────────────────")
    if len(delays) > 0:
        print(f"Количество подходов к столу:       {len(delays)}")
        print(f"Среднее время до следующего гостя: {delays.mean():.1f} сек")
        print(f"Минимальная задержка:              {delays.min():.1f} сек")
        print(f"Максимальная задержка:             {delays.max():.1f} сек")
    else:
        print("Недостаточно данных для подсчёта задержки.")

    # Сохраняем отчёт в файл
    with open("report.txt", "w") as f:
        f.write("ОТЧЁТ: Детекция уборки столиков\n")
        f.write("=" * 40 + "\n\n")
        f.write(df.to_string(index=False))
        f.write("\n\n")
        if len(delays) > 0:
            f.write(f"Среднее время до следующего гостя: {delays.mean():.1f} сек\n")
            f.write(f"Минимальная задержка:              {delays.min():.1f} сек\n")
            f.write(f"Максимальная задержка:             {delays.max():.1f} сек\n")

    print("\n[INFO] Отчёт сохранён в report.txt")


# ─── ТОЧКА ВХОДА ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Детекция уборки столиков по видео")
    parser.add_argument("--video", required=True, help="Путь к видеофайлу")
    args = parser.parse_args()
    main(args.video)