import cv2

# Загружаем видео
cap = cv2.VideoCapture("видео 1.mp4")

# Читаем первый кадр
ret, frame = cap.read()
cap.release()

if not ret:
    print("Не удалось прочитать видео!")
    exit()

# Показываем окно где выбираем зону мышкой
# Зажми левую кнопку мыши и выдели прямоугольник вокруг столика
# Когда выделил — нажми ENTER или SPACE для подтверждения
# Нажми C — чтобы отменить и выбрать заново
print("Выдели зону столика мышкой, затем нажми ENTER или SPACE")
roi = cv2.selectROI("Выбери зону столика", frame, showCrosshair=True)
cv2.destroyAllWindows()

# roi возвращает кортеж (x, y, width, height)
x, y, w, h = roi
print(f"\n--- СКОПИРУЙТЕ ЭТИ ЧИСЛА ---")
print(f"ROI = ({x}, {y}, {w}, {h})")
print(f"--------------------------")