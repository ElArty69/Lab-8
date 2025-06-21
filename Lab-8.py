import cv2
import numpy as np

# 1. Преобразование изображения
def load_and_crop_image(image_path, crop_size=400):
    """
    Загружает изображение по указанному пути и обрезает область 400x400
    из центра изображения.
    """
    # 1. Загрузить изображение
    original_image = cv2.imread(image_path)

    if original_image is None:
        print("Ошибка: Не удалось загрузить изображение. Проверьте путь.")
        exit()

    # 2. Получить центр изображения
    height, width = original_image.shape[:2]
    center_x, center_y = width // 2, height // 2

    # 3. Обрезать изображение
    half_size = crop_size // 2
    if height >= crop_size and width >= crop_size:
        cropped_image = original_image[
            center_y - half_size : center_y + half_size,
            center_x - half_size : center_x + half_size
        ]
    else:
        print("Ошибка: Изображение меньше 400x400 пикселей")
        exit()

    return original_image, cropped_image



# 2. Алгоритм отслеживания метки
def process_camera_feed(fly_image_path):
    """
    Считывает видео с камеры в реальном времени, отслеживает метку
    и накладывает изображение мухи на метку.
    """
    # Инициализировать камеру
    cap = cv2.VideoCapture(0)

    # Загрузить изображение мухи, которое будет накладываться на метку
    fly_image = cv2.imread(fly_image_path, -1)  
    fly_height, fly_width = fly_image.shape[:2]

    while True:
        # Считываем кадр с камеры
        ret, frame = cap.read()
        if not ret:
            break

        # Преобразуем кадр в оттенки серого
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Применить порог для выявления метки
        _, threshold_frame = cv2.threshold(gray_frame, 100, 255, cv2.THRESH_BINARY)

        # Найти контуры в кадре
        contours, _ = cv2.findContours(threshold_frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Если контуры найдены, рисуем контур метки
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)

            # Получаем ограничивающий прямоугольник для самого большого контура
            x, y, w, h = cv2.boundingRect(largest_contour)

            # Рисуем контур и прямоугольник вокруг метки
            cv2.drawContours(frame, [largest_contour], -1, (0, 255, 0), 2)  # Зеленый контур
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)  # Красный прямоугольник

            # Вычисляем центр метки
            center_x = x + w // 2
            center_y = y + h // 2

            # Рисуем вертикальную и горизонтальную линии
            cv2.line(frame, (center_x, 0), (center_x, frame.shape[0]), (255, 0, 0), 2)  # Синяя вертикальная линия
            cv2.line(frame, (0, center_y), (frame.shape[1], center_y), (0, 0, 255), 2)  # Красная горизонтальная линия

            # Дополнительное задание
            # Накладываем изображение мухи в центр метки
            fly_x = center_x - fly_width // 2
            fly_y = center_y - fly_height // 2

            for i in range(fly_height):
                for j in range(fly_width):
                    if 0 <= fly_y + i < frame.shape[0] and 0 <= fly_x + j < frame.shape[1]:
                        if fly_image[i, j][3] != 0:  
                            frame[fly_y + i, fly_x + j] = fly_image[i, j][:3]  

        # Показываем видео с накладываемой мухой
        cv2.imshow("Отслеживание метки с мухой", frame)

        # Выход из программы при нажатии клавиши 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Освободить камеру и закрыть окна
    cap.release()
    cv2.destroyAllWindows()


def main():
    # Загрузить и обрезать изображение
    image_path = "variant-8.jpg"
    original_image, cropped_image = load_and_crop_image(image_path)

    # Показать оригинальное и обрезанное изображение
    cv2.imshow("Оригинальное изображение", original_image)
    cv2.imshow("Обрезка 400x400", cropped_image)

    # Сохранить обрезанное изображение
    output_path = "cropped_variant8.jpg"
    cv2.imwrite(output_path, cropped_image)
    print(f"Обрезка сохранена как: {output_path}")

    # Обработать видео и наложить изображение мухи на центр метки
    fly_image_path = "fly64.png"
    process_camera_feed(fly_image_path)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
    