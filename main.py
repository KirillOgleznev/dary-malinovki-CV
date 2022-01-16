from imageProcessor import ImageProcessor

# def coords(event, x, y):
#     if event == 1:
#         print(x, y)


if __name__ == '__main__':
    # Создание объекта класса анализатора фото
    # 'data/5.jpg', 'data/1.mp4'
    imgAnalyzer = ImageProcessor(srcImg='data/6.jpg', ratio=0.3)
    num = 0
    key = 0

    # Выход при нажалии Esc
    while key != 27:
        imgAnalyzer.find_and_draw_contours()
        imgAnalyzer.resize(1)
        imgAnalyzer.showAll()
        imgAnalyzer.create_report()

        key = imgAnalyzer.get_key(num)
        num = 1
