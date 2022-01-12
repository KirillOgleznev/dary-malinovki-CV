from imageProcessor import ImageProcessor

# def coords(event, x, y):
#     if event == 1:
#         print(x, y)


if __name__ == '__main__':
    # Создание объекта класса анализатора фото
    imgAnalyzer = ImageProcessor(src='data/5.jpg')
    num = 0
    key = 0

    # Выход при нажалии Esc
    while key != 27:
        imgAnalyzer.find_and_draw_contours()
        imgAnalyzer.resize(0.3)
        imgAnalyzer.showAll()
        imgAnalyzer.create_report()

        key = imgAnalyzer.get_key(num)
        num = 1
