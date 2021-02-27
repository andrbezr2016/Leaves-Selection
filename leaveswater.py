import glob
import os
import numpy as np
import cv2

class LeavesWater():
	def leafSelection(self, inImages, outPath):
		"""
		Функция выделения листьев на изображениях.
		Возвращает список(массив) путей ко всем изображениям, список содрержащие площади листьев.
		inImages список путей до изображений которые необходимо обработать.
		outImages список путей до обработанных изображений.
		outPath - путь по которому сохраняем результат.
		"""
		outImages = []
		pix_white = []
		for i in range(len(inImages)):
			image = cv2.imread(inImages[i]) # Считываем изображение
			kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
			erd = cv2.erode(image, kernel, iterations = 1)
			# Выделим зеленый
			hsv = cv2.cvtColor(erd, cv2.COLOR_BGR2HSV) # Меняем цветовую модель с BGR на HSV
			hsv_min = np.array((30, 0, 0), np.uint8)
			hsv_max = np.array((100, 255, 255), np.uint8)
			mask = cv2.inRange(hsv, hsv_min, hsv_max)
			# Зададим маркеры
			pix = image.shape
			markers = np.zeros((pix[0],pix[1]), dtype="int32")
			# Выделяем передний план
			dist_transform = cv2.distanceTransform(mask,cv2.DIST_L2,5)
			r, sure_fg = cv2.threshold(dist_transform,0.3*dist_transform.max(),255,0)
			markers[sure_fg==255] = 2
			# Выделяем фон
			sure_bg = cv2.dilate(mask,kernel,iterations=7)
			sure_bg = cv2.bitwise_not(sure_bg)
			markers[sure_bg==255] = 1
			# Watershed
			markers = cv2.watershed(erd,markers)
			image[markers==1] = (0,0,0)
			image[markers==-1] = (0,0,0)
			image[markers==2] = (48,160,103)
			# Считаем площадь
			square = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
			r, square = cv2.threshold(square, 40, 255, cv2.THRESH_BINARY)
			pix_w = np.count_nonzero(square)
			pix_white.append(pix_w)
			print(pix_w)
			# Сохраняем результаты
			pathsAndNames = inImages[i].rsplit('\\', 1) # Список путей к папке и имен для всех изображений
			pathsAndNames[1] = os.path.splitext(pathsAndNames[1])[0] # Убираем расширение файла из имени
			cv2.imwrite(outPath+pathsAndNames[1]+'_{}_water.jpg'.format(i), image) # Сохраняем файл
			outImages.append(outPath+pathsAndNames[1]+'_{}_water.jpg'.format(i))
		return outImages, np.asarray(pix_white)

	def damageSelection(self, inImages, outPath, inMask):
		"""
		Функция выделения поврежденных участков листа.
		Принимаемое и возвращаемые значения аналогичны leafSelection().
		Кроме inMask - задают области где искать повреждения (на листе).
		"""
		outImages = []
		pix_white = []
		for i in range(len(inImages)):
			image = cv2.imread(inImages[i]) # Считываем изображение
			mask = cv2.imread(inMask[i]) # Считываем маску
			mask_g = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
			r, mask_g = cv2.threshold(mask_g, 60, 255, cv2.THRESH_BINARY)
			kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
			erd = cv2.erode(image, kernel, iterations = 1)
			# Выделим пораженные участки
			hsv = cv2.cvtColor(erd, cv2.COLOR_BGR2HSV) # Меняем цветовую модель с BGR на HSV
			hsv_min = np.array((40, 0, 0), np.uint8)
			hsv_max = np.array((90, 255, 255), np.uint8)
			mask_dam = cv2.inRange(hsv, hsv_min, hsv_max)
			mask_dam = cv2.bitwise_not(mask_dam) # Инвертируем
			mask_dam[mask_g == 0] = 0
			# Зададим маркеры
			pix = image.shape
			markers = np.zeros((pix[0],pix[1]), dtype="int32")
			# Убираем шум
			kernel = np.ones((3,3),np.uint8)
			mask_dam = cv2.morphologyEx(mask_dam,cv2.MORPH_OPEN,kernel, iterations = 2)
			# Выделяем передний план
			dist_transform = cv2.distanceTransform(mask_dam,cv2.DIST_L1,5)
			r, sure_fg = cv2.threshold(dist_transform,0.3*dist_transform.max(),255,0)
			markers[sure_fg==255] = 2
			# Выделяем фон
			sure_bg = cv2.dilate(mask_dam,kernel,iterations=1)
			sure_bg = cv2.bitwise_not(sure_bg) # Инвертируем
			markers[sure_bg==255] = 1
			# Watershed
			markers = cv2.watershed(erd,markers)
			image[mask_g==0] = (0,0,0)
			image[markers==2] = (255,0,0)
			# Считаем площадь
			sq = np.copy(image)
			sq[markers==1] = 0
			sq[markers==-1] = 0
			pix_w = np.count_nonzero(sq)
			pix_white.append(pix_w)
			print(pix_w)
			# Сохраняем результаты
			pathsAndNames = inImages[i].rsplit('\\', 1) # Список путей к папке и имен для всех изображений
			pathsAndNames[1] = os.path.splitext(pathsAndNames[1])[0] # Убираем расширение файла из имени
			cv2.imwrite(outPath+pathsAndNames[1]+'_{}_water.jpg'.format(i), image) # Сохраняем файл
			outImages.append(outPath+pathsAndNames[1]+'_{}_water.jpg'.format(i))
		return outImages, np.asarray(pix_white)

if __name__ == "__main__":
	lv = LeavesWater()
	inPath = 'input\\' # Путь к папке в которой ищем все изображения
	outPath = 'output\\' # Путь к папке в которой сохраняем результат
	inImages = glob.glob(inPath+'**\*.jpg', recursive=True) # Получаем список путей ко всем изображениям .JPG по пути inPath
	# Выделим листья
	outImages, pix_white = lv.leafSelection(inImages, outPath)
	# Выделим пораженные участки на листьях
	outImages_dam, pix_white_dam = lv.damageSelection(inImages, outPath, outImages)
	# Запишем в файл площади и процент повреждения
	f = open('output_LW.txt', 'w')
	f.write("{:70s} {:30s} {:30s} {:30s} \n".format('Имя файла', 'Площадь листа (пиксели)', 'Площадь повреждения (пиксели)', 'Процент повреждения'))
	for i in range(len(outImages_dam)):
		pathsAndNames = inImages[i].rsplit('\\', 1)
		f.write("{:70s} {:^30d} {:^30d} {:^30f} \n".format(pathsAndNames[1], pix_white[i], pix_white_dam[i], 100*pix_white_dam[i]/pix_white[i]))
	f.close()
