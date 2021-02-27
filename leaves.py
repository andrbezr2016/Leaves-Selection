import glob
import os
import numpy as np
import cv2

class Leaves():
	def histogramSearch(self, histr, threshold, skip, mask, mask_def):
		"""
		Функция работает с массивом histr.
		Ищет в нем максимальный по ширине диапазон значений histr выше threshold.
		Если растояние между двумя соседними диапазонами меньше skip,
		то объединяет в один диапазон.
		mask задает границы histr в которых искать диапазоны.
		Функция возвращает границы диапазона.
		Если диапазон не найден, возвращает диапазон по умолчанию mask_def.
		"""
		count = 0
		count_list = []
		h_min = []
		h_max = []
		i = mask[0]
		while i < mask[1]:
			if histr[i] > threshold:
				h_min.append(i)
				while histr[i] > threshold:
					count+=1
					i+=1
				h_max.append(i)
				count_list.append(count)
				count = 0
			i+=1
		if len(count_list) == 0:
			return mask_def[0], mask_def[1]
		# Убираем зазоры между диапазонами если они есть и меньше skip
		j = len(count_list)-1
		while j > 0:
			if h_min[j] - h_max[j-1] < skip:
				del h_max[j-1]
				del h_min[j]
				count_list[j-1] += count_list[j]
				del count_list[j]
			j-=1
		index = count_list.index(max(count_list))
		return h_min[index], h_max[index]

	def leafSelection(self, inImages, outPath):
		"""
		Функция выделения листьев на изображениях.
		Возвращает список(массив) путей ко всем изображениям и списки(массивы) содрержащие площади листьев и всего изображения.
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
			# Выделяем зеленый цвет
			hsv = cv2.cvtColor(erd, cv2.COLOR_BGR2HSV) # Меняем цветовую модель с BGR на HSV
			histr = cv2.calcHist([hsv],[0],None,[181],[0,181])
			h_min, h_max = Leaves.histogramSearch(self, histr, threshold=300, skip=10, mask=(30,101), mask_def = (30, 101))
			hsv_min = np.array((h_min, 0, 0), np.uint8)
			hsv_max = np.array((h_max, 255, 255), np.uint8)
			mask = cv2.inRange(hsv, hsv_min, hsv_max)
			mask = cv2.medianBlur(mask, 25) # Медианный фильтр, чтобы сгладить края маски
			# Накадываем изначальное изображение
			image = cv2.bitwise_and(image, image, mask=mask)
			# Заливаем область вне листа, чтобы удальть близкие к черному пиксели(тени), если они есть
			pix = image.shape
			if not any(image[0,0]):
				cv2.floodFill(image, None, (0,0), (0,0,0), 0, (20,20,20))
			if not any(image[pix[0]-1,pix[1]-1]):
				cv2.floodFill(image, None, (pix[1]-1,pix[0]-1), (0,0,0), 0, (20,20,20))
			if not any(image[0,pix[1]-1]):
				cv2.floodFill(image, None, (pix[1]-1,0), (0,0,0), 0, (20,20,20))
			if not any(image[pix[0]-1,0]):
				cv2.floodFill(image, None, (0,pix[0]-1), (0,0,0), 0, (20,20,20))
			# Расчитаем площадь листьев
			sq = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
			pix_w = cv2.countNonZero(sq)
			pix_white.append(pix_w)
			print(pix_w)
			# Сохраняем результаты
			pathsAndNames = inImages[i].rsplit('\\', 1) # Список путей к папке и имен всех изображений
			pathsAndNames[1] = os.path.splitext(pathsAndNames[1])[0] # Убираем расширение файла из имени
			cv2.imwrite(outPath+pathsAndNames[1]+'_{}_masked.jpg'.format(i), image) # Сохраняем файл
			outImages.append(outPath+pathsAndNames[1]+'_{}_masked.jpg'.format(i))
		return outImages, np.asarray(pix_white)

	def damageSelection(self, inImages, outPath):
		"""
		Функция выделения поврежденных участков листа.
		Принимаемое и возвращаемые значения аналогичны leafSelection().
		"""
		outImages = []
		pix_white = []
		for i in range(len(inImages)):
			image = cv2.imread(inImages[i])
			kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
			erd = cv2.erode(image, kernel, iterations = 1)
			erd_g = cv2.cvtColor(erd, cv2.COLOR_BGR2GRAY)
			# Выделяем пораженные участки
			hsv = cv2.cvtColor(erd, cv2.COLOR_BGR2HSV) # Меняем цветовую модель с BGR на HSV
			hsv_min = np.array((40, 0, 0), np.uint8)
			hsv_max = np.array((90, 255, 255), np.uint8)
			mask = cv2.inRange(hsv, hsv_min, hsv_max)
			mask = cv2.bitwise_not(mask) # Инвертируем
			mask[erd_g == 0] = 0
			# Расчитаем поврежденных участков листа
			pix_w = cv2.countNonZero(mask)
			pix_white.append(pix_w)
			print(pix_w)
			# Накладываем на входные изображения
			image[mask == 255] = [255, 0, 0]
			# Сохраняем результаты
			pathsAndNames = inImages[i].rsplit('\\', 1) # Список путей к папке и имен всех изображений
			cv2.imwrite(outPath+pathsAndNames[1], image) # Сохраняем файл
			outImages.append(outPath+pathsAndNames[1])
		return outImages, np.asarray(pix_white)

if __name__ == "__main__":
	lv = Leaves()
	inPath = 'input\\' # Путь к папке в которой ищем все изображения
	outPath = 'output\\' # Путь к папке в которой сохраняем результат
	inImages = glob.glob(inPath+'**\*.jpg', recursive=True) # Получаем список путей ко всем изображениям .JPG по пути inPath
	# Выделим листья
	outImages, pix_white = lv.leafSelection(inImages, outPath)
	# Выделим пораженные участки на листьях
	outImages_dam, pix_white_dam = lv.damageSelection(outImages, outPath)
	# Запишем в файл площади и процент повреждения
	f = open('output_L.txt', 'w')
	f.write("{:70s} {:30s} {:30s} {:30s} \n".format('Имя файла', 'Площадь листа (пиксели)', 'Площадь повреждения (пиксели)', 'Процент повреждения'))
	for i in range(len(outImages_dam)):
		pathsAndNames = inImages[i].rsplit('\\', 1)
		f.write("{:70s} {:^30d} {:^30d} {:^30f} \n".format(pathsAndNames[1], pix_white[i], pix_white_dam[i], 100*pix_white_dam[i]/pix_white[i]))
	f.close()
