import cv2
import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
import operator
from tkinter import filedialog as fd
import tkinter.messagebox as mb
from skimage import io, measure, transform, metrics
from skimage.measure import block_reduce


def Gradient_func(file):

    ksize = 3
    dx = 1
    dy = 1

    gradient_x = cv2.Sobel(file, cv2.CV_32F, dx, 0, ksize=ksize)
    gradient_y = cv2.Sobel(file, cv2.CV_32F, 0, dy, ksize=ksize)

    abs_gradient_x = cv2.convertScaleAbs(gradient_x)
    abs_gradient_y = cv2.convertScaleAbs(gradient_y)

    gradient = cv2.addWeighted(abs_gradient_x, 0.5, abs_gradient_y, 0.5, 0)

    Gradient_sum = []
    for i in range(0, len(gradient)):
        Gradient_sum.append(round(sum(gradient[i]) / len(gradient[i]), 1))
    return Gradient_sum


def DCT_func(file):
    DCT = cv2.dct(np.float32(file))
    return DCT


def DFT_func(file):
    DFT = cv2.dft(np.float32(file), flags=cv2.DFT_COMPLEX_OUTPUT)

    DFT_shift = np.fft.fftshift(DFT)
    DFT_result = 20 * np.log(cv2.magnitude(DFT_shift[:, :, 0], DFT_shift[:, :, 1]))
    return DFT_result


def Histogram_func(file):
    Histogram = cv2.calcHist([file], [0], None, [256], [0, 256])
    return Histogram


def Scale_func(file, picture_size):
    Result = transform.resize(io.imread(file), (picture_size, picture_size))
    return Result


def Etalon_amount_take():
    Etalon_amount = Etalon_amount_input.get()
    if Etalon_amount.isdigit() and int(Etalon_amount) > 0:
        Face_detection(int(Etalon_amount))
    else:
        tk.showerror("Ошибка", "Введено неверное число, введите целое положительное число")


def Chose_files_manual():
    file1 = fd.askopenfilename()
    file2 = fd.askopenfilename()
    Face_detection_manual(file1, file2)


def Face_detection(Etalon_amount):
    DCT_statistics = []
    DFT_statistics = []
    Scale_statistics = []
    Histagram_statistics = []
    Gradient_statistics = []

    Histogram_delta_limit = 195
    Gradient_delta_limit = 75
    Scale_size = 17

    IMG_primer_array = []
    Histogram_primer = []
    Gradient_primer = []
    DFT_primer = []
    DCT_primer = []
    Scale_primer = []

    IMG_etalon_array = []
    Histogram_etalon = []
    Gradient_etalon = []
    DFT_etalon = []
    DCT_etalon = []
    Scale_etalon = []

    for i in range(1, 11):
        Histagram_sum = 0
        Gradient_sum = 0
        DFT_sim_sum = 0
        DCT_sim_sum = 0
        Scale_sim_sum = 0
        for j in range(1, Etalon_amount + 1):
            Histogram_result_counter = 0
            Gradient_result_counter = 0
            IMG_etalon = cv2.imread(f"s{i}/{j}.pgm", cv2.IMREAD_GRAYSCALE)
            IMG_etalon_array.append(IMG_etalon)
            Histogram_etalon.append(Histogram_func(IMG_etalon))
            Gradient_etalon.append(Gradient_func(IMG_etalon))
            DFT_etalon.append(DFT_func(IMG_etalon))
            DCT_etalon.append(DCT_func(IMG_etalon))
            Scale_etalon.append(Scale_func(f"s{i}/{j}.pgm", Scale_size))

            for k in range(Etalon_amount + 1, 11):
                IMG_primer = cv2.imread(f"S{i}/{k}.pgm", cv2.IMREAD_GRAYSCALE)
                IMG_primer_array.append(IMG_primer)
                Histogram_primer.append(Histogram_func(IMG_primer))
                Gradient_primer.append(Gradient_func(IMG_primer))
                DFT_primer.append(DFT_func(IMG_primer))
                DCT_primer.append(DCT_func(IMG_primer))
                Scale_primer.append(Scale_func(f"S{i}/{k}.pgm", Scale_size))

                Histogram_etalon_index, Histogram_etalon_max_value = max(
                    enumerate(Histogram_etalon[j - 1 + Etalon_amount * (i - 1)]), key=operator.itemgetter(1))
                Gradient_etalon_index, Gradient_etalon_max_value = max(
                    enumerate(Gradient_etalon[j - 1 + Etalon_amount * (i - 1)]), key=operator.itemgetter(1))

                Histogram_primer_max = Histogram_primer[k - Etalon_amount - 1 + (10 - Etalon_amount) * (i - 1)][
                    Histogram_etalon_index]
                Gradient_primer_max = Gradient_primer[k - Etalon_amount - 1 + (10 - Etalon_amount) * (i - 1)][
                    Gradient_etalon_index]

                Histogram_delta = abs(Histogram_etalon_max_value - Histogram_primer_max)
                Gradient_delta = abs(Gradient_etalon_max_value - Gradient_primer_max)

                if Histogram_delta < Histogram_delta_limit:
                    Histogram_result_counter += 1
                if Gradient_delta < Gradient_delta_limit:
                    Gradient_result_counter += 1

                DFT_etalon_mean = np.mean(DFT_etalon[j - 1 + Etalon_amount * (i - 1)])
                DFT_primer_mean = np.mean(DFT_primer[k - Etalon_amount - 1 + (10 - Etalon_amount) * (i - 1)])
                DFT_like_percent = DFT_primer_mean / DFT_etalon_mean

                if DFT_like_percent > 1:
                    DFT_like_percent = 2 - DFT_like_percent

                DFT_sim_sum += DFT_like_percent

                DCT_etalon_linalg_norm = np.linalg.norm(DCT_etalon[j - 1 + Etalon_amount * (i - 1)])
                DCT_primer_linalg_norm = np.linalg.norm(
                    DCT_primer[k - Etalon_amount - 1 + (10 - Etalon_amount) * (i - 1)])
                DCT_like_percent = DCT_primer_linalg_norm / DCT_etalon_linalg_norm

                if DCT_like_percent > 1:
                    DCT_like_percent = 2 - DCT_like_percent

                DCT_sim_sum += DCT_like_percent
                Scale_like_percent = metrics.structural_similarity(Scale_etalon[j - 1 + Etalon_amount * (i - 1)], Scale_primer[k - Etalon_amount - 1 + (10 - Etalon_amount) * ( i - 1)], data_range=255)

                if Scale_like_percent > 1:
                    Scale_like_percent = 2 - Scale_like_percent

                Scale_sim_sum += Scale_like_percent

            Histagram_sum += Histogram_result_counter
            Gradient_sum += Gradient_result_counter

        Histagram_statistics.append(Histagram_sum / ((10 - Etalon_amount) * Etalon_amount))
        Gradient_statistics.append(Gradient_sum / ((10 - Etalon_amount) * Etalon_amount))
        DFT_statistics.append(DFT_sim_sum / ((10 - Etalon_amount) * Etalon_amount))
        DCT_statistics.append(DCT_sim_sum / ((10 - Etalon_amount) * Etalon_amount))
        Scale_statistics.append(Scale_sim_sum / ((10 - Etalon_amount) * Etalon_amount))

    fig, ((Img_etal, Histogram_etal, DFT_etal, DCT_etal, Gradient_etal, Scale_etal),
          (Img_prim, Histogram_prim, DFT_prim, DCT_prim, Gradient_prim, Scale_prim)) = plt.subplots(2, 6,
                                                                                                    label="Сравнение")
    fig2, (Hist_stat, DFT_stat, DCT_stat, Grad_stat, Scale_stat) = plt.subplots(1, 5, label="Итоговая статистика")
    plt.ion()

    # Эталон
    IMG_array_etalon = Img_etal.imshow(IMG_etalon_array[0])
    Img_etal.set_title("Эталон")
    Histogram_array_etalon, = Histogram_etal.plot(Histogram_etalon[0], color="green")
    Histogram_etal.set_title("Гистограмма")
    DFT_array_etalon = DFT_etal.imshow(DFT_etalon[0], cmap='gray', vmin=0, vmax=255)
    DFT_etal.set_title("DFT")
    DCT_array_etalon = DCT_etal.imshow(np.abs(DCT_etalon[0]), vmin=0, vmax=255)
    DCT_etal.set_title("DCT")
    x_etalon = np.arange(len(Gradient_etalon[0]))
    Gradient_array_etalon, = Gradient_etal.plot(x_etalon, Gradient_etalon[0], color="green")
    Gradient_etal.set_title("Градиент")
    Scale_array_etalon = Scale_etal.imshow(Scale_etalon[0])
    Scale_etal.set_title("Scale")

    # Пример
    IMG_array = Img_prim.imshow(IMG_primer_array[0])
    Img_prim.set_title("Пример")
    Histogram_array, = Histogram_prim.plot(Histogram_primer[0], color="green")
    Histogram_prim.set_title("Гистограмма")
    DFT_array = DFT_prim.imshow(DFT_primer[0], cmap='gray', vmin=0, vmax=255)
    DFT_prim.set_title("DFT")
    DCT_array = DCT_prim.imshow(np.abs(DCT_primer[0]), vmin=0, vmax=255)
    DCT_prim.set_title("DCT")
    x = np.arange(len(Gradient_primer[0]))
    Gradient_array, = Gradient_prim.plot(x, Gradient_primer[0], color="green")
    Gradient_prim.set_title("Градиент")
    Scale_array = Scale_prim.imshow(Scale_primer[0])
    Scale_prim.set_title("Scale")

    # Статистика
    Gradient_x_array = np.arange(len(Gradient_statistics))
    Histogram_x_array = np.arange(len(Histagram_statistics))
    DFT_x_array = np.arange(len(DFT_statistics))
    DCT_x_array = np.arange(len(DCT_statistics))
    Scale_x_array = np.arange(len(Scale_statistics))

    Hist_stat.plot(Histogram_x_array, Histagram_statistics, color="green")
    Hist_stat.set_title('Гистограмма')
    DFT_stat.plot(DFT_x_array, DFT_statistics, color="green")
    DFT_stat.set_title('DFT')
    DCT_stat.plot(DCT_x_array, DCT_statistics, color="green")
    DCT_stat.set_title('DCT')
    Grad_stat.plot(Gradient_x_array, Gradient_statistics, color="green")
    Grad_stat.set_title('Градиент')
    Scale_stat.plot(Scale_x_array, Scale_statistics, color="green")
    Scale_stat.set_title('Scale')

    fig.set_size_inches(19, 5)
    fig.show()
    fig2.set_size_inches(19, 4)
    fig2.show()

    for t in range(0, 10):
        for p in range(Etalon_amount * t, Etalon_amount * t + Etalon_amount):
            IMG_array_etalon.set_data(IMG_etalon_array[p])
            Histogram_array_etalon.set_ydata(Histogram_etalon[p])
            DFT_array_etalon.set_data(DFT_etalon[p])
            DCT_array_etalon.set_data(DCT_etalon[p])
            Gradient_array_etalon.set_ydata(Gradient_etalon[p])
            Scale_array_etalon.set_data(Scale_etalon[p])
            fig.canvas.draw()
            fig.canvas.flush_events()
            for m in range(p * (10 - Etalon_amount), (10 - Etalon_amount) * (p + 1)):
                IMG_array.set_data(IMG_primer_array[m])
                Histogram_array.set_ydata(Histogram_primer[m])
                DFT_array.set_data(DFT_primer[m])
                DCT_array.set_data(DCT_primer[m])
                Gradient_array.set_ydata(Gradient_primer[m])
                Scale_array.set_data(Scale_primer[m])
                fig.canvas.draw()
                fig.canvas.flush_events()


def Face_detection_manual(file1, file2):
    Histogram_delta_limit = 100
    Gradient_delta_limit = 80
    Scale_size = 17

    Histogram_result_counter = 0
    Gradient_result_counter = 0

    IMG_etalon = cv2.imread(file1, cv2.IMREAD_GRAYSCALE)
    Histogram_etalon = Histogram_func(IMG_etalon)
    Gradient_etalon = Gradient_func(IMG_etalon)
    DFT_etalon = DFT_func(IMG_etalon)
    DCT_etalon = DCT_func(IMG_etalon)
    Scale_etalon = Scale_func(file1, Scale_size)

    IMG_primer = cv2.imread(file2, cv2.IMREAD_GRAYSCALE)
    Histogram_primer = Histogram_func(IMG_primer)
    Gradient_primer = Gradient_func(IMG_primer)
    DFT_primer = DFT_func(IMG_primer)
    DCT_primer = DCT_func(IMG_primer)
    Scale_primer = Scale_func(file2, Scale_size)

    IMG_etalon_manual = plt.imread(file1, cv2.IMREAD_GRAYSCALE)
    IMG_primer_manual = plt.imread(file2, cv2.IMREAD_GRAYSCALE)

    Histogram_etalon_index, Histogram_etalon_max_value = max(enumerate(Histogram_etalon), key=operator.itemgetter(1))
    Gradient_etalon_index, Gradient_etalon_max_value = max(enumerate(Gradient_etalon), key=operator.itemgetter(1))

    Histogram_primer_max = Histogram_primer[Histogram_etalon_index]
    Gradient_primer_max = Gradient_primer[Gradient_etalon_index]

    Histogram_delta = abs(Histogram_etalon_max_value - Histogram_primer_max)
    Gradient_delta = abs(Gradient_etalon_max_value - Gradient_primer_max)

    if Histogram_delta < Histogram_delta_limit:
        Histogram_result_counter += 1
    if Gradient_delta < Gradient_delta_limit:
        Gradient_result_counter += 1

    plt.figure("Сравнение снимков выбранных вручную")
    plt.subplot(3, 6, 13)
    plt.imshow(IMG_etalon_manual)
    plt.title("Эталон")
    plt.subplot(3, 6, 14)
    plt.plot(Histogram_etalon, color="b")
    plt.title("Гистограмма")
    plt.subplot(3, 6, 15)
    plt.imshow(DFT_etalon, cmap='gray', vmin=0, vmax=255)
    plt.title("DFT")
    plt.subplot(3, 6, 16)
    plt.imshow(np.abs(DCT_etalon), vmin=0, vmax=255)
    plt.title("DCT")
    plt.subplot(3, 6, 17)
    x = np.arange(len(Gradient_etalon))
    plt.plot(x, Gradient_etalon, color="b")
    plt.title("Градиент")
    plt.subplot(3, 6, 18)
    plt.imshow(Scale_etalon)
    plt.title("Scale")

    plt.subplot(3, 6, 1)
    plt.imshow(IMG_primer_manual)
    plt.title("Тестовая")
    plt.subplot(3, 6, 2)
    plt.plot(Histogram_primer, color="b")
    plt.title("Гистограмма")
    plt.subplot(3, 6, 3)
    plt.imshow(DFT_primer, cmap='gray', vmin=0, vmax=255)
    plt.title("DFT")
    plt.subplot(3, 6, 4)
    plt.imshow(np.abs(DCT_primer), vmin=0, vmax=255)
    plt.title("DCT")
    plt.subplot(3, 6, 5)
    x = np.arange(len(Gradient_primer))
    plt.plot(x, Gradient_primer, color="b")
    plt.title("Градиент")
    plt.subplot(3, 6, 6)
    plt.imshow(Scale_primer)
    plt.title("Scale")

    if Gradient_result_counter != 0 and Histogram_result_counter != 0:
        check = "Совпадает"
        check_color = "green"
    else:
        check = "Не совпадает"
        check_color = "red"

    plt.text(0.4, 0.5, check, fontsize=14, color=check_color, transform=plt.gcf().transFigure)
    plt.show()


App = tk.Tk()
App.title('Опознавание лиц №3')
App.geometry("300x100")

Etalon_amount_label = tk.Label(App, text="Количество эталонов:")
Etalon_amount_label.pack()
Etalon_amount_input = tk.Entry(App)
Etalon_amount_input.pack()

App_button = tk.Button(App, text="Построить графики", command=Etalon_amount_take)
App_button.pack()

App_button = tk.Button(App, text="Вручную выбрать эталон и пример", command=Chose_files_manual)
App_button.pack()

App.mainloop()

