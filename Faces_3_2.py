import cv2
import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
import operator
from skimage import io, measure, transform,metrics
import os
import tkinter.filedialog as fd


def Grad(file):
    ksize = 3
    dx = 1
    dy = 1

    gradient_x = cv2.Sobel(file, cv2.CV_64F, dx, 0, ksize=ksize)
    gradient_y = cv2.Sobel(file, cv2.CV_64F, 0, dy, ksize=ksize)

    abs_gradient_x = cv2.convertScaleAbs(gradient_x)
    abs_gradient_y = cv2.convertScaleAbs(gradient_y)

    gradient = cv2.addWeighted(abs_gradient_x, 0.5, abs_gradient_y, 0.5, 0)

    SumGrad = []
    for i in range(0, len(gradient), 1):
        SumGrad.append(round(sum(gradient[i]) / len(gradient[i]), 1))
    return SumGrad


def DCT(file):
    dct = cv2.dct(np.float32(file))
    return dct


def DFT(file):
    dft = cv2.dft(np.float32(file), flags=cv2.DFT_COMPLEX_OUTPUT)

    dft_shift = np.fft.fftshift(dft)
    magnitude_spectrum = 20 * np.log(cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1]))
    return magnitude_spectrum


def Hist(file):
    histg = cv2.calcHist([file], [0], None, [256], [0, 256])
    return histg


def HistTest(file):
    histg = cv2.calcHist([file], [0], None, [300], [0, 256])
    return histg


def Scale(file):
    img = io.imread(file, as_gray=True)
    scaleSize = 18
    img_res = transform.resize(img, (scaleSize, scaleSize))
    return img_res


def Etalon_amount_take(choosed_op):
    Etalon_amount = Etalon_amount_input.get()

    if choosed_op == 0:  # эталоны берутся по порядку
        start_pos = 1
        step = 1
        end_pos = int(Etalon_amount)

    if choosed_op == 1:  # Нечётные
        start_pos = 1
        step = 2
        end_pos = 10
        Etalon_amount = str(5)

    if choosed_op == 2:  # Чётные
        start_pos = 2
        step = 2
        end_pos = 10
        Etalon_amount = str(5)

    if Etalon_amount.isdigit() and int(Etalon_amount) > 0:
        Face_detection(int(Etalon_amount), start_pos, step, end_pos, choosed_op)
    else:
        tk.showerror("Ошибка", "Введите целое положительное число")


def Directory_choose_etalon_amount(Etal_amount, filename1):
    Etalon_amount = Etal_amount
    if Etalon_amount.isdigit() and int(Etalon_amount) > 0:
        Face_detection_manual_directory(filename1, int(Etalon_amount))
    else:
        tk.showerror("Ошибка", "Введите целое положительное число")


def Manual_choose():
    filename1 = fd.askopenfilename()
    filename2 = fd.askopenfilename()
    Face_detection_manual(filename1, filename2)


def Manual_choose_directory():
    filename1 = fd.askdirectory(title="Открыть папку", initialdir="/")
    App2 = tk.Tk()
    App2.title("Количество эталонов в папке")
    App2.geometry('300x75')

    Etalon_amount_label = tk.Label(App2, text="Количество эталонов:")
    Etalon_amount_label.pack()
    Etalon_amount_input = tk.Entry(App2)
    Etalon_amount_input.pack()

    App2_button = tk.Button(App2, text="Построить графики", command=lambda: Directory_choose_etalon_amount(Etalon_amount_input.get(), filename1))
    App2_button.pack()
    App2.mainloop()


def Face_detection(Etalon_amount, Start_position, Step, End_position, Function_chosen):
    DCT_statistics = []
    DFT_statistics = []
    Scale_statistics = []
    Histagram_statistics = []
    Gradient_statistics = []

    DCT_statistics_individual = []
    DFT_statistics_individual = []
    Scale_statistics_individual = []
    Histagram_statistics_individual = []
    Gradient_statistics_individual = []

    Histogram_delta_limit = 230
    Gradient_delta_limit = 80

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

    etalon_index = 0
    primer_index = 0

    if Function_chosen == 0:
        Additional_position = Etalon_amount

    if Function_chosen == 1:
        Additional_position = 1

    if Function_chosen == 2:
        Additional_position = 0

    folders_amount = len([f.path for f in os.scandir("IMG") if f.is_dir()])
    for i in range(1, folders_amount+1, 1):
        Histagram_sum = 0
        Gradient_sum = 0
        DFT_sim_sum = 0
        DCT_sim_sum = 0
        Scale_sim_sum = 0
        files_amount = len([f for f in os.listdir(f"IMG/s{i}") if os.path.isfile(os.path.join(f"IMG/s{i}", f))])
        for j in range(Start_position, End_position + 1, Step):
            Histogram_result_counter = 0
            Gradient_result_counter = 0
            IMG_etalon = cv2.imread(f"IMG/s{i}/{j}.pgm", cv2.IMREAD_GRAYSCALE)
            IMG_etalon_array.append(IMG_etalon)
            Histogram_etalon.append(Hist(IMG_etalon))
            Gradient_etalon.append(Grad(IMG_etalon))
            DFT_etalon.append(DFT(IMG_etalon))
            DCT_etalon.append(DCT(IMG_etalon))
            Scale_etalon.append(Scale(f"IMG/s{i}/{j}.pgm"))
            for k in range(Additional_position + 1, files_amount + 1, Step):
                IMG_primer = cv2.imread(f"IMG/s{i}/{k}.pgm", cv2.IMREAD_GRAYSCALE)
                IMG_primer_array.append(IMG_primer)
                Histogram_primer.append(Hist(IMG_primer))
                Gradient_primer.append(Grad(IMG_primer))
                DFT_primer.append(DFT(IMG_primer))
                DCT_primer.append(DCT(IMG_primer))
                Scale_primer.append(Scale(f"IMG/s{i}/{k}.pgm"))

                if (Function_chosen == 0):
                    etalon_element = j - 1 + Etalon_amount * (i - 1)
                    primer_element = k - Etalon_amount - 1 + (10 - Etalon_amount) * (i - 1)
                else:
                    etalon_element = etalon_index
                    primer_element = primer_index

                Histogram_etalon_index, Histogram_etalon_max_value = max(enumerate(Histogram_etalon[etalon_element]), key=operator.itemgetter(1))
                Gradient_etalon_index, Gradient_etalon_max_value = max(enumerate(Gradient_etalon[etalon_element]), key=operator.itemgetter(1))

                Histogram_primer_max = Histogram_primer[primer_element][Histogram_etalon_index]
                Gradient_primer_max = Gradient_primer[primer_element][Gradient_etalon_index]

                Histogram_delta = abs(Histogram_etalon_max_value - Histogram_primer_max)
                Gradient_delta = abs(Gradient_etalon_max_value - Gradient_primer_max)

                if (Histogram_delta < Histogram_delta_limit):
                    Histagram_statistics_individual.append(1)
                    Histogram_result_counter += 1
                else:
                    Histagram_statistics_individual.append(0)

                if (Gradient_delta < Gradient_delta_limit):
                    Gradient_statistics_individual.append(1)
                    Gradient_result_counter += 1
                else:
                    Gradient_statistics_individual.append(0)

                DFT_etalon_mean = np.mean(DFT_etalon[etalon_element])
                DFT_primer_mean = np.mean(DFT_primer[primer_element])
                DFT_like_percent = DFT_primer_mean / DFT_etalon_mean
                if (DFT_like_percent > 1):
                    DFT_like_percent = 2 - DFT_like_percent
                DFT_sim_sum += DFT_like_percent

                DCT_etalon_linalg_norm = np.linalg.norm(DCT_etalon[etalon_element])
                DCT_primer_linalg_norm = np.linalg.norm(DCT_primer[primer_element])
                DCT_like_percent = DCT_primer_linalg_norm / DCT_etalon_linalg_norm
                if (DCT_like_percent > 1):
                    DCT_like_percent = 2 - DCT_like_percent
                DCT_sim_sum += DCT_like_percent

                Scale_like_percent = metrics.structural_similarity(Scale_etalon[etalon_element], Scale_primer[primer_element], data_range=255)
                if (Scale_like_percent > 1):
                    Scale_like_percent = 2 - Scale_like_percent
                Scale_sim_sum += Scale_like_percent

                DFT_statistics_individual.append(DFT_like_percent)
                DCT_statistics_individual.append(DCT_like_percent)
                Scale_statistics_individual.append(Scale_like_percent)
                primer_index += 1

            Histagram_sum += Histogram_result_counter
            Gradient_sum += Gradient_result_counter
            etalon_index += 1

        amount_difference = files_amount - Etalon_amount
        Histagram_statistics.append(Histagram_sum / ((amount_difference) * Etalon_amount))
        Gradient_statistics.append(Gradient_sum / ((amount_difference) * Etalon_amount))
        DFT_statistics.append(DFT_sim_sum / ((amount_difference) * Etalon_amount))
        DCT_statistics.append(DCT_sim_sum / ((amount_difference) * Etalon_amount))
        Scale_statistics.append(Scale_sim_sum / ((amount_difference) * Etalon_amount))

    Histagram_statistics_individual_new = np.zeros((amount_difference) * folders_amount)
    Gradient_statistics_individual_new = np.zeros((amount_difference) * folders_amount)
    DFT_statistics_individual_new = np.zeros((amount_difference) * folders_amount)
    DCT_statistics_individual_new = np.zeros((amount_difference) * folders_amount)
    Scale_statistics_individual_new = np.zeros((amount_difference) * folders_amount)

    for m in range(1, folders_amount + 1, 1):
        ad = 0
        for l in range(0+(m-1)*amount_difference,amount_difference*m,1):
            for o in range(0, Etalon_amount, 1):
                v = ad + amount_difference * o + (m-1) * Etalon_amount * amount_difference
                Histagram_statistics_individual_new[l] += Histagram_statistics_individual[v]
                Gradient_statistics_individual_new[l] += Gradient_statistics_individual[v]
                DFT_statistics_individual_new[l] += DFT_statistics_individual[v]
                DCT_statistics_individual_new[l] += DCT_statistics_individual[v]
                Scale_statistics_individual_new[l] += Scale_statistics_individual[v]
            ad += 1
            Histagram_statistics_individual_new[l] = Histagram_statistics_individual_new[l] / Etalon_amount
            Gradient_statistics_individual_new[l] = Gradient_statistics_individual_new[l] / Etalon_amount
            DFT_statistics_individual_new[l] = DFT_statistics_individual_new[l] / Etalon_amount
            DCT_statistics_individual_new[l] = DCT_statistics_individual_new[l] / Etalon_amount
            Scale_statistics_individual_new[l] = Scale_statistics_individual_new[l] / Etalon_amount

    fig6, ((Img_etal, Histogram_etal, DFT_etal, DCT_etal, Gradient_etal, Scale_etal), (Img_prim, Histogram_prim, DFT_prim, DCT_prim, Gradient_prim, Scale_prim)) = plt.subplots(2, 6, label="Сравнение")
    fig7, ((Hist_indiv_stat, DFT_indiv_stat, DCT_indiv_stat, Grad_indiv_stat, Scale_indiv_stat), (Hist_stat, DFT_stat, DCT_stat, Grad_stat, Scale_stat)) = plt.subplots(2, 5, label="Итоговая статистика")
    plt.ion()

    Img_prim.set_title('Оригинал')
    IMG_array_etalon = Img_prim.imshow(IMG_etalon_array[0], cmap='gray')
    Histogram_prim.set_title('Гистограмма')
    Histogram_array_etalon, = Histogram_prim.plot(Histogram_etalon[0], color="green")
    DFT_prim.set_title('DFT')
    DFT_array_etalon = DFT_prim.imshow(DFT_etalon[0], cmap='gray', vmin=0, vmax=255)
    DCT_prim.set_title('DCT')
    DCT_array_etalon = DCT_prim.imshow(np.abs(DCT_etalon[0]), vmin=0, vmax=255)
    x_etalon = np.arange(len(Gradient_etalon[0]))
    Gradient_prim.set_title('Градиент')
    Gradient_array_etalon, = Gradient_prim.plot(x_etalon, Gradient_etalon[0], color="green")
    Scale_prim.set_title('Scale')
    Scale_array_etalon = Scale_prim.imshow(Scale_etalon[0], cmap='gray')

    Img_etal.set_title('Пример')
    IMG_array = Img_etal.imshow(IMG_primer_array[0], cmap='gray')
    Histogram_etal.set_title(f'Гистограмма:{round(Histagram_statistics_individual[0],5)}')
    Histogram_array, = Histogram_etal.plot(Histogram_primer[0], color="green")
    DFT_etal.set_title(f'DFT:{round(DFT_statistics_individual[0],5)}')
    DFT_array = DFT_etal.imshow(DFT_primer[0], cmap='gray', vmin=0, vmax=255)
    DCT_etal.set_title(f'DCT:{round(DCT_statistics_individual[0],5)}')
    DCT_array = DCT_etal.imshow(np.abs(DCT_primer[0]), vmin=0, vmax=255)
    x = np.arange(len(Gradient_primer[0]))
    Gradient_etal.set_title(f'Градиент:{round(Gradient_statistics_individual[0],5)}')
    Gradient_array, = Gradient_etal.plot(x, Gradient_primer[0], color="green")
    Scale_etal.set_title(f'Scale:{round(Scale_statistics[0],5)}')
    Scale_array = Scale_etal.imshow(Scale_primer[0], cmap='gray')

    Gradient_x_array = np.arange(len(Gradient_statistics))
    Histogram_x_array = np.arange(len(Histagram_statistics))
    DFT_x_array = np.arange(len(DFT_statistics))
    DCT_x_array = np.arange(len(DCT_statistics))
    Scale_x_array = np.arange(len(Scale_statistics))

    Gradient_x_array_individual = np.arange(len(Gradient_statistics_individual_new))
    Histogram_x_array_individual = np.arange(len(Histagram_statistics_individual_new))
    DFT_x_array_individual = np.arange(len(DFT_statistics_individual_new))
    DCT_x_array_individual = np.arange(len(DCT_statistics_individual_new))
    Scale_x_array_individual = np.arange(len(Scale_statistics_individual_new))

    fig6.set_size_inches(17, 4)
    fig6.show()
    fig7.subplots_adjust(hspace=0.5)
    fig7.set_size_inches(17, 4.5)
    fig7.show()

    folders_amount = len([f.path for f in os.scandir("IMG") if f.is_dir()])
    for t in range(0, folders_amount, 1):
        Hist_stat.plot(Histogram_x_array[0:t+1:1], Histagram_statistics[0:t+1:1], color="green")
        Hist_stat.set_title('Гистограмма')
        Hist_stat.set_xlabel("Папка")
        Hist_stat.set_ylabel("Средний процент по папке")
        Grad_stat.plot(Gradient_x_array[0:t+1:1], Gradient_statistics[0:t+1:1], color="green")
        Grad_stat.set_title('Градиент')
        DFT_stat.plot(DFT_x_array[0:t+1:1], DFT_statistics[0:t+1:1], color="green")
        DFT_stat.set_title('DFT')
        DCT_stat.plot(DCT_x_array[0:t+1:1], DCT_statistics[0:t+1:1], color="green")
        DCT_stat.set_title('DCT')
        Scale_stat.plot(Scale_x_array[0:t+1:1], Scale_statistics[0:t+1:1], color="green")
        Scale_stat.set_title('Scale')
        index = 0
        for p in range(0 + Etalon_amount * t, Etalon_amount * t + Etalon_amount, 1):
            IMG_array_etalon.set_data(IMG_etalon_array[p])
            Histogram_array_etalon.set_ydata(Histogram_etalon[p])
            DFT_array_etalon.set_data(DFT_etalon[p])
            DCT_array_etalon.set_data(DCT_etalon[p])
            Gradient_array_etalon.set_ydata(Gradient_etalon[p])
            Scale_array_etalon.set_data(Scale_etalon[p])
            files_amount = len([f for f in os.listdir(f"IMG/s{i}") if os.path.isfile(os.path.join(f"IMG/s{i}", f))])
            for m in range((p * (files_amount - Etalon_amount)), (files_amount - Etalon_amount) * (p + 1), 1):
                IMG_array.set_data(IMG_primer_array[m])
                Histogram_array.set_ydata(Histogram_primer[m])
                DFT_array.set_data(DFT_primer[m])
                DCT_array.set_data(DCT_primer[m])
                Gradient_array.set_ydata(Gradient_primer[m])
                Scale_array.set_data(Scale_primer[m])

                Histogram_etal.set_title(f'Гистограмма:{round(Histagram_statistics_individual[m], 5)}')
                DFT_etal.set_title(f'DFT:{round(DFT_statistics_individual[m], 5)}')
                DCT_etal.set_title(f'DCT:{round(DCT_statistics_individual[m], 5)}')
                Gradient_etal.set_title(f'Градиент:{round(Gradient_statistics_individual[m], 5)}')
                Scale_etal.set_title(f'Scale:{round(Scale_statistics_individual[m], 5)}')

                if p + 1 == Etalon_amount * t + Etalon_amount:
                    index_2 = index + t * (files_amount - Etalon_amount) + 1
                    Hist_indiv_stat.plot(Histogram_x_array_individual[0:index_2:1], Histagram_statistics_individual_new[0:index_2:1], color="green")
                    Hist_indiv_stat.set_title('Гистограмма')
                    Hist_indiv_stat.set_xlabel("Пример")
                    Hist_indiv_stat.set_ylabel("Средний процент по примеру")
                    Grad_indiv_stat.plot(Gradient_x_array_individual[0:index_2:1], Gradient_statistics_individual_new[0:index_2:1], color="green")
                    Grad_indiv_stat.set_title('Градиент')
                    DFT_indiv_stat.plot(DFT_x_array_individual[0:index_2:1], DFT_statistics_individual_new[0:index_2:1], color="green")
                    DFT_indiv_stat.set_title('DFT')
                    DCT_indiv_stat.plot(DCT_x_array_individual[0:index_2:1], DCT_statistics_individual_new[0:index_2:1], color="green")
                    DCT_indiv_stat.set_title('DCT')
                    Scale_indiv_stat.plot(Scale_x_array_individual[0:index_2:1], Scale_statistics_individual_new[0:index_2:1], color="green")
                    Scale_indiv_stat.set_title('Scale')
                    index += 1
                fig6.canvas.draw()
                fig6.canvas.flush_events()
                fig7.canvas.draw()
                fig7.canvas.flush_events()
    plt.pause(50)
    plt.close()

def Face_detection_manual(filename1, filename2):
    Histogram_delta_limit = 100
    Gradient_delta_limit = 80

    Histogram_result_counter = 0
    Gradient_result_counter = 0
    DFT_sim_sum = 0
    DCT_sim_sum = 0

    IMG_etalon = cv2.imread(filename1, cv2.IMREAD_GRAYSCALE)
    Histogram_etalon = Hist(IMG_etalon)
    Gradient_etalon = Grad(IMG_etalon)
    DFT_etalon = DFT(IMG_etalon)
    DCT_etalon = DCT(IMG_etalon)
    Scale_etalon = Scale(filename1)

    IMG_primer = cv2.imread(filename2, cv2.IMREAD_GRAYSCALE)
    Histogram_primer = Hist(IMG_primer)
    Gradient_primer = Grad(IMG_primer)
    DFT_primer = DFT(IMG_primer)
    DCT_primer = DCT(IMG_primer)
    Scale_primer = Scale(filename2)

    Histogram_etalon_index, Histogram_etalon_max_value = max(enumerate(Histogram_etalon), key=operator.itemgetter(1))
    Gradient_etalon_index, Gradient_etalon_max_value = max(enumerate(Gradient_etalon), key=operator.itemgetter(1))

    Histogram_primer_max = Histogram_primer[Histogram_etalon_index]
    Gradient_primer_max = Gradient_primer[Gradient_etalon_index]

    Histogram_delta = abs(Histogram_etalon_max_value - Histogram_primer_max)
    Gradient_delta = abs(Gradient_etalon_max_value - Gradient_primer_max)

    if (Histogram_delta < Histogram_delta_limit):
        Histogram_result_counter += 1

    if (Gradient_delta < Gradient_delta_limit):
        Gradient_result_counter += 1

    DFT_etalon_mean = np.mean(DFT_etalon)
    DFT_primer_mean = np.mean(DFT_primer)
    DFT_like_percent = DFT_primer_mean / DFT_etalon_mean
    if (DFT_like_percent > 1):
        DFT_like_percent = 2 - DFT_like_percent
    DFT_sim_sum += DFT_like_percent

    DCT_etalon_linalg_norm = np.linalg.norm(DCT_etalon)
    DCT_primer_linalg_norm = np.linalg.norm(DCT_primer)
    DCT_like_percent = DCT_primer_linalg_norm / DCT_etalon_linalg_norm
    if (DCT_like_percent > 1):
        DCT_like_percent = 2 - DCT_like_percent
    DCT_sim_sum += DCT_like_percent

    Scale_like_percent = metrics.structural_similarity(Scale_etalon, Scale_primer, data_range=255)

    plt.figure("Сравнение снимков выбранных вручную")
    plt.subplot(3, 6, 13)
    plt.imshow(IMG_etalon, cmap='gray')
    plt.title("Эталон")
    plt.subplot(3, 6, 14)
    plt.plot(Histogram_etalon, color="green")
    plt.title("Гистограмма")
    plt.subplot(3, 6, 15)
    plt.imshow(DFT_etalon, cmap='gray', vmin=0, vmax=255)
    plt.title("DFT")
    plt.subplot(3, 6, 16)
    plt.imshow(np.abs(DCT_etalon), vmin=0, vmax=255)
    plt.title("DCT")
    plt.subplot(3, 6, 17)
    x = np.arange(len(Gradient_etalon))
    plt.plot(x, Gradient_etalon, color="green")
    plt.title("Градиент")
    plt.subplot(3, 6, 18)
    plt.imshow(Scale_etalon,cmap='gray')
    plt.title("Scale")

    plt.subplot(3, 6, 1)
    plt.imshow(IMG_primer, cmap='gray')
    plt.title("Пример")
    plt.subplot(3, 6, 2)
    plt.plot(Histogram_primer, color="green")
    plt.title("Гистограмма")
    plt.subplot(3, 6, 3)
    plt.imshow(DFT_primer, cmap='gray', vmin=0, vmax=255)
    plt.title("DFT")
    plt.subplot(3, 6, 4)
    plt.imshow(np.abs(DCT_primer), vmin=0, vmax=255)
    plt.title("DCT")
    plt.subplot(3, 6, 5)
    x = np.arange(len(Gradient_primer))
    plt.plot(x, Gradient_primer, color="green")
    plt.title("Градиент")
    plt.subplot(3, 6, 6)
    plt.imshow(Scale_primer,cmap='gray')
    plt.title("Scale")

    if (Gradient_result_counter != 0 and Histogram_result_counter != 0 and DFT_like_percent >=0.5 and DCT_like_percent >=0.5 and Scale_like_percent >=0.5):
        check = "Совпадает"
        check_color = "green"
    else:
        check = "Не совпадает"
        check_color = "red"

    plt.text(0.4, 0.5, check, fontsize=14, color=check_color, transform=plt.gcf().transFigure)
    plt.show()


def Face_detection_manual_directory(filename, Etalon_amount):
    DCT_statistics = []
    DFT_statistics = []
    Scale_statistics = []
    Histagram_statistics = []
    Gradient_statistics = []

    DCT_statistics_individual = []
    DFT_statistics_individual = []
    Scale_statistics_individual = []
    Histagram_statistics_individual = []
    Gradient_statistics_individual = []

    Histogram_delta_limit = 230
    Gradient_delta_limit = 80

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

    for i in range(1, 2, 1):
        Histagram_sum = 0
        Gradient_sum = 0
        DFT_sim_sum = 0
        DCT_sim_sum = 0
        Scale_sim_sum = 0
        Filepath = [f for f in os.listdir(f"{filename}") if os.path.isfile(os.path.join(f"{filename}", f))]
        files_amount = len([f for f in os.listdir(f"{filename}") if os.path.isfile(os.path.join(f"{filename}", f))])
        for j in range(1, Etalon_amount + 1, 1):
            Histogram_result_counter = 0
            Gradient_result_counter = 0
            Etalon_filepath = f"{filename}/{Filepath[j-1]}"
            IMG_etalon = cv2.imread(Etalon_filepath, cv2.IMREAD_GRAYSCALE)
            IMG_etalon_array.append(IMG_etalon)
            Histogram_etalon.append(Hist(IMG_etalon))
            Gradient_etalon.append(Grad(IMG_etalon))
            DFT_etalon.append(DFT(IMG_etalon))
            DCT_etalon.append(DCT(IMG_etalon))
            Scale_etalon.append(Scale(Etalon_filepath))
            for k in range(Etalon_amount + 1, files_amount + 1, 1):
                Primer_filepath = f"{filename}/{Filepath[k-1]}"
                IMG_primer = cv2.imread(Primer_filepath, cv2.IMREAD_GRAYSCALE)
                IMG_primer_array.append(IMG_primer)
                Histogram_primer.append(Hist(IMG_primer))
                Gradient_primer.append(Grad(IMG_primer))
                DFT_primer.append(DFT(IMG_primer))
                DCT_primer.append(DCT(IMG_primer))
                Scale_primer.append(Scale(Primer_filepath))

                Histogram_etalon_index, Histogram_etalon_max_value = max(enumerate(Histogram_etalon[j - 1 + Etalon_amount * (i - 1)]), key=operator.itemgetter(1))
                Gradient_etalon_index, Gradient_etalon_max_value = max(enumerate(Gradient_etalon[j - 1 + Etalon_amount * (i - 1)]), key=operator.itemgetter(1))

                Histogram_primer_max = Histogram_primer[k - Etalon_amount - 1 + (files_amount - Etalon_amount) * (i - 1)][Histogram_etalon_index]
                Gradient_primer_max = Gradient_primer[k - Etalon_amount - 1 + (files_amount - Etalon_amount) * (i - 1)][Gradient_etalon_index]

                Histogram_delta = abs(Histogram_etalon_max_value - Histogram_primer_max)
                Gradient_delta = abs(Gradient_etalon_max_value - Gradient_primer_max)

                if (Histogram_delta < Histogram_delta_limit):
                    Histagram_statistics_individual.append(1)
                    Histogram_result_counter += 1
                else:
                    Histagram_statistics_individual.append(0)

                if (Gradient_delta < Gradient_delta_limit):
                    Gradient_statistics_individual.append(1)
                    Gradient_result_counter += 1
                else:
                    Gradient_statistics_individual.append(0)

                DFT_etalon_mean = np.mean(DFT_etalon[j - 1 + Etalon_amount * (i - 1)])
                DFT_primer_mean = np.mean(DFT_primer[k - Etalon_amount - 1 + (files_amount - Etalon_amount) * (i - 1)])
                DFT_like_percent = DFT_primer_mean / DFT_etalon_mean
                if (DFT_like_percent > 1):
                    DFT_like_percent = 2 - DFT_like_percent
                DFT_sim_sum += DFT_like_percent

                DCT_etalon_linalg_norm = np.linalg.norm(DCT_etalon[j - 1 + Etalon_amount * (i - 1)])
                DCT_primer_linalg_norm = np.linalg.norm(DCT_primer[k - Etalon_amount - 1 + (files_amount - Etalon_amount) * (i - 1)])
                DCT_like_percent = DCT_primer_linalg_norm / DCT_etalon_linalg_norm
                if (DCT_like_percent > 1):
                    DCT_like_percent = 2 - DCT_like_percent
                DCT_sim_sum += DCT_like_percent

                Scale_like_percent = metrics.structural_similarity(Scale_etalon[j - 1 + Etalon_amount * (i - 1)],
                                                                   Scale_primer[k - Etalon_amount - 1 + (files_amount - Etalon_amount) * (i - 1)],
                                                                   data_range=255)
                if (Scale_like_percent > 1):
                    Scale_like_percent = 2 - Scale_like_percent
                Scale_sim_sum += Scale_like_percent
                DFT_statistics_individual.append(DFT_like_percent)
                DCT_statistics_individual.append(DCT_like_percent)
                Scale_statistics_individual.append(Scale_like_percent)
            Histagram_sum += Histogram_result_counter
            Gradient_sum += Gradient_result_counter
            amount_difference = files_amount - Etalon_amount
        Histagram_statistics.append(Histagram_sum / ((files_amount - Etalon_amount) * Etalon_amount))
        Gradient_statistics.append(Gradient_sum / ((files_amount - Etalon_amount) * Etalon_amount))
        DFT_statistics.append(DFT_sim_sum / ((files_amount - Etalon_amount) * Etalon_amount))
        DCT_statistics.append(DCT_sim_sum / ((files_amount - Etalon_amount) * Etalon_amount))
        Scale_statistics.append(Scale_sim_sum / ((files_amount - Etalon_amount) * Etalon_amount))
    Histagram_statistics_individual_new = np.zeros((amount_difference) * 1)
    Gradient_statistics_individual_new = np.zeros((amount_difference) * 1)
    DFT_statistics_individual_new = np.zeros((amount_difference) * 1)
    DCT_statistics_individual_new = np.zeros((amount_difference) * 1)
    Scale_statistics_individual_new = np.zeros((amount_difference) * 1)
    for m in range(1, 1 + 1, 1):
        ad = 0
        for l in range(0+(m-1)*amount_difference,amount_difference*m,1):
            for o in range(0, Etalon_amount, 1):
                v = ad + amount_difference * o + (m-1) * Etalon_amount * amount_difference
                Histagram_statistics_individual_new[l] += Histagram_statistics_individual[v]
                Gradient_statistics_individual_new[l] += Gradient_statistics_individual[v]
                DFT_statistics_individual_new[l] += DFT_statistics_individual[v]
                DCT_statistics_individual_new[l] += DCT_statistics_individual[v]
                Scale_statistics_individual_new[l] += Scale_statistics_individual[v]
            ad += 1
            Histagram_statistics_individual_new[l] = Histagram_statistics_individual_new[l] / Etalon_amount
            Gradient_statistics_individual_new[l] = Gradient_statistics_individual_new[l] / Etalon_amount
            DFT_statistics_individual_new[l] = DFT_statistics_individual_new[l] / Etalon_amount
            DCT_statistics_individual_new[l] = DCT_statistics_individual_new[l] / Etalon_amount
            Scale_statistics_individual_new[l] = Scale_statistics_individual_new[l] / Etalon_amount

    fig1, ((Img_etal, Histogram_etal, DFT_etal, DCT_etal, Gradient_etal, Scale_etal), (Img_prim, Histogram_prim, DFT_prim, DCT_prim, Gradient_prim, Scale_prim)) = plt.subplots(2, 6, label="Сравнение")
    fig3, ((Hist_indiv_stat, Grad_indiv_stat, DFT_indiv_stat, DCT_indiv_stat, Scale_indiv_stat), (Hist_stat, Grad_stat, DFT_stat, DCT_stat, Scale_stat)) = plt.subplots(2, 5, label="Итоговая статистика")
    plt.ion()

    Img_etal.set_title('Пример')
    IMG_array = Img_etal.imshow(IMG_primer_array[0], cmap='gray')
    Histogram_etal.set_title(f'Гистограмма:{Histagram_statistics_individual[0]}')
    Histogram_array, = Histogram_etal.plot(Histogram_primer[0], color="green")
    DFT_etal.set_title(f'DFT:{DFT_statistics_individual[0]}')
    DFT_array = DFT_etal.imshow(DFT_primer[0], cmap='gray', vmin=0, vmax=255)
    DCT_etal.set_title(f'DCT:{DCT_statistics_individual[0]}')
    DCT_array = DCT_etal.imshow(np.abs(DCT_primer[0]), vmin=0, vmax=255)
    x = np.arange(len(Gradient_primer[0]))
    Gradient_etal.set_title(f'Градиент:{Gradient_statistics_individual[0]}')
    Gradient_array, = Gradient_etal.plot(x, Gradient_primer[0], color="green")
    Scale_etal.set_title(f'Scale:{Scale_statistics[0]}')
    Scale_array = Scale_etal.imshow(Scale_primer[0], cmap='gray')

    Img_prim.set_title('Оригинал')
    IMG_array_etalon = Img_prim.imshow(IMG_etalon_array[0], cmap='gray')
    Histogram_prim.set_title('Гистограмма')
    Histogram_array_etalon, = Histogram_prim.plot(Histogram_etalon[0], color="green")
    DFT_prim.set_title('DFT')
    DFT_array_etalon = DFT_prim.imshow(DFT_etalon[0], cmap='gray', vmin=0, vmax=255)
    DCT_prim.set_title('DCT')
    DCT_array_etalon = DCT_prim.imshow(np.abs(DCT_etalon[0]), vmin=0, vmax=255)
    x_etalon = np.arange(len(Gradient_etalon[0]))
    Gradient_prim.set_title('Градиент')
    Gradient_array_etalon, = Gradient_prim.plot(x_etalon, Gradient_etalon[0], color="green")
    Scale_prim.set_title('Scale')
    Scale_array_etalon = Scale_prim.imshow(Scale_etalon[0], cmap='gray')

    Gradient_x_array = np.arange(len(Gradient_statistics))
    Histogram_x_array = np.arange(len(Histagram_statistics))
    DFT_x_array = np.arange(len(DFT_statistics))
    DCT_x_array = np.arange(len(DCT_statistics))
    Scale_x_array = np.arange(len(Scale_statistics))

    Gradient_x_array_individual = np.arange(len(Gradient_statistics_individual_new))
    Histogram_x_array_individual = np.arange(len(Histagram_statistics_individual_new))
    DFT_x_array_individual = np.arange(len(DFT_statistics_individual_new))
    DCT_x_array_individual = np.arange(len(DCT_statistics_individual_new))
    Scale_x_array_individual = np.arange(len(Scale_statistics_individual_new))

    fig1.set_size_inches(17, 4)
    fig1.show()
    fig3.set_size_inches(17, 4.5)
    fig3.subplots_adjust(hspace=0.5)
    fig3.show()

    for t in range(0, 1, 1):
        Hist_stat.plot(Histogram_x_array[0:t+1:1], Histagram_statistics[0:t+1:1], color="green")
        Hist_stat.set_xlabel("Папка s")
        Hist_stat.set_ylabel("Средний процент по папке")
        Grad_stat.plot(Gradient_x_array[0:t+1:1], Gradient_statistics[0:t+1:1], color="green")
        DFT_stat.plot(DFT_x_array[0:t+1:1], DFT_statistics[0:t+1:1], color="green")
        DCT_stat.plot(DCT_x_array[0:t+1:1], DCT_statistics[0:t+1:1], color="green")
        Scale_stat.plot(Scale_x_array[0:t+1:1], Scale_statistics[0:t+1:1], color="green")
        index = 0
        for p in range(0 + Etalon_amount * t, Etalon_amount * t + Etalon_amount, 1):
            IMG_array_etalon.set_data(IMG_etalon_array[p])
            Histogram_array_etalon.set_ydata(Histogram_etalon[p])
            DFT_array_etalon.set_data(DFT_etalon[p])
            DCT_array_etalon.set_data(DCT_etalon[p])
            Gradient_array_etalon.set_ydata(Gradient_etalon[p])
            Scale_array_etalon.set_data(Scale_etalon[p])
            Files_amount = len([f for f in os.listdir(f"{filename}") if os.path.isfile(os.path.join(f"{filename}", f))])
            for m in range((0 + p * (Files_amount - Etalon_amount)), (Files_amount - Etalon_amount) * (p + 1), 1):
                IMG_array.set_data(IMG_primer_array[m])
                Histogram_array.set_ydata(Histogram_primer[m])
                DFT_array.set_data(DFT_primer[m])
                DCT_array.set_data(DCT_primer[m])
                Gradient_array.set_ydata(Gradient_primer[m])
                Scale_array.set_data(Scale_primer[m])
                Histogram_etal.set_title(f'Гистограмма:{Histagram_statistics_individual[m]}')
                DFT_etal.set_title(f'DFT:{DFT_statistics_individual[m]}')
                DCT_etal.set_title(f'DCT:{DCT_statistics_individual[m]}')
                Gradient_etal.set_title(f'Градиент:{Gradient_statistics_individual[m]}')
                Scale_etal.set_title(f'Scale:{Scale_statistics_individual[m]}')
                if p + 1 == Etalon_amount * t + Etalon_amount:
                    index_2 = index + t * (Files_amount - Etalon_amount)+1
                    Hist_indiv_stat.plot(Histogram_x_array_individual[0:index_2:1], Histagram_statistics_individual_new[0:index_2:1], color="green")
                    Hist_indiv_stat.set_title('Гистограмма')
                    Hist_indiv_stat.set_xlabel("Пример")
                    Hist_indiv_stat.set_ylabel("Средний процент по примеру")
                    Grad_indiv_stat.plot(Gradient_x_array_individual[0:index_2:1], Gradient_statistics_individual_new[0:index_2:1], color="green")
                    Grad_indiv_stat.set_title('Градиент')
                    DFT_indiv_stat.plot(DFT_x_array_individual[0:index_2:1], DFT_statistics_individual_new[0:index_2:1], color="green")
                    DFT_indiv_stat.set_title('DFT')
                    DCT_indiv_stat.plot(DCT_x_array_individual[0:index_2:1], DCT_statistics_individual_new[0:index_2:1], color="green")
                    DCT_indiv_stat.set_title('DCT')
                    Scale_indiv_stat.plot(Scale_x_array_individual[0:index_2:1], Scale_statistics_individual_new[0:index_2:1], color="green")
                    Scale_indiv_stat.set_title('Scale')
                    index += 1
                fig1.canvas.draw()
                fig1.canvas.flush_events()
                fig3.canvas.draw()
                fig3.canvas.flush_events()
    plt.waitforbuttonpress()
    plt.close()


App = tk.Tk()
App.title('Опознавание лиц №3')
App.geometry('300x200')

Etalon_amount_label = tk.Label(App, text="Количество эталонов:")
Etalon_amount_label.pack()
Etalon_amount_input = tk.Entry(App)
Etalon_amount_input.pack()

App_button = tk.Button(App, text="Построить графики", command=lambda: Etalon_amount_take(int(0)))
App_button.pack()

App_button = tk.Button(App, text="Вручную выбрать эталон и пример", command=Manual_choose)
App_button.pack()

App_button = tk.Button(App, text="Вручную выбрать папку", command=Manual_choose_directory)
App_button.pack()

App_button = tk.Button(App, text="Выбирать нечетные эталоны", command=lambda: Etalon_amount_take(int(1)))
App_button.pack()

App_button = tk.Button(App, text="Выбирать четные эталоны", command=lambda: Etalon_amount_take(int(2)))
App_button.pack()

# Запускаем главный цикл обработки событий
App.mainloop()