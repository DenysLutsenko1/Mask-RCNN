

import mrcnn
import mrcnn.config
import mrcnn.model
import mrcnn.visualize
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt



CLASS_NAMES = ['BG', 'cobble']

class SimpleConfig(mrcnn.config.Config):
    NAME = "coco_inference"

    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    
    NUM_CLASSES = len(CLASS_NAMES)

class Metrics:
    def __init__(self, weights_path) -> None:
        self.weights_path = weights_path
        self.masks = []

        self.roughness_list = []
        self.elongation_list = []
        self.angularity_list = []
    
    def predict_masks(self, image_path):
        model = mrcnn.model.MaskRCNN(mode="inference", 
                             config=SimpleConfig(),
                             model_dir=os.getcwd())
        
        model.load_weights(filepath=os.path.join(self.weights_path), 
                        by_name=True)

        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        r = model.detect([image], verbose=0)
        r = r[0]

        num_objects = r["masks"].shape[-1]

        masks = []
        for i in range(num_objects):
            masks.append (r["masks"][:, :, i])

        return masks
    
    def load_masks(self, images_folder):
        for image in os.listdir(images_folder):
            self.masks.append(self.predict_masks(os.path.join(images_folder, image)))

    import numpy as np

    def calculate_elongation(self, boundary_points):
        x_min = np.min(boundary_points[:, 0])
        x_max = np.max(boundary_points[:, 0])
        y_min = np.min(boundary_points[:, 1])
        y_max = np.max(boundary_points[:, 1])

        width = x_max - x_min
        height = y_max - y_min

        Dminor, Dmajor = min(width, height), max(width, height)

        return Dminor / Dmajor


    def calculate_angularity(self, boundary_points):
        angles = np.arctan2(np.diff(boundary_points[:, 1]), np.diff(boundary_points[:, 0]))
        angle_diffs = np.abs(np.diff(angles))
        AI = np.sum(angle_diffs) / (2 * np.pi) - 1
        return AI

    def smooth_boundary_fourier(self, boundary_points):
        n_components = len(boundary_points)
        complex_boundary = boundary_points[:, 0] + 1j * boundary_points[:, 1]
        
        fft_boundary = np.fft.fft(complex_boundary)
        fft_boundary[n_components:] = 0

        smoothed_complex_boundary = np.fft.ifft(fft_boundary)
        smoothed_boundary = np.column_stack((smoothed_complex_boundary.real, smoothed_complex_boundary.imag))
        
        return smoothed_boundary

    def calculate_roughness(self, boundary_points):
        smoothed_boundary = self.smooth_boundary_fourier(boundary_points)

        deviations = np.linalg.norm(boundary_points - smoothed_boundary, axis=1)
        Ra = np.mean(np.abs(deviations))
        return Ra
    
    def extract_boundary_points_from_mask(self, binary_mask):
        binary_mask = binary_mask.astype(np.uint8) * 255

        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        largest_contour = max(contours, key=cv2.contourArea)
        boundary_points = np.squeeze(largest_contour)
        
        return boundary_points


    def generate_roughness_list(self):
        self.roughness_list = []

        for mask_file in self.masks:
            for mask in mask_file:
                boundary_points = self.extract_boundary_points_from_mask(mask)
                self.roughness_list.append(self.calculate_roughness(boundary_points))
    
    def generate_angularity_list(self):
        self.angularity_list = []

        for mask_file in self.masks:
            for mask in mask_file:
                boundary_points = self.extract_boundary_points_from_mask(mask)
                self.angularity_list.append(self.calculate_angularity(boundary_points))
    
    def generate_elongation_list(self):
        self.elongation_list = []

        for mask_file in self.masks:
            for mask in mask_file:
                boundary_points = self.extract_boundary_points_from_mask(mask)
                self.elongation_list.append(self.calculate_elongation(boundary_points))

    def plot_multiple_histograms(self, data_lists, titles, bins=10):
        """
        Функция для построения нескольких гистограмм в одном окне.
        
        :param data_lists: list of lists, массивы значений для визуализации
        :param titles: list of str, заголовки для каждой гистограммы
        :param bins: int, количество интервалов (столбцов) для гистограммы, по умолчанию 10
        """
        num_graphs = len(data_lists)
        plt.figure(figsize=(15, 5))  # Увеличим размер окна для трех графиков
        
        for i in range(num_graphs):
            plt.subplot(1, num_graphs, i + 1)
            plt.hist(data_lists[i], bins=bins, edgecolor='black')  # Построение гистограммы
            plt.title(titles[i])
            plt.xlabel('Value Intervals')
            plt.ylabel('Frequency')
        
        plt.tight_layout()
        plt.show()
    
    def generate_metrics():
        metrics.generate_roughness_list()
        metrics.generate_angularity_list()
        metrics.generate_elongation_list()

metrics = Metrics(os.path.join("weights", "cobble_mask_rcnn_trained.h5"))
metrics.load_masks(os.path.join("metrics", "images")) #after your training completed change foto in folder metrics (but dont do this)
metrics.generate_metrics()
metrics.plot_multiple_histograms([metrics.roughness_list, metrics.angularity_list, metrics.elongation_list],['Roughness Distribution', 'Angularity Distribution', 'Elongation Distribution'],bins=15)


# датасеты , точность  ,