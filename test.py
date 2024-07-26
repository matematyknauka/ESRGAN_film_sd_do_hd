import sys
import cv2
import torch
import numpy as np
from moviepy.editor import VideoFileClip
from PIL import Image
from torchvision.transforms import ToTensor

# Dodaj ścieżkę do repozytorium ESRGAN
# sys.path.append('/ESRGAN_sd_do_hd') nie wiem czy zadziała

# Importuj definicję modelu
import RRDBNet_arch as arch

class ESRGAN:
    def __init__(self, model_path):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.load_model(model_path)
        
    def load_model(self, model_path):
        model = arch.RRDBNet(3, 3, 64, 23, gc=32)
        model.load_state_dict(torch.load(model_path), strict=True)
        model.eval()
        model.to(self.device)
        return model
    
    def enhance_frame(self, frame):
        img = frame.astype(np.float32) / 255.0  # Normalizacja
        img = torch.from_numpy(np.transpose(img[:, :, [2, 1, 0]], (2, 0, 1))).float()
        img_LR = img.unsqueeze(0).to(self.device)

        with torch.no_grad():
            output = self.model(img_LR).data.squeeze().float().cpu().clamp_(0, 1).numpy()
        output = np.transpose(output[[2, 1, 0], :, :], (1, 2, 0))
        output = (output * 255.0).round().astype(np.uint8)
        return output

def enhance_video(input_file, output_file, model_path):
    video = VideoFileClip(input_file)
    esrgan = ESRGAN(model_path)
    
    def process_frame(frame):
        frame = np.array(frame)  # Konwertuj do numpy array
        enhanced_frame = esrgan.enhance_frame(frame)  # Popraw jakość
        return enhanced_frame  # Zwróć numpy array

    # Przetwarzaj wideo
    video_hd = video.fl_image(process_frame)
    video_hd.write_videofile(output_file, codec='libx264')

# Ścieżki do plików
input_file = "LR/Kamil_sd.mp4"  # Ścieżka do sd
output_file = "results/Kamil_hd.mp4"  # Ścieżka do hd
model_path = "models/RRDB_ESRGAN_x4.pth"  # Ścieżka do modelu ESRGAN

# Uruchom przetwarzanie wideo
enhance_video(input_file, output_file, model_path)
