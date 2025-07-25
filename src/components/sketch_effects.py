from src.exceptions import CustomException
from src.logger import logging

from src.entity.config import ConfigEntity, AnimeGANConfig, GlitchEffectConfig,PencilSketchConfig
from src.entity.artifact import AnimeGANArtifact,GlitchEffectArtifact,PencilSketchArtifact

from src.utils import generate_unique_filename1

import os
import sys
import cv2
import typing
import numpy as np
import onnxruntime as ort


import time
from PIL import ImageFont, ImageDraw, Image
import random
import dlib
import os
import argparse
import imutils
import imutils.face_utils
from imutils.face_utils import FACIAL_LANDMARKS_68_IDXS
import sys

class AnimeSketch:
    def __init__(self):
        try:
            self.anime_gan_config = AnimeGANConfig(config=ConfigEntity())
            logging.info("AnimeSketch initialized with AnimeGANConfig.")
        except Exception as e:
            logging.error("Failed to initialize AnimeSketch.", exc_info=True)
            raise CustomException(e, sys)
    
    def to_32s(self, x):
        return 256 if x < 256 else x - x % 32

    def process_frame(self, frame: np.ndarray, x32: bool = True) -> np.ndarray:
        try:
            h, w = frame.shape[:2]
            if x32:
                frame = cv2.resize(
                    frame, 
                    (self.to_32s(int(w * self.anime_gan_config.downsize_ratio)),
                     self.to_32s(int(h * self.anime_gan_config.downsize_ratio)))
                )
            frame = frame.astype(np.float32) / 127.5 - 1.0
            logging.debug("Frame processed successfully.")
            return frame
        except Exception as e:
            logging.error("Error processing frame.", exc_info=True)
            raise CustomException(e, sys)

    def post_process(self, frame: np.ndarray, wh: typing.Tuple[int, int]) -> np.ndarray:
        try:
            frame = (frame.squeeze() + 1.) / 2 * 255
            frame = frame.astype(np.uint8)
            frame = cv2.resize(frame, (wh[0], wh[1]))
            logging.debug("Post processing completed.")
            return frame
        except Exception as e:
            logging.error("Error during post processing.", exc_info=True)
            raise CustomException(e, sys)

    def run(self, model_path: str, frame: np.ndarray) -> np.ndarray:
        try:
            if not os.path.exists(model_path):
                msg = f"Model doesn't exist at {model_path}"
                logging.error(msg)
                raise CustomException(msg, sys)

            providers = ['CUDAExecutionProvider'] if ort.get_device() == "GPU" else ['CPUExecutionProvider']
            ort_sess = ort.InferenceSession(model_path, providers=providers)

            image = self.process_frame(frame)
            input_name = ort_sess.get_inputs()[0].name
            outputs = ort_sess.run(None, {input_name: np.expand_dims(image, axis=0)})
            processed = self.post_process(outputs[0], frame.shape[:2][::-1])
            logging.info(f"Model inference successful for {model_path}.")
            return processed
        except Exception as e:
            logging.error(f"Error running inference with model at {model_path}.", exc_info=True)
            raise CustomException(e, sys)

    def initiate_anime_sketch(self, image_path: str) -> AnimeGANArtifact:
        try:
            image = cv2.imread(image_path)
            if image is None:
                msg = f"Image not found at {image_path}"
                logging.error(msg)
                raise CustomException(msg, sys)

            result1 = self.run(self.anime_gan_config.model_hayao_path, image)
            result2 = self.run(self.anime_gan_config.model_paprika_path, image)
            result3 = self.run(self.anime_gan_config.model_shinkai_path, image)

            output_dir = os.path.join(
                self.anime_gan_config.output_dir,
                self.anime_gan_config.anime_sketch_dir
            )
            os.makedirs(output_dir, exist_ok=True)

            hayao_path = os.path.join(output_dir, generate_unique_filename1("hayao", ".jpg"))
            paprika_path = os.path.join(output_dir, generate_unique_filename1("paprika", ".jpg"))
            shinkai_path = os.path.join(output_dir, generate_unique_filename1("shinkai", ".jpg"))

            cv2.imwrite(hayao_path, result1)
            cv2.imwrite(paprika_path, result2)
            cv2.imwrite(shinkai_path, result3)

            logging.info("Anime sketch generation complete.")
            return AnimeGANArtifact(
                model_hayao_img_path=hayao_path,
                model_paprika_img_path=paprika_path,
                model_shinkai_img_path=shinkai_path
            )
        except Exception as e:
            logging.error("Error generating anime sketch.", exc_info=True)
            raise CustomException(e, sys)


class GlitchEffect:
    def __init__(self):
        try:
            logging.info("Initializing GlitchEffect class")
            
            self.glitch_effect_config = GlitchEffectConfig(config=ConfigEntity())
            logging.info("GlitchEffect configuration loaded successfully")
            
            # Load face detection DNN model
            if not os.path.exists(self.glitch_effect_config.deploy_protox_apth):
                logging.error(f"Deploy prototxt file not found: {self.glitch_effect_config.deploy_protox_apth}")
                raise CustomException(f"Deploy prototxt file not found: {self.glitch_effect_config.deploy_protox_apth}", sys)
                
            if not os.path.exists(self.glitch_effect_config.deploy_caffemodel_path):
                logging.error(f"Deploy caffemodel file not found: {self.glitch_effect_config.deploy_caffemodel_path}")
                raise CustomException(f"Deploy caffemodel file not found: {self.glitch_effect_config.deploy_caffemodel_path}", sys)
            
            self.face_net_dnn = cv2.dnn.readNetFromCaffe(
                self.glitch_effect_config.deploy_protox_apth,
                self.glitch_effect_config.deploy_caffemodel_path
            )
            logging.info("Face detection DNN model loaded successfully")
            
            # Load facial landmark predictor
            if not os.path.exists(self.glitch_effect_config.shape_predictor_path):
                logging.error(f"Shape predictor file not found: {self.glitch_effect_config.shape_predictor_path}")
                raise CustomException(f"Shape predictor file not found: {self.glitch_effect_config.shape_predictor_path}", sys)
                
            self.face_landmarker = dlib.shape_predictor(self.glitch_effect_config.shape_predictor_path)
            logging.info("Facial landmark predictor loaded successfully")
            
            self.models_loaded = True
            logging.info("All models loaded successfully")

            self.VHS_TEXT_PC = 0.0625
            self.UNI_FONT_EXTRA_PTS_PC = 1/24
            self.vhs_font_loaded = True
            
            logging.info("GlitchEffect initialization completed successfully")
            
        except Exception as e:
            logging.error(f"Error during GlitchEffect initialization: {str(e)}")
            raise CustomException(e, sys)

    def find_faces_dnn(self, img, min_confidence=0.6):
        """Finds faces within the given image using OpenCV's DNN method."""
        try:
            logging.info(f"Starting face detection with min_confidence: {min_confidence}")
            
            if not self.models_loaded:
                logging.warning("Models not loaded, returning empty arrays")
                return (np.array([]), np.array([]))
            
            if img is None:
                logging.error("Input image is None")
                raise CustomException("Input image is None", sys)
            
            (h, w) = img.shape[:2]
            logging.info(f"Input image dimensions: {w}x{h}")
            
            blob = cv2.dnn.blobFromImage(cv2.resize(img, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
            
            self.face_net_dnn.setInput(blob)
            detections = self.face_net_dnn.forward()
            
            confidences = []
            coords = []
            
            for i in range(0, detections.shape[2]):
                confidence = detections[0, 0, i, 2]
                
                if confidence < min_confidence:
                    continue
                    
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                box = box.astype("int")
                
                # Ensure bounds are within image
                box[0] = max(0, min(box[0], w - 1))
                box[1] = max(0, min(box[1], h - 1))
                box[2] = max(0, min(box[2], w - 1))
                box[3] = max(0, min(box[3], h - 1))
                
                confidences.append(confidence)
                coords.append(box)
            
            logging.info(f"Face detection completed. Found {len(confidences)} faces")
            return (np.array(confidences, dtype="float"), np.array(coords, dtype="int32"))
            
        except Exception as e:
            logging.error(f"Error in face detection: {str(e)}")
            raise CustomException(e, sys)
    
    def find_facial_landmarks(self, face_boxes, img):
        """Returns facial landmark points for detected faces."""
        try:
            logging.info(f"Finding facial landmarks for {len(face_boxes)} faces")
            
            if not self.models_loaded or len(face_boxes) == 0:
                logging.warning("Models not loaded or no face boxes provided")
                return np.array([])
            
            if img is None:
                logging.error("Input image is None")
                raise CustomException("Input image is None", sys)
            
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            all_pts = []
            
            for i, box in enumerate(face_boxes):
                logging.info(f"Processing landmarks for face {i+1}")
                
                # Adjust box for better landmark detection
                box = box.copy()
                box[1] = int(box[1] * 1.15)
                box[2] = int(box[2] * 1.05)
                
                rect = dlib.rectangle(box[0], box[1], box[2], box[3])
                shape = self.face_landmarker(gray, rect)
                all_pts.append(imutils.face_utils.shape_to_np(shape, dtype="int32"))
            
            logging.info(f"Facial landmarks detection completed for {len(all_pts)} faces")
            return np.array(all_pts, dtype="int32")
            
        except Exception as e:
            logging.error(f"Error in facial landmarks detection: {str(e)}")
            raise CustomException(e, sys)

    def find_blackout_bar(self, face_pts, face_box):
        """Creates a blackout bar over the eyes."""
        try:
            logging.info("Creating blackout bar over eyes")
            
            if face_pts is None or len(face_pts) == 0:
                logging.error("No facial landmarks provided")
                raise CustomException("No facial landmarks provided", sys)
            
            (lStart, lEnd) = FACIAL_LANDMARKS_68_IDXS["left_eye"]
            (rStart, rEnd) = FACIAL_LANDMARKS_68_IDXS["right_eye"]
            
            left_pts = face_pts[lStart:lEnd]
            right_pts = face_pts[rStart:rEnd]
            
            left_center = left_pts.mean(axis=0).astype("int")
            right_center = right_pts.mean(axis=0).astype("int")
            
            # Fix: Convert to float and ensure proper format for OpenCV
            eyes_center = (float((left_center[0] + right_center[0]) / 2),
                        float((left_center[1] + right_center[1]) / 2))
            
            # Calculate angle between eyes
            dY = right_center[1] - left_center[1]
            dX = right_center[0] - left_center[0]
            angle = np.degrees(np.arctan2(dY, dX)) - 180
            
            # Create blackout bar dimensions
            face_w = face_box[2] - face_box[0]
            face_h = face_box[3] - face_box[1]
            width = float(face_w * 1.20)
            height = float(face_h * 0.25)
            
            # Fix: Ensure all values are floats for OpenCV
            rrect = (eyes_center, (width, height), float(angle))
            blackout_bar = cv2.boxPoints(rrect).astype("int32")
            
            logging.info("Blackout bar created successfully")
            return blackout_bar
            
        except Exception as e:
            logging.error(f"Error creating blackout bar: {str(e)}")
            raise CustomException(e, sys)
    
    def strip_shift_face(self, face_box, img, orientation="h", n_strips=16, move_min_pc=0.1, move_max_pc=0.4):
        """Applies face glitch effect by shifting strips."""
        try:
            logging.info(f"Applying strip shift effect with orientation: {orientation}, strips: {n_strips}")
            
            if img is None:
                logging.error("Input image is None")
                raise CustomException("Input image is None", sys)
            
            startX, startY, endX, endY = face_box
            face_h = endY - startY
            face_w = endX - startX
            img_h, img_w = img.shape[:2]
            
            if face_h <= 0 or face_w <= 0:
                logging.error(f"Invalid face dimensions: width={face_w}, height={face_h}")
                raise CustomException(f"Invalid face dimensions: width={face_w}, height={face_h}", sys)
            
            if orientation == "h":
                strip_width = face_h // n_strips
            else:
                strip_width = face_w // n_strips
            
            if strip_width == 0:
                strip_width = 1
                if orientation == "h":
                    n_strips = face_h
                else:
                    n_strips = face_w
                logging.warning(f"Adjusted strip width to 1 and n_strips to {n_strips}")
            
            if orientation == "h":
                for i in range(n_strips):
                    strip_startY = startY + i * strip_width
                    strip_endY = min(startY + i * strip_width + strip_width, endY)
                    shift = int(random.uniform(move_min_pc * face_w, move_max_pc * face_w))
                    shift *= random.choice([-1, 1])
                    
                    strip_startX = max(0, min(startX + shift, img_w - 1))
                    strip_endX = max(0, min(endX + shift, img_w))
                    
                    # Boundary checks
                    shrink_l, shrink_r = 0, 0
                    if startX + shift < 0:
                        shrink_l = abs(startX + shift)
                    if endX + shift > img_w:
                        shrink_r = (endX + shift) - img_w
                    
                    src_startX = max(0, startX + shrink_l)
                    src_endX = min(img_w, endX - shrink_r)
                    
                    if strip_startX < strip_endX and src_startX < src_endX and strip_startY < strip_endY:
                        width_to_copy = min(strip_endX - strip_startX, src_endX - src_startX)
                        img[strip_startY:strip_endY, strip_startX:strip_startX + width_to_copy] = img[strip_startY:strip_endY, src_startX:src_startX + width_to_copy]
            
            else:  # Vertical
                for i in range(n_strips):
                    strip_startX = startX + i * strip_width
                    strip_endX = min(startX + i * strip_width + strip_width, endX)
                    shift = int(random.uniform(move_min_pc * face_h, move_max_pc * face_h))
                    shift *= random.choice([-1, 1])
                    
                    strip_startY = max(0, min(startY + shift, img_h - 1))
                    strip_endY = max(0, min(endY + shift, img_h))
                    
                    # Boundary checks
                    shrink_t, shrink_b = 0, 0
                    if startY + shift < 0:
                        shrink_t = abs(startY + shift)
                    if endY + shift > img_h:
                        shrink_b = (endY + shift) - img_h
                    
                    src_startY = max(0, startY + shrink_t)
                    src_endY = min(img_h, endY - shrink_b)
                    
                    if strip_startY < strip_endY and src_startY < src_endY and strip_startX < strip_endX:
                        height_to_copy = min(strip_endY - strip_startY, src_endY - src_startY)
                        img[strip_startY:strip_startY + height_to_copy, strip_startX:strip_endX] = img[src_startY:src_startY + height_to_copy, strip_startX:strip_endX]
            
            logging.info("Strip shift effect applied successfully")
            
        except Exception as e:
            logging.error(f"Error applying strip shift effect: {str(e)}")
            raise CustomException(e, sys)

    def add_scanlines(self, img, width_pc=1/150, darken=15):
        """Adds VHS-style scanlines."""
        try:
            logging.info(f"Adding scanlines with width_pc: {width_pc}, darken: {darken}")
            
            if img is None:
                logging.error("Input image is None")
                raise CustomException("Input image is None", sys)
            
            h, w = img.shape[:2]
            n_lines = int(h / (width_pc * 100))
            width_px = int(h * width_pc)
            
            if width_px <= 0:
                logging.warning("Width pixels is 0 or negative, setting to 1")
                width_px = 1
            
            for i in range(n_lines):
                start_y = i * width_px * 2
                end_y = start_y + width_px
                if end_y < h:
                    dark_line = np.maximum(img[start_y:end_y, :].astype("int16") - darken, 0).astype("uint8")
                    img[start_y:end_y, :] = dark_line
            
            logging.info(f"Scanlines added successfully. Total lines: {n_lines}")
            
        except Exception as e:
            logging.error(f"Error adding scanlines: {str(e)}")
            raise CustomException(e, sys)

    def add_vhs_color_distortion(self, img, spacing=5):
        """Adds VHS color distortion effect."""
        try:
            logging.info(f"Adding VHS color distortion with spacing: {spacing}")
            
            if img is None:
                logging.error("Input image is None")
                raise CustomException("Input image is None", sys)
            
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Create colored versions
            sat = np.full(gray.shape, 100, dtype="uint8")
            blue_hue = np.full(gray.shape, 90, dtype="uint8")
            red_hue = np.full(gray.shape, 0, dtype="uint8")
            
            blue = cv2.merge([blue_hue, sat, gray])
            red = cv2.merge([red_hue, sat, gray])
            
            blue = cv2.cvtColor(blue, cv2.COLOR_HSV2BGR)
            red = cv2.cvtColor(red, cv2.COLOR_HSV2BGR)
            
            # Shift colors
            blue = imutils.translate(blue, spacing, 0)
            red = imutils.translate(red, -spacing, 0)
            
            # Apply lighten blend mode
            np.copyto(img, blue, where=(blue > img))
            np.copyto(img, red, where=(red > img))
            
            # Enhance saturation
            result_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype("float32")
            h, s, v = cv2.split(result_hsv)
            s *= 1.5
            s = np.clip(s, 0, 255)
            v += 20
            v = np.clip(v, 0, 255)
            result_hsv = cv2.merge([h, s, v])
            
            result = cv2.cvtColor(result_hsv.astype("uint8"), cv2.COLOR_HSV2BGR)
            logging.info("VHS color distortion applied successfully")
            return result
            
        except Exception as e:
            logging.error(f"Error adding VHS color distortion: {str(e)}")
            raise CustomException(e, sys)

    def draw_vhs_text(self, img, top="PLAY", bottom=None):
        """Draws VHS-style text overlay."""
        try:
            logging.info(f"Drawing VHS text overlay. Top: {top}")
            
            if img is None:
                logging.error("Input image is None")
                raise CustomException("Input image is None", sys)
            
            if not self.vhs_font_loaded:
                logging.warning("VHS font not loaded, returning original image")
                return img
            
            h, w = img.shape[:2]
            
            # Create text using OpenCV instead of PIL for simplicity
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = h / 720
            thickness = 2
            
            # Top text
            cv2.putText(img, f"▶ {top}", (20, 40), font, font_scale, (255, 255, 255), thickness)
            
            # Bottom text
            if bottom is None:
                cur_time = time.localtime()
                bottom = time.strftime("%p %I:%M\n%b. %d %Y", cur_time).upper()
            
            lines = bottom.split('\n')
            for i, line in enumerate(lines):
                y = h - 60 + i * 30
                cv2.putText(img, line, (20, y), font, font_scale * 0.8, (255, 255, 255), thickness)
            
            logging.info("VHS text overlay added successfully")
            return img
            
        except Exception as e:
            logging.error(f"Error drawing VHS text: {str(e)}")
            raise CustomException(e, sys)

    def gaussian_noise(self, img, variance=8):
        """Adds gaussian noise to the image."""
        try:
            logging.info(f"Adding gaussian noise with variance: {variance}")
            
            if img is None:
                logging.error("Input image is None")
                raise CustomException("Input image is None", sys)
            
            row, col, ch = img.shape
            mean = 0
            sigma = variance ** 0.5
            gauss = np.random.normal(mean, sigma, (row, col, ch))
            gauss = gauss.reshape(row, col, ch)
            noisy = img + gauss
            cv2.normalize(noisy, noisy, 0, 255, cv2.NORM_MINMAX)
            
            result = noisy.astype("uint8")
            logging.info("Gaussian noise added successfully")
            return result
            
        except Exception as e:
            logging.error(f"Error adding gaussian noise: {str(e)}")
            raise CustomException(e, sys)

    def get_save_name(self):
        """Returns a timestamp-based filename."""
        try:
            timestamp = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
            logging.info(f"Generated save name: {timestamp}")
            return timestamp
            
        except Exception as e:
            logging.error(f"Error generating save name: {str(e)}")
            raise CustomException(e, sys)
    
    def initiate_glitch_effect(self, input_path):
        """Main function to process an image with glitch effects."""
        try:
            logging.info(f"Starting image processing for: {input_path}")
            
            # Validate input path
            if not input_path:
                logging.error("Input path is empty or None")
                raise CustomException("Input path is empty or None", sys)
            
            if not os.path.exists(input_path):
                logging.error(f"Input image not found: {input_path}")
                raise CustomException(f"Input image not found: {input_path}", sys)
            
            # Load image
            img = cv2.imread(input_path)
            if img is None:
                logging.error(f"Could not load image: {input_path}")
                raise CustomException(f"Could not load image: {input_path}", sys)
            
            logging.info(f"Image loaded successfully. Shape: {img.shape}")
            
            # Default effects configuration
            effects = {"all_effects": True}
            resize_width = effects.get("resize_width", 512)
            
            # Resize image
            img = imutils.resize(img, width=resize_width)
            logging.info(f"Image resized to width: {resize_width}")
            
            # Create copy for processing
            result_img = img.copy()
            
            # Determine which effects to apply
            apply_all = effects.get("all_effects", False)
            apply_blackout = apply_all or effects.get("blackout_bar", False)
            apply_scanlines = apply_all or effects.get("scanlines", False)
            apply_vhs_text = apply_all or effects.get("vhs_text", False)
            apply_color = apply_all or effects.get("color_distortion", False)
            apply_noise = apply_all or effects.get("noise", False)
            
            logging.info(f"Effects to apply - Blackout: {apply_blackout}, Scanlines: {apply_scanlines}, "
                        f"VHS Text: {apply_vhs_text}, Color: {apply_color}, Noise: {apply_noise}")
            
            # Face detection for blackout bar
            if apply_blackout:
                logging.info("Starting face detection for blackout effect")
                _, face_boxes = self.find_faces_dnn(result_img)
                
                if len(face_boxes) > 0:
                    logging.info(f"Found {len(face_boxes)} face(s)")
                    all_faces_pts = self.find_facial_landmarks(face_boxes, result_img)
                    
                    for i, face_box in enumerate(face_boxes):
                        if len(all_faces_pts) > i:
                            logging.info(f"Applying blackout bar to face {i+1}")
                            bar = self.find_blackout_bar(all_faces_pts[i], face_box)
                            cv2.drawContours(result_img, [bar], 0, (0, 0, 0), -1)
                else:
                    logging.info("No faces detected for blackout effect")
            
            # Apply non-face-based effects
            if apply_color:
                logging.info("Applying color distortion effect")
                result_img = self.add_vhs_color_distortion(result_img)
            
            if apply_scanlines:
                logging.info("Applying scanlines effect")
                self.add_scanlines(result_img)
            
            if apply_vhs_text:
                logging.info("Applying VHS text effect")
                result_img = self.draw_vhs_text(result_img)
            
            if apply_noise:
                logging.info("Applying noise effect")
                result_img = self.gaussian_noise(result_img)
            
            # Create output directory
            output_dir = os.path.join(
                self.glitch_effect_config.output_dir,
                self.glitch_effect_config.glitch_effect_dir
            )
            os.makedirs(output_dir, exist_ok=True)
            logging.info(f"Output directory created: {output_dir}")

            # Generate unique filename and save
            img_path = generate_unique_filename1(prefix="glitch_effect", ext=".jpg", directory=output_dir)
            
            success = cv2.imwrite(img_path, result_img)
            if not success:
                logging.error(f"Failed to save image to: {img_path}")
                raise CustomException(f"Failed to save image to: {img_path}", sys)
            
            logging.info(f"Image processing completed successfully. Output saved to: {img_path}")
            return GlitchEffectArtifact(glitch_effect_img_path=img_path)
            
        except Exception as e:
            logging.error(f"Error in image processing: {str(e)}")
            raise CustomException(e, sys)
        
class PencilSketch:
    def __init__(
        self,
        blur_simga: int = 5,
        ksize: typing.Tuple[int, int] = (0, 0),
        sharpen_value: int = None,
        kernel: np.ndarray = None,
        edge_enhancement: bool = True,
        edge_threshold: int = 50,
    ) -> None:
        try:
            self.pencil_sketch_config = PencilSketchConfig(config=ConfigEntity())

            self.blur_simga = blur_simga
            self.ksize = ksize
            self.sharpen_value = sharpen_value
            self.edge_enhancement = edge_enhancement
            self.edge_threshold = edge_threshold
            self.kernel = (
                np.array([[0, -1, 0], [-1, sharpen_value, -1], [0, -1, 0]])
                if kernel is None
                else kernel
            )

            logging.info(f"Initialized PencilSketch with sigma={blur_simga}, ksize={ksize}, sharpen_value={sharpen_value}")
        except Exception as e:
            raise CustomException(e, sys)

    def dodge(self, front: np.ndarray, back: np.ndarray) -> np.ndarray:
        try:
            logging.info("Applying dodge blend")

            front = front.astype(np.float32)
            back = back.astype(np.float32)

            # Prevent division by zero
            result = np.where(front == 255, 255, np.minimum(255, (back * 255.0) / (255.0 - front)))

            return result.astype("uint8")
        except Exception as e:
            raise CustomException(e, sys)

    def sharpen(self, image: np.ndarray) -> np.ndarray:
        try:
            logging.info("Starting sharpening")
            if self.sharpen_value is not None and isinstance(self.sharpen_value, int):
                inverted = 255 - image
                sharpened = 255 - cv2.filter2D(src=inverted, ddepth=-1, kernel=self.kernel)
                logging.info("Sharpening applied using custom kernel")
            else:
                sharpened = image.copy()
                logging.warning("Sharpening skipped due to missing sharpen_value")

            if self.edge_enhancement:
                logging.info("Applying edge enhancement")
                gray = (
                    cv2.cvtColor(sharpened, cv2.COLOR_BGR2GRAY)
                    if len(sharpened.shape) == 3
                    else sharpened
                )

                sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
                sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
                sobel_combined = np.sqrt(sobel_x**2 + sobel_y**2)

                laplacian = cv2.Laplacian(gray, cv2.CV_64F, ksize=3)
                laplacian = np.absolute(laplacian)

                edges = np.maximum(sobel_combined, laplacian)
                edges = np.clip(edges, 0, 255).astype(np.uint8)

                _, edge_mask = cv2.threshold(edges, self.edge_threshold, 255, cv2.THRESH_BINARY)

                kernel_dilate = np.ones((2, 2), np.uint8)
                edge_mask = cv2.dilate(edge_mask, kernel_dilate, iterations=1)

                if len(sharpened.shape) == 3:
                    edge_mask = cv2.cvtColor(edge_mask, cv2.COLOR_GRAY2BGR)

                edge_factor = 0.3
                sharpened = sharpened.astype(np.float32)
                edge_mask = edge_mask.astype(np.float32) / 255.0

                sharpened = sharpened * (1 - edge_factor * edge_mask)
                sharpened = np.clip(sharpened, 0, 255).astype(np.uint8)

                logging.info("Edge enhancement complete")

            return sharpened
        except Exception as e:
            raise CustomException(e, sys)

    def apply_pencil_sketch(self, frame: np.ndarray) -> np.ndarray:
        try:
            logging.info("Converting image to grayscale")
            grayscale = np.array(
                np.dot(frame[..., :3], [0.299, 0.587, 0.114]), dtype=np.uint8
            )
            grayscale = np.stack((grayscale,) * 3, axis=-1)

            logging.info("Creating inverted image")
            inverted_img = 255 - grayscale

            logging.info("Applying Gaussian Blur")
            blur_img = cv2.GaussianBlur(
                inverted_img, ksize=self.ksize, sigmaX=self.blur_simga
            )

            logging.info("Blending grayscale and blurred image using dodge function")
            final_img = self.dodge(blur_img, grayscale)

            logging.info("Applying final sharpening")
            sharpened_image = self.sharpen(final_img)

            return sharpened_image
        except Exception as e:
            raise CustomException(e, sys)

    def initiate_pencil_sketch(self, image_path: str) -> PencilSketchArtifact:
        try:
            logging.info(f"Reading image from path: {image_path}")
            image = cv2.imread(image_path)

            if image is None:
                raise CustomException("Image not found or could not be read", sys)

            logging.info("Creating multiple pencil sketch variations")

            pencilSketch1 = PencilSketch(
                blur_simga=10, sharpen_value=5, edge_enhancement=True, edge_threshold=30
            )
            pencilSketch2 = PencilSketch(
                blur_simga=8, sharpen_value=7, edge_enhancement=True, edge_threshold=20
            )
            pencilSketch3 = PencilSketch(
                blur_simga=10,
                kernel=self.pencil_sketch_config.custom_kernel,
                edge_enhancement=True,
                edge_threshold=40,
            )

            logging.info("Applying sketch variations")
            img1 = pencilSketch1.apply_pencil_sketch(image)
            img2 = pencilSketch2.apply_pencil_sketch(image)
            img3 = pencilSketch3.apply_pencil_sketch(image)

            output_dir = os.path.join(
                self.pencil_sketch_config.output_dir,
                self.pencil_sketch_config.pencil_sketch_dir
            )
            os.makedirs(output_dir, exist_ok=True)

            img1_path = generate_unique_filename1(prefix="blur_sigma_10", ext=".jpg", directory=output_dir)
            img2_path = generate_unique_filename1(prefix="blur_sigma_8", ext=".jpg", directory=output_dir)
            img3_path = generate_unique_filename1(prefix="custom_kernel", ext=".jpg", directory=output_dir)

            cv2.imwrite(img1_path, img1)
            cv2.imwrite(img2_path, img2)
            cv2.imwrite(img3_path, img3)

            logging.info("Sketch images written to disk")

            return PencilSketchArtifact(
                blur_sigma_10_img_path=img1_path,
                blur_sigma_8_img_path=img2_path,
                custom_kernel_img_path=img3_path,
            )
        except Exception as e:
            raise CustomException(e, sys)