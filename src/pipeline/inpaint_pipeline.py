from PIL import Image
import torch
import torchvision.transforms as T
import os
import sys
from src.exceptions import CustomException
from src.logger import logging
from src.entity.artifact import InpaintEntityArtifact
from src.entity.config import ConfigEntity


def inpaint_image(image_path, mask_path):
    try:
        config = ConfigEntity()

        checkpoint_path = config.check_point_path
        output_path = config.inpaint_output_dir
        logging.info(f"Starting inpainting process for image: {image_path}")

        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")
        if not os.path.exists(mask_path):
            raise FileNotFoundError(f"Mask not found: {mask_path}")
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        logging.info("Loading generator checkpoint")
        generator_state_dict = torch.load(checkpoint_path, map_location='cpu')['G']

        if 'stage1.conv1.conv.weight' in generator_state_dict.keys():
            from src.components.inpaint.networks import Generator
            logging.info("Using Generator from src.model.networks")
        else:
            from src.components.inpaint.networks_tf import Generator
            logging.info("Using Generator from src.model.networks_tf")

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logging.info(f"Using device: {device}")

        generator = Generator(cnum_in=5, cnum=48, return_flow=False).to(device)
        generator.load_state_dict(generator_state_dict, strict=True)
        generator.eval()

        logging.info("Loading and preprocessing input image and mask")
        image = Image.open(image_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")
        image = T.ToTensor()(image)
        mask = T.ToTensor()(mask)

        _, h, w = image.shape
        grid = 8
        h, w = h // grid * grid, w // grid * grid
        image = image[:, :h, :w].unsqueeze(0)
        mask = mask[0:1, :h, :w].unsqueeze(0)
        image = (image * 2 - 1.).to(device)
        mask = (mask > 0.5).float().to(device)

        logging.info("Creating masked input")
        image_masked = image * (1 - mask)
        ones_x = torch.ones_like(image_masked)[:, 0:1, :, :]
        x = torch.cat([image_masked, ones_x, ones_x * mask], dim=1)

        logging.info("Running inference")
        with torch.inference_mode():
            _, x_stage2 = generator(x, mask)

        logging.info("Combining inpainted and unmasked regions")
        image_inpainted = image * (1 - mask) + x_stage2 * mask
        img_out = ((image_inpainted[0].permute(1, 2, 0) + 1) * 127.5).clamp(0, 255)
        img_out = img_out.to('cpu', dtype=torch.uint8)
        result_image = Image.fromarray(img_out.numpy())
        result_image.save(output_path)
        logging.info(f"Inpainting complete. Output saved at: {output_path}")

        return InpaintEntityArtifact(out_put_path=output_path)

        # logging.info(f"Inpainting complete. Output saved at: {output_path}")

    except FileNotFoundError as fnf_error:
        logging.error(f"FileNotFoundError: {fnf_error}")
        raise CustomException(fnf_error, sys)

    except Exception as e:
        logging.error(f"Exception occurred during inpainting: {e}", exc_info=True)
        raise CustomException(e, sys)
