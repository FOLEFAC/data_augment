"""Stable Diffusion Inpainting augmentation plugin.
"""

from bson import json_util
import inspect
import os

import numpy as np### math computations
import matplotlib.pyplot as plt### plotting bar chart
import pandas as pd
import cv2
from PIL import Image
import torch
import numpy as np
from diffusers.utils import load_image, make_image_grid
import torch
from torchvision.transforms.functional import to_pil_image
import numpy as np
import random
import hashlib
from diffusers import StableDiffusionInpaintPipeline, EulerDiscreteScheduler

import fiftyone as fo
import fiftyone.operators as foo
from fiftyone.operators import types
from fiftyone.core.utils import add_sys_path

MODEL_ID = "stabilityai/stable-diffusion-2-inpainting"

def _get_target_view(ctx, target):
    if target == "SELECTED_SAMPLES":
        return ctx.view.select(ctx.selected)

    if target == "DATASET":
        return ctx.dataset

    return ctx.view

def create_pipeline(model_id):
  scheduler = EulerDiscreteScheduler.from_pretrained(model_id, subfolder="scheduler")
  pipe = StableDiffusionInpaintPipeline.from_pretrained(model_id,
                                                        scheduler=scheduler,
                                                        revision="fp16",
                                                        torch_dtype=torch.float16)

  pipe = pipe.to("cuda")
  pipe.enable_xformers_memory_efficient_attention()
  return pipe

def generate_inputs(im_path,mask_path, mask_id):
  print("the mask id is ===", mask_id)
  #select_class = id2label[mask_id]

  source_image = cv2.imread(im_path)
  print("sourceimg", source_image.shape)
  sd_mask = cv2.imread(mask_path,cv2.IMREAD_GRAYSCALE)

  out = (sd_mask+(-mask_id*np.ones_like(sd_mask)))
  mask=-(np.clip(1e10*np.multiply(out,out),a_min=0,a_max=255)-255)
  pil_image = Image.fromarray(source_image).resize((512,512))
  pil_mask = Image.fromarray(mask).resize((512,512))
  return pil_image, pil_mask

def augpaint(pipe, prompt, pil_image, pil_mask, num_images_per_prompt, guidance_scale, num_inference_steps):

  # Generate a random seed
  generator = torch.Generator(device="cuda").manual_seed(10)

  # Run stable diffusion pipeline in inpainting mode and store the generated images in a list
  encoded_images = []
  for i in range(num_images_per_prompt):
      image = pipe(prompt=prompt, guidance_scale=guidance_scale,
                    num_inference_steps=num_inference_steps, generator=generator,
                    image=pil_image, mask_image=pil_mask).images[0]
      encoded_images.append(image.resize((550,825)))
  return encoded_images[0]

def create_hash():
  randint = random.randint(0, 100000000)
  hash = hashlib.sha256(str(randint).encode("utf-8")).hexdigest()[:10]
  return hash

def list_classes(mask_path):

  mask = cv2.imread(mask_path,cv2.IMREAD_GRAYSCALE)
  list_unique = np.unique(mask)
  return list_unique

def transform_sample(sample, select_class, prompt, num_images_per_prompt, guidance_scale, num_inference_steps):
    hash = create_hash()
    filename = sample.filepath.split("/")[-1][:-4]+"_"+str(hash)+".png"
    pipe = create_pipeline(model_id)
    im,mask = generate_inputs(
        sample.filepath, sample.ground_truth.mask_path,
        label2id[select_class])

    out = augpaint(pipe, prompt, im, mask,num_images_per_prompt, guidance_scale, num_inference_steps)

    shutil.copy(sample.ground_truth.mask_path,
                sample.ground_truth.mask_path[:-4]+"_"+str(hash)+".png",
                )

    new_sample = fo.Sample(
        filepath=sample.filepath[:-4]+"_"+str(hash)+".png",
        ground_truth=fo.Segmentation(
            mask_path=sample.ground_truth.mask_path[:-4]+"_"+str(hash)+".png"),
    )

    return new_sample

def labels_from_id(labels_filepath):
    df = pd.read_csv(labels_filepath)
    id2label = {}
    for i,j in df.iterrows():
        if i==0:
            id2label[i] = 'nan'
        else:
            id2label[i] = j['label_list']
    return id2label


class SDAugment(foo.Operator):
    @property
    def config(self):
        _config = foo.OperatorConfig(
            name="augment_with_sd_inpainting",
            label="Augment with Stable Diffusion Inpainting",
            description="Apply Augmentation with Stable Diffusion Inpainting Model to an image based on a mask found in image.",
            dynamic=True,
        )
        #_config.icon = "/assets/icon.svg"
        return _config

    def resolve_input(self, ctx):
        inputs = types.Object()
        form_view = types.View(
            label="Augment with Stable Diffusion Inpainting",
            description="Apply an Stable Diffusion Inpainting to the (image,mask) pair the sample.",
        )

        inputs.int(
            "num_augs",
            label="Number of augmentations per sample",
            description="The number of random augmentations to apply to each sample",
            default=1,
            view=types.FieldView(),
        )

        target_view = ctx.view.select(ctx.selected)
        labels_filepath = "/assets/labels.csv"

        id2label = labels_from_id(labels_filepath)
        
        
        for sample in target_view:
            mask = cv2.imread(sample.ground_truth.mask_path,cv2.IMREAD_GRAYSCALE)
            unique_choices = tuple([id2label[i] for i in np.unique(mask)[1:]])
            
            break

        class_choices = types.Dropdown(label="Class")
        for clas in unique_choices:
            class_choices.add_choice(clas, label=clas)


        inputs.enum(
            "class_choices",
            class_choices.values(),
            default="skin",
            view=class_choices,
        )

        inputs.str(
            "prompt",
            label="Prompt",
            description="The prompt to generate new data from",
            required=True,
        )


        inference_steps_slider = types.SliderView(
                label="Num Inference Steps",
                componentsProps={"slider": {"min": 50, "max": 200, "step": 10}},
            )
        inputs.int(
            "inference_steps",
             default=50,
              view=inference_steps_slider
        )


        guidance_scale_slider = types.SliderView(
                label="Guidance_scale",
                componentsProps={"slider": {"min": 1, "max": 30, "step": 2}},
            )
        inputs.int(
            "guidance_scale",
             default=7,
              view=guidance_scale_slider
        )


        return types.Property(inputs, view=form_view)

    def resolve_delegation(self, ctx):
        return ctx.params.get("delegate", False)

    def execute(self, ctx):
        num_images_per_prompt = ctx.params.get("num_augs", 1)
        select_class = ctx.params.get("class_choices", "None provided")
        prompt = ctx.params.get("prompt", "None provided")
        guidance_scale = int(ctx.params.get("guidance_scale", "None provided"))
        num_inference_steps = int(ctx.params.get("num_inference_steps", "None provided"))

        label_fields = _get_label_fields_to_transform(ctx)

        target_view = ctx.view.select(ctx.selected)

        for sample in target_view:
            new_sample = transform_sample(sample, select_class, prompt,
                 num_images_per_prompt, guidance_scale, num_inference_steps)
            dataset.add_sample(new_sample)
            break
            

        ctx.trigger("reload_dataset")

def register(plugin):
    plugin.register(SDAugment)