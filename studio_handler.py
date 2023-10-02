# Copyright (c) Meta Platforms, Inc. and affiliates.

# NEED TO BE REFACTOR

import datetime
import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
import streamlit as st

st.set_page_config(page_title="Difussion Genetic UI", layout="wide")
verbose = False
import base64
import pickle

import joblib

import matplotlib.pyplot as plt
import matplotlib.ticker as plticker
import numpy as np
import pandas as pd
import torch
from diffusers import StableDiffusionPipeline
from gfpgan.utils import GFPGANer
from PIL import Image
from RealESRGAN import RealESRGAN
from streamlit_drawable_canvas import st_canvas

try:
    from PIL import Image
except ImportError:
    import Image
import ast

import json, requests
import webbrowser
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import nevergrad as ng
import numpy as np
import rasterio.features

from diffusers import StableDiffusionInpaintPipeline
from google.colab import auth
from oauth2client.client import GoogleCredentials
from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
from shapely.geometry import Polygon
from sklearn import tree
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier

model_use_id = 1

MIN_TOKEN_LENGTH = 16
HEIGHT_RESOLUTION = 512
WIDTH_RESOLUTION = 512
SIGMA_REDUCTION_PER_CHOICE = 0.7
MINIMUM_BAD_NUMBER_FOR_MLP = 10
POINT_COLUMNS = ["top", "left", "image", "x", "y"]
DEVICE = "cuda"
loading_image = "https://newsandstory.com/tempImage/15121520094520186903.jpg"
# 0-sd 1 -sdmj
from typing import Callable, List, Optional, Union

from diffusers.pipelines.stable_diffusion import StableDiffusionPipelineOutput


def slerp(t, v0, v1, DOT_THRESHOLD=0.9995):
    """helper function to spherically interpolate two arrays v1 v2"""

    if not isinstance(v0, np.ndarray):
        inputs_are_torch = True
        input_device = v0.device
        v0 = v0.cpu().numpy()
        v1 = v1.cpu().numpy()

    dot = np.sum(v0 * v1 / (np.linalg.norm(v0) * np.linalg.norm(v1)))
    if np.abs(dot) > DOT_THRESHOLD:
        v2 = (1 - t) * v0 + t * v1
    else:
        theta_0 = np.arccos(dot)
        sin_theta_0 = np.sin(theta_0)
        theta_t = theta_0 * t
        sin_theta_t = np.sin(theta_t)
        s0 = np.sin(theta_0 - theta_t) / sin_theta_0
        s1 = sin_theta_t / sin_theta_0
        v2 = s0 * v0 + s1 * v1

    if inputs_are_torch:
        v2 = torch.from_numpy(v2).to(input_device)

    return v2


@torch.no_grad()
def callSD(
    pipe,
    text_embeddings: Union[torch.Tensor, List],
    do_classifier_free_guidance,
    height: Optional[int] = None,
    width: Optional[int] = None,
    num_inference_steps: int = 50,
    guidance_scale: float = 7.5,
    negative_prompt: Optional[Union[str, List[str]]] = None,
    num_images_per_prompt: Optional[int] = 1,
    eta: float = 0.0,
    generator: Optional[torch.Generator] = None,
    latents: Optional[torch.FloatTensor] = None,
    output_type: Optional[str] = "pil",
    return_dict: bool = True,
    callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
    callback_steps: Optional[int] = 1,
):

    # 0. Default height and width to unet
    height = height or pipe.unet.config.sample_size * pipe.vae_scale_factor
    width = width or pipe.unet.config.sample_size * pipe.vae_scale_factor

    # 1. Check inputs. Raise error if not correct
    pipe.check_inputs(prompt, height, width, callback_steps)

    # 2. Define call parameters
    batch_size = 1 if isinstance(prompt, str) else len(prompt)
    device = pipe._execution_device

    # 3. Encode input promp
    # skip this stpe
    # 4. Prepare timesteps
    pipe.scheduler.set_timesteps(num_inference_steps, device=device)
    timesteps = pipe.scheduler.timesteps

    # 5. Prepare latent variables
    num_channels_latents = pipe.unet.in_channels

    # 6. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
    extra_step_kwargs = pipe.prepare_extra_step_kwargs(generator, eta)

    # 7. Denoising loop
    if "pndm_order" in pipe.scheduler.__dict__.keys():
        order_c = pipe.scheduler.pndm_order
    elif "order" in pipe.scheduler.__dict__.keys():
        order_c = pipe.scheduler.order
    else:
        order_c = 1

    num_warmup_steps = len(timesteps) - num_inference_steps * order_c

    with pipe.progress_bar(range(num_inference_steps)) as progress_bar:
        for i, t in enumerate(timesteps):
            # expand the latents if we are doing classifier free guidance
            latent_model_input = (
                torch.cat([latents] * 2) if do_classifier_free_guidance else latents
            )
            latent_model_input = pipe.scheduler.scale_model_input(latent_model_input, t)

            # predict the noise residual

            noise_pred = pipe.unet(
                latent_model_input, t, encoder_hidden_states=text_embeddings
            ).sample

            # perform guidance
            if do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (
                    noise_pred_text - noise_pred_uncond
                )

                # compute the previous noisy sample x_t -> x_t-1
                latents = pipe.scheduler.step(
                    noise_pred, t, latents, **extra_step_kwargs
                ).prev_sample

            # call the callback, if provided
            if i == len(timesteps) - 1 or (
                (i + 1) > num_warmup_steps and (i + 1) % order_c == 0
            ):
                progress_bar.update()
                if callback is not None and i % callback_steps == 0:
                    callback(i, t, latents)

    # 8. Post-processing
    image = pipe.decode_latents(latents)

    # 9. Run safety checker
    image, has_nsfw_concept = pipe.run_safety_checker(
        image, device, text_embeddings.dtype
    )

    # 10. Convert to PIL
    if output_type == "pil":
        image = pipe.numpy_to_pil(image)

    if not return_dict:
        return (image, has_nsfw_concept)

    return StableDiffusionPipelineOutput(
        images=image, nsfw_content_detected=has_nsfw_concept
    )


def save_high_res(imfile, output):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = RealESRGAN(device, scale=4)
    model.load_weights("weights/RealESRGAN_x4.pth", download=True)
    image = Image.open(imfile).convert("RGB")
    sr_image = model.predict(image)
    sr_image.save(output)
    return sr_image


def save_high_res_face(path, output_path):
    fe = GFPGANer(
        model_path="GFPGAN/GFPGANv1.3.pth",
        upscale=4,
        arch="clean",
        channel_multiplier=2,
    )
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    _, _, output = fe.enhance(
        img, has_aligned=False, only_center_face=False, paste_back=True
    )
    cv2.imwrite(output_path, output)

    return output


def short(x, resolution):
    x = np.array(x).reshape((4, resolution, resolution))

    y = np.zeros((4, 16, 16))
    for u in range(4):
        for v in range(resolution):
            for w in range(resolution):
                y[u][v // int(resolution / 16)][w // int(resolution / 16)] += x[u][v][w]
    return y.flatten()


def generate_state(
    prompt: str = "drawing low-resolution: Greenpeace activists on battlefiled in heroes of might and magic 3",
    llambda: int = 4,
    sigma: float = 1.0,
):
    state = {}
    state["prompt"] = prompt
    state["llambda"] = llambda
    state["sigma"] = sigma
    state["no_ml"] = False
    state["imagev"] = []
    state["images_latents"] = []
    state["total_choosen"] = set()
    state["all_points"] = pd.DataFrame(columns=POINT_COLUMNS)
    state["images_filenames"] = []
    state["used_indexes"] = []
    state["iterations"] = []
    state["same_z"] = False
    state["movie_order"] = []
    state["gif_names"] = []
    state["prompts"] = []
    state["resolution"] = 64
    state["image_dimensions"] = 512
    state["polygons"] = []
    state["load_inpaint"] = False
    state["verbose"] = False
    state["change_selected"] = "save"
    state["open_HF_SD_14"] = False
    state["drive_images"] = dict()
    state["drive"] = None
    state["model_use_id"] = 0
    return state


if "state" not in st.session_state:
    state = generate_state()
else:
    state = st.session_state["state"]

verbose = state["verbose"]
used_indexes_flatten = [False] * len(state["images_filenames"])
points = []
is_drive = True
try:
    gauth = joblib.load("gauth.j")
    drive = GoogleDrive(gauth)
    state["drive"] = drive
except:
    is_drive = False

# def save_and_create_image(filepath: str, state,drive=None):

fb_logo = "https://w7.pngwing.com/pngs/550/868/png-transparent-facebook-facebook-share-facebook-share-button-share-facebook-facebook-button-facebook-icon-socialmedia-marketing-fb-social-media-flat-icon.png"
links_dict = {}
links_dict_file = "links_dict.j"
if Path(links_dict_file).is_file():
    links_dict = joblib.load(links_dict_file)


def show_picture(state, image_index, points, used_indexes_flatten, preview=False):
    st.caption(f"image from generation {state['iterations'][image_index]} ")
    image_filename = state["images_filenames"][image_index]
    # image = Image.open(image_filename)
    image = state["imagev"][image_index]
    # image.thumbnail((WIDTH_RESOLUTION, HEIGHT_RESOLUTION))
    width, height = image.size
    width, height = (WIDTH_RESOLUTION, HEIGHT_RESOLUTION)
    if not preview:
        used_indexes_flatten[image_index] = st.checkbox(
            "Inspiration from this image",
            image_index in state["used_indexes"],
            key=f"checkbox_{image_index}",
        )
    else:
        used_indexes_flatten[image_index] = False
    if image_filename in links_dict.keys():
        st.markdown(
            f'<a href="{links_dict[image_filename]}"> <img src="{fb_logo}" width="100" height="30"> </a>',
            unsafe_allow_html=True,
        )

    img_hr_file = f"image_{image_index}_hr.png"
    if not Path(img_hr_file).is_file() and not preview and False:
        btn_hr = st.button(
            label="create high resolution", key=f"hr_button_{image_index}"
        )
        if btn_hr:
            save_high_res(image_filename, img_hr_file)
    elif not preview and False:
        st.caption("High resloution saved")
    img_face_file = f"image_{image_index}_face_hr.png"
    if not Path(img_face_file).is_file() and not preview:
        btn_face = st.button(
            label="fix faces and save in high resolution",
            key=f"face_button_{image_index}",
        )
        if btn_face:
            save_high_res_face(image_filename, img_face_file)
    elif not preview:
        st.caption("High resloution with fixed faces available on High resolution tab")

    if used_indexes_flatten[image_index] and not preview:
        container = st.container()
        canvas_result = st_canvas(
            fill_color="rgba(255, 165, 0, 0.3)",  # Fixed fill color with some opacity
            stroke_width=3,
            background_image=image,
            update_streamlit=True,
            width=width,
            height=height,
            drawing_mode=drawing_mode,
            point_display_radius=2 * state["correction_multiplier"]
            if drawing_mode == "point"
            else 0,
            key=f"canvas_{image_index}",
        )
        objects = None
        if canvas_result.json_data is not None:
            objects = pd.json_normalize(
                canvas_result.json_data["objects"]
            )  # need to convert obj to str because PyArrow
            for col in objects.select_dtypes(include=["object"]).columns:
                objects[col] = objects[col].astype("str")
            if objects is not None and len(objects) > 0:
                objects["y"] = objects["top"] / height
                objects["x"] = objects["left"] / width
                objects["image"] = image_index
                points.append(objects)

        with container:
            if objects is not None and len(objects) > 0:
                st.caption(
                    f'Parts of image near points will be {state["change_selected"]}d. (Select one or several parts of the image for choosing what to preserve or what to change in the image)'
                )
            else:
                if len(state["all_points"]) > 0:
                    original_title = '<p style="color:Red;"> You need to pickup point in at least two images or don\'t pickup any point If at least one point was chosen, we ignore images without points</p>'
                    st.markdown(original_title, unsafe_allow_html=True)
                else:
                    st.caption(
                        "No points selected: variations of the entire image (selects points on the image for keeping or modifying specific parts)."
                    )
    else:
        st.image(image)
    st.markdown("""---""")


def generate_movie(
    state: dict,
    pipe=None,
    verbose=False,
    generating_bar=None,
    guidance_scale=7.5,
    temp_container=None,
):
    result_pictures = []
    if len(state["movie_order"]) < 2:
        st.error("Not enough images selected")
        st.stop()
    first_picture_index = state["movie_order"][0]
    do_classifier_free_guidance = guidance_scale > 1.0
    total_images = state["llambda"] * (len(state["movie_order"]))

    for index, pict_index in enumerate(state["movie_order"][1:]):
        result_pictures.append(state["imagev"][first_picture_index])
        prompt_0 = state["prompts"][first_picture_index]
        latents_0 = state["images_latents"][first_picture_index]

        prompt_1 = state["prompts"][pict_index]
        latents_1 = state["images_latents"][pict_index]

        prompt_enc_0 = pipe._encode_prompt(
            prompt_0,
            DEVICE,
            num_images_per_prompt=1,
            do_classifier_free_guidance=do_classifier_free_guidance,
            negative_prompt=None,
        )
        prompt_enc_1 = pipe._encode_prompt(
            prompt_1,
            DEVICE,
            num_images_per_prompt=1,
            do_classifier_free_guidance=do_classifier_free_guidance,
            negative_prompt=None,
        )
        interpolate_prompts = np.linspace(
            prompt_enc_0.detach().cpu().numpy(),
            prompt_enc_1.detach().cpu().numpy(),
            num=state["llambda"] + 1,
            endpoint=False,
        )
        # interpolate_latents = np.linspace(latents_0.detach().cpu().numpy(), latents_1.detach().cpu().numpy(), num=state["llambda"]+1,endpoint=False)
        st_index = 1
        if verbose:
            st.caption(
                f'1 {type(state["imagev"][first_picture_index])} {state["imagev"][first_picture_index].size}'
            )

        for gen_index, prompt_gen in enumerate(interpolate_prompts[st_index:]):
            latents_gen = slerp(
                (gen_index + st_index) / (state["llambda"] + 1), latents_0, latents_1
            )
            res_img = callSD(
                pipe,
                torch.from_numpy(prompt_gen).to(DEVICE),
                latents=latents_gen.to(DEVICE),
                do_classifier_free_guidance=do_classifier_free_guidance,
            ).images[0]
            if temp_container is not None:
                with temp_container:
                    st.caption(f"image #{gen_index}")
                    st.image(res_img)
                    #st.write(st.dg)
            st.caption(f"{type(res_img)} {res_img.size}")
            result_pictures.append(res_img)
            if generating_bar is not None:
                generating_bar.progress(0.8 / total_images)
        first_picture_index = pict_index
    result_pictures.append(state["imagev"][state["movie_order"][-1]])
    st.caption(
        f'1 {type(state["imagev"][state["movie_order"][-1]])} {state["imagev"][state["movie_order"][-1]].size}'
    )

    img_gif = result_pictures[0]
    gif_name = f'movie_{len(state["gif_names"])}.gif'
    img_gif.save(
        fp=gif_name,
        format="GIF",
        append_images=result_pictures[1:],
        save_all=True,
        duration=state["duration_between_slides"] * len(result_pictures),
        loop=0,
    )

    state["gif_names"].append(gif_name)
    return state


def generate_inpaint(
    state: dict,
    pipe=None,
    verbose=False,
    generating_bar=None,
    cols: list = None,
    containers: list = [],
    points=None,
    used_indexes_flatten=None,
    chosen=0,
    preserved_images=[],
    preserved_latent=[[0]],
):

    st.dataframe(state["polygons"])
    choosen_polys = state["polygons"][state["polygons"]["image"] == chosen[0]]
    polys_to_show = []
    for _, poly_row in choosen_polys.iterrows():
        poly = poly_row["path"]
        poly = ast.literal_eval(poly)
        new_poly = [(l[1], l[2]) for l in poly if len(l) > 2]
        new_poly.append(new_poly[0])
        new_poly = Polygon(new_poly)
        polys_to_show.append(new_poly)

    img_mask = rasterio.features.rasterize(
        polys_to_show, out_shape=(state["image_dimensions"], state["image_dimensions"])
    )
    img_mask *= 255
    if state["change_selected"] != "change":
        img_mask -= 255
        img_mask *= -1

    if len(state["iterations"]) > 0:
        gen_iteration = state["iterations"][-1] + 1
    else:
        gen_iteration = 0
    latents = preserved_latent[0]
    images = state["pipe_inpaint"](
        prompt=state["prompt"],
        image=preserved_images[0],
        mask_image=img_mask,
        num_images_per_prompt=state["llambda"],
    ).images
    for image in images:
        # image_index = total_count - i - 1
        image_index = len(state["imagev"])
        image_name = "image_{}.png".format(image_index)
        if verbose:
            st.text(f"saving {image_name}")
        image.save(image_name)
        hr_face = f"image_{image_index}_face_hr.png"
        if os.path.exists(hr_face):
            os.remove(hr_face)
        hr_res = f"image_{image_index}_hr.png"
        if os.path.exists(hr_res):
            os.remove(hr_res)
        # state["imagev"][image_index] = image
        # state["iterations"][image_index] = gen_iteration
        # state["images_filenames"][image_index] = image_name
        state["imagev"].append(image)
        state["images_filenames"].append(image_name)
        state["iterations"].append(gen_iteration)
        state["prompts"].append(state["prompt"])
        used_indexes_flatten.append(False)
        # Here we show picture

        # state["images_latents"][image_index] = latents
        state["images_latents"].append(latents)
        if verbose:
            st.text(f'image latents last { state["images_latents"][-1].shape}')
    state["used_indexes"] = []
    state["all_points"] = []
    with open(".state", "wb") as f:
        joblib.dump(state, f)
    return state


def generate_pictures(
    state: dict,
    pipe=None,
    verbose=False,
    generating_bar=None,
    cols: list = None,
    containers: list = [],
    points=None,
    used_indexes_flatten=None,
):

    latent_base = None  # if not None, base for random mutations
    latent_basev = None  # if not None, list of latent vectors
    chosen = []
    preserved_latent = []
    preserved_images = []
    llambda = state["llambda"]
    state["total_choosen"].update(state["used_indexes"])
    for choice in state["used_indexes"]:
        if choice not in chosen:
            preserved_latent += [state["images_latents"][choice]]  # [latentv[choice]]
            preserved_images += [state["imagev"][choice]]
        chosen += [choice]
    st.text(f"We keep {len(preserved_latent)} inspirational images, from {len(chosen)} clicks.")
    good = [
        short(
            state["images_latents"][i].cpu().detach().numpy().flatten(),
            state["resolution"],
        )
        for i in range(len(state["images_latents"]))
        if i in state["total_choosen"]
    ]
    bad = [
        short(
            state["images_latents"][i].cpu().detach().numpy().flatten(),
            state["resolution"],
        )
        for i in range(len(state["images_latents"]))
        if i not in state["total_choosen"]
    ]
    st.text(f"ready to generate, at time {datetime.datetime.now()}")

    if len(preserved_latent) == 1:
        st.text("We work from a single selected image")
        if len(state["polygons"]) > 0:
            if verbose:
                st.text("INPAINT")
            return generate_inpaint(
                state,
                pipe,
                verbose,
                generating_bar,
                cols,
                containers,
                points,
                used_indexes_flatten,
                chosen,
                preserved_images,
                preserved_latent,
            )
        if verbose:
            st.text("no inpaint")
            st.text(f'len state all_points {len(state["all_points"])}')

        if len(state["all_points"]) > 0:
            st.text(f"The user selected {len(state['all_points'])} points, all in a same image.")
            state["no_ml"] = True
            image_to_pick = state["imagev"][chosen[0]]
            clicks_df = state["all_points"][state["all_points"]["image"] == chosen[0]]
            clicks = [
                np.array((point["x"], point["y"])) for _, point in clicks_df.iterrows()
            ]
            if verbose:
                st.text(f"{len(clicks)} clicks, namely {clicks}")
            if (
                len(clicks) > 0
            ):  # Here we create llambda images, corresponding to the chosen image + local modifications at the clicks.
                the_base = state["images_latents"][chosen[0]]
                latent_basev = []
                for idx in reversed(range(llambda)):
                    correction_multiplier = state["correction_multiplier"]
                    radius = correction_multiplier * (idx + 1) / (16.0 * (llambda))
                    base = the_base.clone()
                    randomized = 0
                    for h in range(state["resolution"]):
                        xh = (h + 0.5) / float(state["resolution"])
                        for v in range(state["resolution"]):
                            xv = (v + 0.5) / float(state["resolution"])
                            distances = [
                                np.max(np.abs(np.array([xh, xv]) - c)) for c in clicks
                            ]
                            # if verbose:
                            #  print_text = f"idx {idx} h {h} xh {xh} v {v} xv {xv} dist min {min(distances)} rad {radius}"
                            if (
                                min(distances) > radius
                                and state["change_selected"] == "save"
                            ):
                                base[0, :, h, v] = torch.randn(4)
                                randomized += 1
                            if min(distances) < radius and change_selected == "change":
                                base[0, :, h, v] = torch.randn(4)
                                randomized += 1
                    if verbose:
                        st.text(f"idx {idx} randomized {randomized}")

                    latent_basev.append(base)
            else:
                if verbose:
                    st.text("The user did not select any point in the images, global mutations of a single image.")
                latent_base = state["images_latents"][chosen[0]]
                state["sigma"] *= SIGMA_REDUCTION_PER_CHOICE
        else:
            if verbose:
                st.text("The user did not select any point in the images, global mutations of a single image.")
            latent_base = state["images_latents"][chosen[0]]
            #state["sigma"] *= SIGMA_REDUCTION_PER_CHOICE
    elif len(preserved_latent) > 1:
        assert len(chosen) >= len(preserved_latent)
        num_points = [
            len(state["all_points"][state["all_points"]["image"] == c]) for c in chosen
        ]
        voronoi = len(state["all_points"]) > 1 and min(num_points) > 0
        # We can do Voronoi only if we have points on each selected image.
        if voronoi:
            st.text(
                f"We apply a Voronoi crossover between {len(num_points)} images with {num_points} points per image."
            )
            state["no_ml"] = True

            latent_basev = []
            for idx in range(llambda):
                if verbose:
                    st.text(f"We generate image {idx}")
                latent_base = state["images_latents"][chosen[0]].clone()
                b = len(preserved_images)
                # correction_multiplier = 1.1
                correction_multiplier = state["correction_multiplier"] / 4
                # ratio = 1.0 + correction_multiplier * 0.25 * (
                #    (idx) / (1e-5 + llambda - b - 1.0)
                # )
                ratio = 1.0 + (idx / llambda)

                # ratio = 1.0 + 0.25 * ((idx - b) / (1e-5 + llambda - b - 1.0))
                choices = []
                choosen_dict = {"random": 0}
                randomized = 0
                min_radius = 0.3
                assert state["resolution"] > 0
                for u in range(state["resolution"]):
                    xu = (u + 0.5) / float(state["resolution"])
                    assert xu >= 0
                    assert xu < 1
                    for v in range(state["resolution"]):
                        xv = (v + 0.5) / float(state["resolution"])
                        assert xv >= 0
                        assert xv < 1
                        dist = []
                        assert chosen > 0
                        for i, c in enumerate(chosen):
                            dists_point = [
                                np.linalg.norm(
                                    np.array((xu, xv))
                                    - np.array([point["x"], point["y"]])
                                )
                                for _, point in state["all_points"][
                                    state["all_points"]["image"] == c
                                ].iterrows()
                            ]
                            dist += [min(dists_point)]

                        sorted_dist = np.sort(dist)
                        if len(sorted_dist) < 2:
                            e = RuntimeError("Not enoght points selected")
                            st.exception(e)
                            st.stop()
                        #if state["change_selected"] == "save":
                        choice = chosen[int(np.argmin(dist))]
                        #elif state["change_selected"] == "change":
                        #    choice = chosen[int(np.argmax(dist))]

                        choosen_dict[str(choice)] = choosen_dict.get(str(choice), 0) + 1
                        choices += [choice]
                        latent_base[0, :, u, v] = state["images_latents"][choice][
                            0, :, u, v
                        ]
                        if (
                            sorted_dist[0]
                            > sorted_dist[1] / ratio
                            or sorted_dist[0] < 2. / float(state["resolution"])
                            # and sorted_dist[0] > min_radius  # This implies that with several points there is no randomness anymore; 0.3 is big.
                        ):
                            randomized += 1  # Let us count the number of randomized points.
                            choosen_dict["random"] += 1
                            latent_base[0, :, u, v] = torch.randn((4))
                if verbose:
                    st.text(f"idx {idx} randomized {randomized} chosen {choosen_dict}")
                assert randomized > 0 or llambda == 1, f"Not a single point is randomized ? Ratio={ratio}, sorted_dist={sorted_dist}"
                latent_basev += [latent_base]

        else:
            if verbose:
                st.text(f"Here we consider the global average of images! {chosen}")
            latent_base = torch.mean(
                torch.stack([state["images_latents"][i].to("cuda") for i in chosen]),
                0,
                False,
            ).half()

    latentv = []
    if verbose:
        st.text(f"llambda={llambda}")

    if latent_basev is not None:  # We have already created a batch of latent.
        if verbose:
            st.text(f"set new latentv and latent_basev from scratch")
        latentv = latent_basev
        latent_basev = None
    else:
        # We have a few images from previous iterations in preserved_latent
        # and preserved_images, and we create the rest.
        latentv = []

        if not state["same_z"]:
            st.text(f"All z are perturbated")
            for i in range(llambda):

                if latent_base is None:
                    latents = torch.randn(
                        (1, 4, state["resolution"], state["resolution"])
                    ).half()
                else:
                    if verbose:
                        st.text(f"setted new latent from latent_base ")
                    latents = latent_base.to("cuda") + state["sigma"] * (
                        (i + 1) / (llambda + 1)
                    ) * torch.randn(
                        (1, 4, state["resolution"], state["resolution"])
                    ).half().to(
                        "cuda"
                    )

                    l = latents.cpu().numpy().flatten()
                    coef = np.sqrt(len(l) / np.sum(l**2))
                    latents = coef * latents
                latentv += [latents]
        else:
            if latent_base is None:
                st.error("Can't build similar to this type of pictures choice")
            if verbose:
                st.text("Let us add latent_base")
            latentv += [latent_base.to("cuda")]
    if len(bad) < MINIMUM_BAD_NUMBER_FOR_MLP or len(good) == 0:
        state["no_ml"] = True
    if not state["no_ml"]:
        state["no_ml"] = no_ml_checkbox
    if not state["no_ml"]:
        clf = MLPClassifier(
            solver="lbfgs", alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1
        )
        tclf = clf.fit(good + bad, [1] * len(good) + [0] * len(bad))
    if generating_bar is not None:
        generating_bar.progress(0.8 / (len(latentv) + 1))
    total_count = len(state["imagev"]) + llambda
    # state["imagev"] += [ None] * llambda
    # state["images_filenames"] += [ None] * llambda
    # state["images_latents"] += [ None] * llambda

    # used_indexes_flatten += [False] * llambda
    if len(state["iterations"]) > 0:
        gen_iteration = state["iterations"][-1] + 1
    else:
        gen_iteration = 0
    # state["iterations"] += [ gen_iteration] * llambda

    if cols is not None and used_indexes_flatten is not None:
        for img_index in range(
            len(state["imagev"]), len(state["imagev"]) + len(latentv)
        ):
            column = cols[0] if img_index % 2 == 0 else cols[1]
            with column:
                cont = st.container()
                containers.insert(0, cont)
    if verbose:
        st.text(f'imagev len new  {len(state["imagev"])}')

    for i, latents in enumerate(latentv):

        if not state[
            "no_ml"
        ]:  # We are doing machine learning, a surrogate model and so on.
            use_rs = False
            st.text("Applying machine learning, i.e. a surrogate model.")
            if use_rs:
                st.text("Applying random search")
                opt = ng.optimizers.registry["RandomSearch"](
                    4 * state["resolution"] * state["resolution"],
                    budget=7,
                    num_workers=6,
                )
                opt.suggest(
                    short(latents.cpu().detach().numpy().flatten(), state["resolution"])
                )
                opt.minimize(
                    lambda x: tclf.predict_proba(
                        [
                            short(
                                (np.sqrt(len(x)) / np.linalg.norm(x)) * x,
                                state["resolution"],
                            )
                        ]
                    )[0][0]
                )
                recom = opt.recommend().value
                recom = (np.sqrt(len(recom)) / np.linalg.norm(recom)) * recom
            else:
                st.text("Applying differential evolution")
                opt = ng.optimizers.registry["DE"](
                    4 * state["resolution"] * state["resolution"],
                    budget=20 + i * 3,
                    num_workers=1,
                )
                z0 = latents.cpu().detach().numpy().flatten()
                epsilon = np.exp(i - llambda)

                def loss(x):
                    l = z0 + epsilon * x
                    l = (np.sqrt(len(l)) / np.linalg.norm(l)) * l
                    return tclf.predict_proba([short(l, state["resolution"])])[0][0]

                opt.minimize(loss)
                recom = z0 + epsilon * opt.recommend().value
                recom = (np.sqrt(len(recom)) / np.linalg.norm(recom)) * recom
            latents = (
                torch.from_numpy(
                    recom.reshape((1, 4, state["resolution"], state["resolution"]))
                )
                .half()
                .to("cuda")
            )
            latentv[i] = latents
            if verbose:
                st.text(f"new latents {latents.shape}")
        if verbose:
            st.text(f"generating latents {latents.shape}")
        # Here we call SD. This uses the GPU and takes a bit of time.
        if pipe is not None:
            image = pipe(prompt=state["prompt"], latents=latents.half()).images[0]
            # image_index = total_count - i - 1
            image_index = len(state["imagev"])
            image_name = "image_{}.png".format(image_index)
            if verbose:
                st.text(f"saving {image_name}")
            image.save(image_name)
            hr_face = f"image_{image_index}_face_hr.png"
            if os.path.exists(hr_face):
                os.remove(hr_face)
            hr_res = f"image_{image_index}_hr.png"
            if os.path.exists(hr_res):
                os.remove(hr_res)
            # state["imagev"][image_index] = image
            # state["iterations"][image_index] = gen_iteration
            # state["images_filenames"][image_index] = image_name
            state["imagev"].append(image)
            state["images_filenames"].append(image_name)
            state["iterations"].append(gen_iteration)
            state["prompts"].append(state["prompt"])
            used_indexes_flatten.append(False)
            # Here we show picture
            if containers is not None and used_indexes_flatten is not None:
                with containers[image_index]:
                    show_picture(
                        state, image_index, points, used_indexes_flatten, preview=True
                    )

        # state["images_latents"][image_index] = latents
        state["images_latents"].append(latents)
        if verbose:
            st.text(f'image latents last { state["images_latents"][-1].shape}')
        if generating_bar is not None:
            step = 1 / (len(latentv) + 1)
            start = 2 * step
            generating_bar.progress(start + 0.8 * i * step)
    state["used_indexes"] = []
    state["all_points"] = []
    with open(".state", "wb") as f:
        joblib.dump(state, f)
    return state

head_container = st.container()

# st.write(
#     "You can publish all these picture in [FB Ads](https://www.facebook.com/business/tools/ads-manager )"
# )

head_container.caption(
    """
 How to guide the processus:\n

 - you can modify the text (prompt) below\n

 - you can click on some boxes "Inspiration from this image" for guiding the search.\n

 - if you just click ONE box, next images will be more similar to that one. If you repeatedly prefer the same image, the variations will become closer to that image.\n

 - if you click ONE box and parts of the same image, the next images will be similar EXCEPT where you click (if you choose CHANGE in the SELECTED-AREA section). Generated images will be a mix of similar images and completly diffeent images.\n

 - if you click ONE box and parts of the same image, the next images will be similar MOSTLY where you click (if you choose SAVE in the SELECTED-AREA section). Generated images will be a mix of similar images and completly diffeent images.\n

 - if you click AT LEAST TWO boxes (two different images), they will be combined. Clicks on the image indicate the parts you want to keep.\n

 If you have an error, try to "reload last state" or "reset state" buttons. If UI stucks for more than 2 minutes with any reason try to reload this page and if it doesn't help - restart\
 in colab (Runtime->Restart and run all), then click on the newly created link"""
)

# Initialization
with st.sidebar:

    nn_options = ["SD v2", "SD v1.4", "SD v1.4 tuned on MidJourney", "Download another"]

    nn_to_gen = st.radio(
        "Advanced: Select Neural Network",
        key="NN choise",
        options=nn_options,
    )
    model_use_id = nn_options.index(nn_to_gen)
    if nn_to_gen == "SD v1.4":
        open_hugging_expander = True
    else:
        open_hugging_expander = False
    if nn_to_gen == "Download another":
        open_another_expander = True
    else:
        open_another_expander = False
    with st.expander("Download neural network ", open_another_expander):
        nn_path = st.text_input(
            label="past path from huggingface here",
            value="stabilityai/stable-diffusion-2-inpainting",
            help="past path for huggingface model as <vendor>/<model name> for example: stabilityai/stable-diffusion-2-inpainting",
        )
    with st.expander("Hugging face login ", open_hugging_expander):
        header_container = st.container()
        # login from colab temporary off
        # st.caption(
        #     "Instead of storing token each time you also can login to hugging face in colab"
        # )
        # state["colab_login"] = st.checkbox("Use colab login to hugging face")
        # try:
        #     with open(".hugging_token", "rb") as handle:
        #         state["token"] = joblib.load(handle)
        # except:
        #     pass

        with header_container:
            if "colab_login" not in state or state["colab_login"] == False:
                st.markdown(
                    """You need first register on [hugging face](https://huggingface.co/join)
        and get [access token](https://huggingface.co/settings/tokens)
        (it's in Menu/Settings/Access Tokens).
        The paste the token to the window"""
                )
                def_value = "paste token here"
                if (
                    "token" in state
                    and len(state["token"]) > MIN_TOKEN_LENGTH
                    and state["token"] != def_value
                ):
                    def_value = state["token"]

                token = st.text_input(
                    label="copy your hugging face token here",
                    value="",
                    help="paste your token here",
                )
                if len(token) > 5:
                    state["token"] = token
                    with open(".hugging_token", "wb") as f:
                        joblib.dump(token, f)
            else:

                st.caption("huggingface token stored or colab login")
    state["resolution"] = 64  # if model_use_id >0 else 96
    state["image_dimensions"] = 512  # if model_use_id >0 else 768
    state["load_inpaint"] = st.checkbox(
        "download inpaint NN (may need colab pro version with additional memory)"
    )
with st.sidebar:
    st.caption("""This checkbox for cases without good outcome.""")
    no_ml_checkbox = st.checkbox(
        "Use surrogate ML model based on your previous clicks. "
    )
    llambda = st.slider("How many pictures should we generate?", 1, 20, 4)

    correction_multiplier = st.slider(
        "How huge radius accross the point we should use", 1, 10, 2
    )
    state["correction_multiplier"] = correction_multiplier
    state["llambda"] = llambda
    refresh_button = st.button("Reload last state")
    if refresh_button:
        state["refreshes"] = 1
        with open(".state", "rb") as f:
            state = joblib.load(f)
    reset_bt = st.button("Reset state")
    if reset_bt:
        state = generate_state(prompt=prompt)
        state["state"] = state
    verbose = st.checkbox("verbose mode for testing", False)
    state["verbose"] = verbose
    state["same_z"] = st.checkbox(
        "Create a new picture exactly similar to the selected picture (up to prompt variations)"
    )

    drawing_mode = "point"
    if state["load_inpaint"]:
        drawing_mode = st.sidebar.selectbox("Drawing tool:", ("point", "polygon"))

    change_selected = st.sidebar.selectbox("Selected area :", ("change", "save"))
    state["change_selected"] = change_selected

# make sure you're logged in with `huggingface-cli login`
if state["load_inpaint"]:
    if "pipe_inpaint" not in state:
        with st.spinner(
            "Downloading Diffusion inpaint data. It would take about 40 seconds."
        ):

            pipe_inpaint = StableDiffusionInpaintPipeline.from_pretrained(
                "stabilityai/stable-diffusion-2-inpainting",
                # revision="fp16",
                # torch_dtype=torch.float16,
                #  use_auth_token=
            )

        state["pipe_inpaint"] = pipe_inpaint
    else:
        pipe_inpaint = state["pipe_inpaint"]
    pipe_inpaint = pipe_inpaint.to("cuda")
if nn_to_gen + "pipe" not in state or state["model_use_id"] != model_use_id:
    dict_names = {0: "SD2.0", 1: "SD1.4", 2: "SD1.4 tuned on MJ", 3: "other"}
    model_name = dict_names[model_use_id] if model_use_id in dict_names else "unknown"
    with st.spinner(f"Downloading {model_name} diffusion model should take ~40 seconds."):
        if model_use_id == 0:
            pipe_file_name = ".pipeSD2"

        elif model_use_id == 1:
            pipe_file_name = ".pipeSD"
        elif model_use_id == 2:
            pipe_file_name = ".pipeSDMJ"
        elif model_use_id == 3:
            pipe_file_name = ".pipeAnother"
        pipe_file = Path(pipe_file_name)
        if pipe_file.is_file():
            with open(pipe_file_name, "rb") as pipeFile:
                pipe = joblib.load(pipeFile)
        else:
            use_auth_token = (
                True
                if "token" not in state
                or ("colab_login" in state and state["colab_login"])
                else state["token"]
            )
            try:

                if model_use_id == 0:
                    model_sd2 = "stabilityai/stable-diffusion-2-base"
                    pipe = StableDiffusionPipeline.from_pretrained(
                        model_sd2,
                        revision="fp16",
                        torch_dtype=torch.float16  # ,
                        #   use_auth_token=use_auth_token,
                    )
                elif model_use_id == 1:
                    model_sd = "CompVis/stable-diffusion-v1-4"
                    if state["verbose"]:
                        st.text(f"using token {use_auth_token}")

                    pipe = StableDiffusionPipeline.from_pretrained(
                        model_sd,
                        revision="fp16",
                        torch_dtype=torch.float16,
                        use_auth_token=use_auth_token,
                    )
                elif model_use_id == 2:

                    model_sd_mj = "prompthero/midjourney-v4-diffusion"
                    pipe = StableDiffusionPipeline.from_pretrained(
                        model_sd_mj,
                        torch_dtype=torch.float16,
                        use_auth_token=use_auth_token,
                    )
                elif model_use_id == 3:

                    model_nn = nn_path
                    pipe = StableDiffusionPipeline.from_pretrained(
                        model_nn,
                        torch_dtype=torch.float16,
                        use_auth_token=use_auth_token,
                    )
                else:
                    st.error("bad model id")
            except Exception:
                token_to_print = (
                    ""
                    if type(use_auth_token) is bool
                    else f"current token {use_auth_token}"
                )
                e = RuntimeError(
                    f"Problem with Diffusion downloading, please paste huggingface token in the box under << HuggingFace login >> on the left  {token_to_print}"
                )
                st.exception(e)
                st.stop()
            with open(pipe_file_name, "wb") as f:
                joblib.dump(pipe, f)

        state["model_use_id"] = model_use_id
        state[nn_to_gen + "pipe"] = pipe
else:
    pipe = state[nn_to_gen + "pipe"]
pipe = pipe.to("cuda")
create_tab, view_tab, gif_tab = st.tabs(
    [
        "Images to generate",
        "High-resolution images (select one image first)",
        "Animation generation (select two images first)",
    ]
)
with create_tab:

    header_col1, header_col2 = st.columns([4, 1])
    with header_col1:
        st.caption("Write sentence to visualise here")
        prompt = st.text_area(
            label="Type a sentence to generate a picture",
            placeholder="A [Type of picture] of a [composition], [style]",
            help="""'
    In general, the best diffusion prompts will have this form:

      “A [Type of picture] of a [composition], [style]”
      Type of picture: photo or painting (with styles brush/matte/oil) or 3d renger or map (modern/medieval)
      Compositions: what is on a picture. Usually several objects with minimum verbs.
      Style: style of drawing, for example: steampunk or cyberpunk, realistic, Van Gogh, etc.
      You can use famous person names to use their faces or famous painters/creators to use their styles.
  """,
        )
        if prompt != state["prompt"]:
            rework = True
            state["prompt"] = prompt
        else:
            rework = False
    with header_col2:
        regenerate_button = st.button("Generate")
        # st.caption("if page stuck")

    with st.expander("How to guide the next generation of images"):
        st.caption(
            """
            * No selected image = just rerun generation with new random.
            * One selected image and no point = we generate variations of this image.
            * One image and selected points = regenerate image with keeping or abandoning area around the points according to choice (see bottom left)
            * Several images without points = we merge this whole images.
            * Several images with selected points = we merge these image with keeping or abandoning area around the points according to choice  (see bottom left)
            You also have an options:
            ML prediction checkbox - generation images based on previous choices
            Fix face and generate high qulaity image - generate images of the next tap
            WARNING: for the first several iterations it's usually better to just select one image.
            """
        )

    picture_col1, picture_col2 = st.columns(2)
    cols = [picture_col1, picture_col2]
    containers = []
    for img_index in range(len(state["imagev"])):
        column = cols[0] if img_index % 2 == 0 else cols[1]
        with column:
            cont = st.container()
            containers.insert(0, cont)
    if regenerate_button:
        st.caption(
            "generation progress may take several minutes, depends on GPU allocation and number of images"
        )
        generating_bar = head_container.progress(0)
        state = generate_pictures(
            state,
            pipe=pipe,
            verbose=verbose,
            generating_bar=generating_bar,
            cols=cols,
            containers=containers,
            points=points,
            used_indexes_flatten=used_indexes_flatten,
        )
        st.session_state["state"] = state
        st.experimental_rerun()

    used_indexes_flatten = [False] * len(state["images_filenames"])
    points = []
    num_image_to_show = len(state["images_filenames"])
    num_image_to_show -= state["llambda"] if regenerate_button else 0
    for image_index in reversed(list(range(num_image_to_show))):
        #       column = cols[0] if image_index % 2 == 0 else cols[1]
        containers[image_index].empty()
        with containers[image_index]:
            show_picture(state, image_index, points, used_indexes_flatten)

prev_iter = -1
for index_image, iter in enumerate(state["iterations"]):
    if iter != prev_iter:
        st.text(f'On iteration {iter} prompt was: {state["prompts"][index_image]}')
        prev_iter = iter


if len(points) > 0:
    state["all_points"] = pd.concat(points)
    state["polygons"] = state["all_points"][state["all_points"]["type"] == "path"]
    state["all_points"] = state["all_points"][state["all_points"]["type"] == "circle"]


if len(state["all_points"]) < 1:
    state["all_points"] = pd.DataFrame(columns=POINT_COLUMNS)

state["used_indexes"] = [
    img_idx for img_idx, val in enumerate(used_indexes_flatten) if val == True
]

if set(state["movie_order"]) != set(state["used_indexes"]):
    state["movie_order"] = state["used_indexes"]
if verbose:
    st.text("Internal log. Points stored")
    st.dataframe(state["all_points"][["top", "left", "image", "x", "y"]])

    st.text(f"state all { {k:str(v)[:30] for k,v in state.items()} }")

st.session_state["state"] = state
if verbose:
    st.text(f'len state all_points after set {len(state["all_points"])}')

with view_tab:
    files = os.listdir()
    files = [f for f in files if "hr" in f.lower() and "png" in f.lower()]
    for index, filename in enumerate(files):
        st.markdown("""---""")
        st.image(Image.open(filename))

        if filename in links_dict.keys():
            st.markdown(
                f'<a href="{links_dict[filename]}"> <img src="{fb_logo}" width="100" height="30"> </a>',
                unsafe_allow_html=True,
            )
        with open(filename, "rb") as file:
            btn_d = st.download_button(
                label="Download high res image",
                data=file,
                file_name=filename,
                mime="image/jpg",
                key=f"download_button_{index}",
            )
with gif_tab:
    if len(state["movie_order"]) > 1:
        gif_cols = st.columns(len(state["movie_order"]))

        for index, chosen in enumerate(state["movie_order"]):
            with gif_cols[index]:
                image_filename = state["images_filenames"][chosen]
                image = Image.open(image_filename)
                st.image(image)
                if index > 0:
                    btn_r = st.button(
                        label="<-",
                        key=f"gif_right_button_{index}",
                    )
                    if btn_r:
                        state["movie_order"][index], state["movie_order"][index - 1] = (
                            state["movie_order"][index - 1],
                            state["movie_order"][index],
                        )
                        st.experimental_rerun()

                if index < len(state["movie_order"]) - 1:
                    btn_l = st.button(
                        label="->",
                        key=f"gif_left_button_{index}",
                    )
                    if btn_l:
                        state["movie_order"][index], state["movie_order"][index + 1] = (
                            state["movie_order"][index + 1],
                            state["movie_order"][index],
                        )
                        st.experimental_rerun()
        state["duration_between_slides"] = st.slider(
            "Interval between slides in ms",
            min_value=10,
            max_value=100,
            value=30,
            step=10,
        )
        if st.button("Generate Gif"):
            generating_bar_m = st.progress(0)
            temp_container = st.container()

            state = generate_movie(
                state,
                pipe,
                verbose=verbose,
                generating_bar=generating_bar_m,
                guidance_scale=7.5,
                temp_container=temp_container,
            )
            st.experimental_rerun()

    for gif_name in state["gif_names"]:
        file_ = open(gif_name, "rb")
        contents = file_.read()
        data_url = base64.b64encode(contents).decode("utf-8")
        file_.close()
        st.text("Genrerated gif")
        st.markdown(
            f'<img src="data:image/gif;base64,{data_url}" alt="cat gif">',
            unsafe_allow_html=True,
        )

        with open(gif_name, "rb") as file:
            btn_d = st.download_button(
                label="Download gif",
                data=file,
                file_name=gif_name,
                mime="image/gif",
                key=f"download_gif_{gif_name}",
            )
