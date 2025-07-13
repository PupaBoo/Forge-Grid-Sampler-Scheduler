import gradio as gr
import modules.scripts as scripts
from modules import shared
from modules.processing import process_images, Processed
from modules.sd_samplers import samplers
from modules.sd_schedulers import schedulers
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import random
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def safe_processed(*args):
    processed = Processed(*args)
    processed.info = str(processed.info or "")
    processed.comments = str(processed.comments or "")
    return processed


def wrap_text_to_fit(draw, text, font_path, max_width, initial_font_size=18):
    if font_path and Path(font_path).exists():
        initial_font_size = max(12, min(40, max_width // 50))
        font_size = initial_font_size
        while font_size > 8:
            font = ImageFont.truetype(str(font_path), font_size)
            lines, cur = [], ""
            for word in text.split():
                test = f"{cur} {word}".strip()
                bbox = font.getbbox(test)
                w = bbox[2] - bbox[0]
                if w <= max_width:
                    cur = test
                else:
                    if cur:
                        lines.append(cur)
                    cur = word
            if cur:
                lines.append(cur)
            if lines:
                return lines, font
            font_size -= 2
        logger.warning("Font size reduced to minimum, text may be clipped")
        return lines, font
    else:
        logger.warning("Font file not found, using default font")
        font = ImageFont.load_default()
        lines, cur = [], ""
        for word in text.split():
            test = f"{cur} {word}".strip()
            bbox = font.getbbox(test)
            w = bbox[2] - bbox[0]
            if w <= max_width:
                cur = test
            else:
                if cur:
                    lines.append(cur)
                cur = word
        if cur:
            lines.append(cur)
        return lines, font


class Script(scripts.Script):
    def title(self):
        return "🔬 Forge Grid: Sampler × Scheduler"

    def show(self, is_img2img):
        return not is_img2img

    def ui(self, is_img2img):
        sampler_list = [s.name for s in samplers if hasattr(s, "name")]
        scheduler_list = [s.label for s in schedulers if hasattr(s, "label")]

        mode_selector = gr.Radio(
            ["XY Grid", "Batch Grid"], value="XY Grid", label="🏁 Grid Mode")
        stop_btn = gr.Button("🛑 Stop Grid Generation")
        stop_btn.click(lambda: shared.state.interrupt(), [], [])

        xy_group = gr.Group(visible=True)
        with xy_group:
            xy_samplers = gr.Dropdown(
                choices=sampler_list, multiselect=True, label="🗳️ Sampler(s)")
            xy_schedulers = gr.Dropdown(
                choices=scheduler_list, multiselect=True, label="📆 Scheduler(s)")
            axis_x = gr.Radio(
                ["Sampler", "Scheduler"], value="Sampler", label="🧭 Axis X")
            axis_y = gr.Radio(
                ["Sampler", "Scheduler"], value="Scheduler", label="🧭 Axis Y")

            def validate_axes(x, y):
                if x == y:
                    logger.error("Axes X and Y must be different!")
                    return x, y
                return x, y

            axis_x.change(validate_axes, inputs=[
                          axis_x, axis_y], outputs=[axis_x, axis_y])
            axis_y.change(validate_axes, inputs=[
                          axis_x, axis_y], outputs=[axis_x, axis_y])

        batch_group = gr.Group(visible=False)
        with batch_group:
            dropdown_sampler = gr.Dropdown(
                choices=sampler_list, label="🗳️ Select Sampler")
            dropdown_scheduler = gr.Dropdown(
                choices=scheduler_list, label="📆 Select Scheduler")
            batch_axis_x = gr.Radio(
                ["Sampler", "Scheduler"], value="Sampler", label="🧭 Axis X")
            batch_axis_y = gr.Radio(
                ["Sampler", "Scheduler"], value="Scheduler", label="🧭 Axis Y")
            add_pair_btn = gr.Button("➕ Add Pair")
            clear_pairs_btn = gr.Button("🧹 Clear All Pairs")
            pair_list = gr.Textbox(
                label="🔗 Added Pairs", placeholder="Sampler, Scheduler per line", lines=6, interactive=True)
            pair_count = gr.Textbox(label="🧮 Total Pairs", interactive=False)
            pair_state = gr.State([])

            def validate_batch_axes(x, y):
                if x == y:
                    logger.error("Batch Grid axes (X, Y) must be different!")
                    return x, y
                return x, y

            batch_axis_x.change(validate_batch_axes, inputs=[
                                batch_axis_x, batch_axis_y], outputs=[batch_axis_x, batch_axis_y])
            batch_axis_y.change(validate_batch_axes, inputs=[
                                batch_axis_x, batch_axis_y], outputs=[batch_axis_x, batch_axis_y])

            def validate_pairs(txt, samplers, schedulers):
                samplers = samplers or [
                    s.name for s in samplers if hasattr(s, "name")]
                schedulers = schedulers or [
                    s.label for s in schedulers if hasattr(s, "label")]
                pairs = []
                for line in txt.strip().splitlines():
                    if "," in line:
                        s, sch = [x.strip() for x in line.split(",", 1)]
                        if s not in samplers or sch not in schedulers:
                            logger.error(
                                f"Invalid pair: {line}. Sampler and Scheduler must be valid!")
                            return pairs, str(len(pairs))
                        pairs.append(f"{s},{sch}")
                return pairs, str(len(pairs))

            pair_list.change(
                validate_pairs,
                inputs=[pair_list, gr.State(
                    sampler_list), gr.State(scheduler_list)],
                outputs=[pair_state, pair_count]
            )

            def add_pair_fn(s, sch, cur, samplers, schedulers):
                samplers = samplers or [
                    s.name for s in samplers if hasattr(s, "name")]
                schedulers = schedulers or [
                    s.label for s in schedulers if hasattr(s, "label")]
                if not s or not sch:
                    logger.error("Sampler and Scheduler must be selected!")
                    return cur, "\n".join(cur), str(len(cur))
                if s not in samplers or sch not in schedulers:
                    logger.error(f"Invalid pair: {s}, {sch}")
                    return cur, "\n".join(cur), str(len(cur))
                p_ = f"{s},{sch}"
                if p_ in cur:
                    return cur, "\n".join(cur), str(len(cur))
                new = cur + [p_]
                return new, "\n".join(new), str(len(new))

            def clear_pairs_fn():
                return [], "", "0"

            add_pair_btn.click(
                add_pair_fn,
                inputs=[dropdown_sampler, dropdown_scheduler, pair_state,
                        gr.State(sampler_list), gr.State(scheduler_list)],
                outputs=[pair_state, pair_list, pair_count]
            )
            clear_pairs_btn.click(
                clear_pairs_fn,
                inputs=[],
                outputs=[pair_state, pair_list, pair_count]
            )

        pos_prompt = gr.Textbox(label="✅ Positive Prompt",
                                placeholder="What to include", lines=3)
        neg_prompt = gr.Textbox(label="⛔ Negative Prompt",
                                placeholder="What to avoid", lines=2)
        seed = gr.Textbox(label="🎲 Seed (optional)",
                          placeholder="Leave blank for random")
        steps = gr.Slider(1, 100, value=50, step=1, label="🚀 Steps")
        cfg_scale = gr.Slider(1.0, 30.0, value=7.5,
                              step=0.1, label="🎯 CFG Scale")
        width = gr.Slider(256, 2048, value=832, step=64, label="↔️ Width")
        height = gr.Slider(256, 2048, value=1216, step=64, label="↕️ Height")
        padding = gr.Slider(0, 200, value=8, step=2, label="📏 Padding (px)")
        save_formats = gr.CheckboxGroup(choices=["WEBP", "PNG"], value=[
                                        "WEBP"], label="💾 Save as")
        show_labels = gr.Checkbox(label="📝 Add Text Labels", value=True)
        auto_downscale = gr.Checkbox(
            label="🪄 Auto-downscale if too large", value=True)
        save_large = gr.Checkbox(
            label="🧩 Enable saving large grids (if 50+ cells)", value=False)

        mode_selector.change(
            lambda m: {
                xy_group: gr.update(visible=m == "XY Grid"),
                batch_group: gr.update(visible=m == "Batch Grid")
            },
            inputs=[mode_selector],
            outputs=[xy_group, batch_group]
        )

        return [
            mode_selector,
            xy_samplers, xy_schedulers, axis_x, axis_y,
            dropdown_sampler, dropdown_scheduler, batch_axis_x, batch_axis_y, pair_list, pair_state, pair_count,
            pos_prompt, neg_prompt, seed, steps, cfg_scale, width, height, padding,
            save_formats, show_labels, auto_downscale, save_large
        ]

    def run(self, p, *args):
        shared.state.interrupted = False
        (mode, xy_samplers, xy_schedulers, axis_x, axis_y,
         dropdown_sampler, dropdown_scheduler, batch_axis_x, batch_axis_y, pair_list, pair_state, pair_count,
         pos_prompt, neg_prompt, seed, steps, cfg_scale, width, height, padding,
         save_formats, show_labels, auto_downscale, save_large) = args

        try:
            sd = int(seed) if seed.strip() else random.randint(1, 2**32 - 1)
        except ValueError:
            logger.warning(
                "Invalid seed format, using random. Please enter a valid integer or leave blank.")
            sd = random.randint(1, 2**32 - 1)

        p.width = int(width)
        p.height = int(height)
        p.steps = int(steps)
        p.cfg_scale = float(cfg_scale)
        p.seed = sd
        p.negative_prompt = neg_prompt
        p.override_settings = {}
        p.extra_generation_params = {}

        font_path = Path(__file__).parent / "Barlow-SemiBold.ttf"
        if not font_path.exists():
            logger.warning(
                f"Font file {font_path} not found, using default font")
            font_path = None

        LIMIT = 16383
        all_images = []
        output_dir = Path(p.outpath_samples) / "Forge_Grid_Sampler_x_Scheduler"
        output_dir.mkdir(parents=True, exist_ok=True)

        if mode == "Batch Grid":
            batch_pairs = []
            sampler_list = [s.name for s in samplers if hasattr(s, "name")]
            scheduler_list = [
                s.label for s in schedulers if hasattr(s, "label")]
            for line in pair_list.strip().splitlines():
                if "," in line:
                    s, sch = [x.strip() for x in line.split(",", 1)]
                    if s not in sampler_list or sch not in scheduler_list:
                        logger.error(
                            f"Invalid pair: {line}. Please use valid Sampler and Scheduler names.")
                        return safe_processed(p, [], sd, sd, 0.0, pos_prompt, neg_prompt, f"⚠️ Invalid pair: {line}.", "")
                batch_pairs.append((s, sch))

            if not batch_pairs:
                logger.error(
                    "No valid pairs provided. Please add at least one valid Sampler-Scheduler pair.")
                return safe_processed(p, [], sd, sd, 0.0, pos_prompt, neg_prompt, "⚠️ No valid pairs.", "")

            if len(batch_pairs) > 50 and not save_large:
                logger.error(
                    "Grid too large for Batch Grid mode. Please enable 'Enable saving large grids' or reduce the number of pairs.")
                return safe_processed(p, [], sd, sd, 0.0, pos_prompt, neg_prompt, "⚠️ Grid too large.", "")

            shared.state.job_count = len(batch_pairs)
            shared.state.job = 0

            images = []
            labels = []
            for samp, sch in batch_pairs:
                if shared.state.interrupted:
                    logger.info("Generation interrupted by user")
                    return safe_processed(p, all_images, sd, sd, 0.0, pos_prompt, neg_prompt, "❗ Stopped by user.", "")

                p.prompt = pos_prompt
                p.sampler_name = samp
                p.scheduler_name = sch

                logger.info(
                    f"[BATCH] Generating: prompt={p.prompt}, sampler={samp}, scheduler={sch}")

                try:
                    res = process_images(p)
                    img = res.images[0] if res and getattr(
                        res, "images", None) else None
                    if img is None:
                        raise ValueError("No image returned")
                    img_array = np.array(img)
                    if np.all(img_array[..., :3] == [0, 0, 0]) or img_array.size == 0:
                        raise ValueError("Generated image is black or empty")
                except Exception as e:
                    logger.error(f"[BATCH] Generation failed: {str(e)}")
                    return safe_processed(p, all_images, sd, sd, 0.0, pos_prompt, neg_prompt, f"⚠️ Generation failed: {str(e)}.", "")

                images.append(img)
                labels.append(f"{samp} | {sch}")
                shared.state.job += 1

            cols = len(images)
            rows = 1
            dummy = ImageDraw.Draw(Image.new("RGB", (10, 10)))
            if show_labels:
                lh = []
                for L in labels:
                    lines, fnt = wrap_text_to_fit(dummy, L, font_path, p.width)
                    lh.append(
                        (fnt.getbbox("A")[3] - fnt.getbbox("A")[1] + 2) * len(lines))
                label_h = max(lh) + 30
            else:
                label_h = 0

            grid_w = cols * p.width + (cols - 1) * padding
            grid_h = p.height + label_h
            grid = Image.new("RGB", (grid_w, grid_h), (255, 255, 255))
            draw = ImageDraw.Draw(grid)

            x_vals = [samp if batch_axis_x ==
                      "Sampler" else sch for samp, sch in batch_pairs]
            y_vals = [""]

            for i, img in enumerate(images):
                x0 = i * (p.width + padding)
                grid.paste(img, (x0, 0))
                if show_labels:
                    lines, fnt = wrap_text_to_fit(
                        draw, labels[i], font_path, p.width)
                    y0 = p.height
                    for ln in lines:
                        bbox = fnt.getbbox(ln)
                        tw = bbox[2] - bbox[0]
                        th = bbox[3] - bbox[1]
                        draw.text((x0 + (p.width - tw) // 2, y0),
                                  ln, (0, 0, 0), font=fnt)
                        y0 += th + 2

            base = "batchgrid"
            for fmt in save_formats:
                ext = fmt.lower()
                out = grid
                if out.width > LIMIT or out.height > LIMIT:
                    if auto_downscale:
                        sc = min(LIMIT / out.width, LIMIT / out.height)
                        out = out.resize(
                            (int(out.width * sc), int(out.height * sc)), Image.Resampling.LANCZOS, quality=100)
                    else:
                        logger.warning(f"Grid too large for {fmt}, skipping")
                        continue
                out.save(output_dir / f"{base}.{ext}", fmt.upper(),
                         quality=100 if ext == "webp" else None)

            all_images.append(grid)

        else:
            x_vals = xy_samplers if axis_x == "Sampler" else xy_schedulers
            y_vals = xy_schedulers if axis_y == "Scheduler" else xy_samplers

            if not x_vals or not y_vals:
                logger.error(
                    "Invalid axis values. Please ensure both axes have valid selections.")
                return safe_processed(p, [], sd, sd, 0.0, pos_prompt, neg_prompt, "⚠️ Invalid axis values.", "")

            if len(x_vals) * len(y_vals) > 50 and not save_large:
                logger.error(
                    "Grid too large for XY Grid mode. Please enable 'Enable saving large grids' or reduce the number of selections.")
                return safe_processed(p, [], sd, sd, 0.0, pos_prompt, neg_prompt, "⚠️ Grid too large.", "")

            images = []
            labels = []
            shared.state.job_count = len(x_vals) * len(y_vals)
            shared.state.job = 0

            for yv in y_vals:
                for xv in x_vals:
                    if shared.state.interrupted:
                        logger.info("Generation interrupted by user")
                        return safe_processed(p, all_images, sd, sd, 0.0, pos_prompt, neg_prompt, "❗ Stopped by user.", "")

                    samp = xv if axis_x == "Sampler" else yv
                    sched = yv if axis_y == "Scheduler" else xv
                    p.prompt = pos_prompt
                    p.sampler_name = samp
                    p.scheduler_name = sched

                    logger.info(
                        f"[XY] Generating: prompt={p.prompt}, sampler={samp}, scheduler={sched}")

                    try:
                        res = process_images(p)
                        img = res.images[0] if res and getattr(
                            res, "images", None) else None
                        if img is None:
                            raise ValueError("No image returned")
                        img_array = np.array(img)
                        if np.all(img_array[..., :3] == [0, 0, 0]) or img_array.size == 0:
                            raise ValueError(
                                "Generated image is black or empty")
                    except Exception as e:
                        logger.error(f"[XY] Generation failed: {str(e)}")
                        return safe_processed(p, all_images, sd, sd, 0.0, pos_prompt, neg_prompt, f"⚠️ Generation failed: {str(e)}.", "")

                    images.append(img)
                    labels.append(f"{samp} | {sched}")
                    shared.state.job += 1

            cols = len(x_vals)
            rows = len(y_vals)
            dummy = ImageDraw.Draw(Image.new("RGB", (10, 10)))
            if show_labels:
                lh = []
                for L in labels:
                    lines, fnt = wrap_text_to_fit(dummy, L, font_path, p.width)
                    lh.append(
                        (fnt.getbbox("A")[3] - fnt.getbbox("A")[1] + 2) * len(lines))
                label_h = max(lh) + 30
            else:
                label_h = 0

            grid_w = cols * p.width + (cols - 1) * padding
            grid_h = rows * (p.height + label_h) + (rows - 1) * padding
            grid = Image.new("RGB", (grid_w, grid_h), (255, 255, 255))
            draw = ImageDraw.Draw(grid)

            for idx, img in enumerate(images):
                row, col = divmod(idx, cols)
                x0 = col * (p.width + padding)
                y0 = row * (p.height + label_h + padding)
                grid.paste(img, (x0, y0))
                if show_labels:
                    lines, fnt = wrap_text_to_fit(
                        draw, labels[idx], font_path, p.width)
                    y1 = y0 + p.height
                    for ln in lines:
                        bbox = fnt.getbbox(ln)
                        tw = bbox[2] - bbox[0]
                        th = bbox[3] - bbox[1]
                        draw.text((x0 + (p.width - tw) // 2, y1),
                                  ln, (0, 0, 0), font=fnt)
                        y1 += th + 2

            base = "xygrid"
            for fmt in save_formats:
                ext = fmt.lower()
                out = grid
                if out.width > LIMIT or out.height > LIMIT:
                    if auto_downscale:
                        sc = min(LIMIT / out.width, LIMIT / out.height)
                        out = out.resize(
                            (int(out.width * sc), int(out.height * sc)), Image.Resampling.LANCZOS, quality=100)
                    else:
                        logger.warning(f"Grid too large for {fmt}, skipping")
                        continue
                out.save(output_dir / f"{base}.{ext}", fmt.upper(),
                         quality=100 if ext == "webp" else None)

            all_images.append(grid)

        info = f"✅ Done: {len(all_images)} grid(s)"
        logger.info(info)
        return safe_processed(p, all_images, sd, sd, 0.0, pos_prompt, neg_prompt, info, "")
