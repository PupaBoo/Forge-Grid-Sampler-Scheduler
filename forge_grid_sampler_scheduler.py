import math
import gradio as gr
import modules.scripts as scripts
from modules.sd_samplers import samplers
from modules.sd_schedulers import schedulers
from modules.processing import process_images, Processed
from modules.sd_samplers import samplers
from modules.sd_schedulers import schedulers
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import random
import logging
import sys
logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger(__name__)


def generate_or_warn(p, sampler, scheduler, font_path):
    try:
        apply_params(p, sampler, scheduler)

        res = process_images(p)
        img = res.images[0] if res and getattr(res, "images", None) else None
        if img is None:
            raise ValueError()

    except Exception as e:
        logger.warning(
            f"⚠️ Generation failed for Sampler = '{sampler}', Scheduler = '{scheduler}': {e}")
        img = create_fallback_image(
            p.width, p.height, sampler, scheduler, font_path)

    return img


def apply_params(p, sampler_label, scheduler_label):
    sampler_label = sampler_label.strip()
    sampler_index = next((i for i, s in enumerate(
        samplers) if s.name == sampler_label), None)
    if sampler_index is None:
        raise ValueError(f"Sampler '{sampler_label}' not found")

    p.sampler_index = sampler_index
    p.sampler_name = samplers[sampler_index].name

    p.scheduler = scheduler_label.strip()

    logger.info(
        f"🧪 Applied Sampler = '{sampler_label}' → index {sampler_index}")
    logger.info(f"🧪 Applied Scheduler = '{scheduler_label.strip()}'")


def create_fallback_image(width, height, sampler, scheduler, font_path):
    img = Image.new("RGB", (width, height), (255, 230, 230))
    draw = ImageDraw.Draw(img)
    font = ImageFont.truetype(
        str(font_path), 38) if font_path else ImageFont.load_default()
    msg = f"⚠️ Pair failed\n{sampler} × {scheduler}"
    lines = msg.split("\n")
    total_h = len(lines) * sum(font.getmetrics())
    y = (height - total_h) // 2
    for line in lines:
        text_w = font.getbbox(line)[2]
        x = (width - text_w) // 2
        draw.text((x, y), line, font=font, fill=(150, 0, 0))
        y += sum(font.getmetrics())
    return img


def safe_processed(*args):
    processed = Processed(*args)
    processed.info = str(processed.info or "")
    processed.comments = str(processed.comments or "")
    return processed


def wrap_text(text, font, max_width):
    words = text.split()
    lines, cur = [], ""
    for word in words:
        test = f"{cur} {word}".strip()
        w = font.getbbox(test)[2]
        if w <= max_width - 10:
            cur = test
        else:
            if cur:
                lines.append(cur)
            cur = word
    if cur:
        lines.append(cur)
    return lines


def wrap_text_to_fit(draw, text, font_path, max_width, initial_size=42, min_size=18):
    for size in range(initial_size, min_size - 1, -1):
        font = ImageFont.truetype(
            str(font_path), size) if font_path else ImageFont.load_default()
        lines = wrap_text(text, font, max_width)
        if all(font.getbbox(line)[2] <= max_width - 10 for line in lines):
            return lines, font
    font = ImageFont.truetype(
        str(font_path), min_size) if font_path else ImageFont.load_default()
    return wrap_text(text, font, max_width), font


def create_batch_grid(images, width=832, height=1216, padding=10, margin_top=6, margin_side=10, bg_color=(255, 255, 255)):
    if not images:
        return Image.new("RGB", (width, height), bg_color)

    cols = math.ceil(math.sqrt(len(images)))
    rows = math.ceil(len(images) / cols)

    cell_w = images[0].width
    cell_h = images[0].height

    grid_w = cols * cell_w + (cols - 1) * padding
    grid_h = rows * cell_h + (rows - 1) * padding

    total_w = grid_w + 2 * margin_side
    total_h = grid_h + 2 * margin_top

    grid = Image.new("RGB", (total_w, total_h), bg_color)

    for i, img in enumerate(images):
        r, c = divmod(i, cols)
        x = margin_side + c * (cell_w + padding)
        y = margin_top + r * (cell_h + padding)
        grid.paste(img, (x, y))

    return grid


def annotate_batch_image(img, sampler, scheduler, font_path=None):
    label_text = f"{sampler.strip()}\n{scheduler.strip()}"
    dummy_draw = ImageDraw.Draw(Image.new("RGB", (10, 10)))
    lines, font = wrap_text_to_fit(
        dummy_draw, label_text, font_path, img.width)

    ascent, descent = font.getmetrics()
    line_h = ascent + descent
    total_text_h = line_h * len(lines)

    top_margin = 10
    gap = 6
    bottom_margin = descent + 6

    final_h = top_margin + img.height + gap + total_text_h + bottom_margin
    final_w = img.width

    out = Image.new("RGB", (final_w, final_h), (255, 255, 255))
    out.paste(img, (0, top_margin))

    draw = ImageDraw.Draw(out)
    line_h = sum(font.getmetrics())
    total_text_h = line_h * len(lines)
    y_text = top_margin + img.height + gap + \
        ((final_h - top_margin - img.height - gap - total_text_h) // 2)

    for line in lines:
        text_w = font.getbbox(line)[2]
        x_text = (final_w - text_w) // 2
        draw.text((x_text, y_text), line, font=font, fill=(0, 0, 0))
        y_text += line_h

    return out


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
            sampler_axis = gr.Radio(
                ["Axis X", "Axis Y"], value="Axis X", label="🧭 Place Sampler on")

        batch_group = gr.Group(visible=False)
        with batch_group:
            dropdown_sampler = gr.Dropdown(
                choices=sampler_list, label="🗳️ Select Sampler")
            dropdown_scheduler = gr.Dropdown(
                choices=scheduler_list, label="📆 Select Scheduler")
            add_pair_btn = gr.Button("➕ Add Pair")
            clear_pairs_btn = gr.Button("🧹 Clear All Pairs")
            pair_list = gr.Textbox(
                label="🔗 Added Pairs", placeholder="Sampler, Scheduler per line", lines=6)
            pair_count = gr.Textbox(label="🧮 Total Pairs", interactive=False)
            pair_state = gr.State([])

            def parse_pairs(txt):
                lines = [line.strip()
                         for line in txt.splitlines() if "," in line]
                return list(dict.fromkeys(lines))

            pair_list.change(
                lambda txt: (
                    parse_pairs(txt),
                    str(len(parse_pairs(txt)))
                ),
                inputs=[pair_list],
                outputs=[pair_state, pair_count]
            )

            add_pair_btn.click(
                lambda s, sch, cur: cur +
                [f"{s},{sch}"] if s and sch and f"{s},{sch}" not in cur else cur,
                inputs=[dropdown_sampler, dropdown_scheduler, pair_state],
                outputs=[pair_state]
            )

            pair_state.change(
                lambda st: ("\n".join(st), str(len(st))),
                inputs=[pair_state],
                outputs=[pair_list, pair_count]
            )

            clear_pairs_btn.click(lambda: [], [], [pair_state])

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
        padding = gr.Slider(0, 200, value=10, step=2, label="📏 Padding (px)")
        save_formats = gr.CheckboxGroup(choices=["WEBP", "PNG"], value=[
                                        "WEBP"], label="💾 Save As")
        show_labels = gr.Checkbox(label="📝 Add Labels", value=True)
        save_cells = gr.Checkbox(
            label="💾 Save each cell individually", value=True)

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
            xy_samplers, xy_schedulers, sampler_axis,
            dropdown_sampler, dropdown_scheduler,
            pair_list, pair_state, pair_count,
            pos_prompt, neg_prompt, seed, steps, cfg_scale,
            width, height, padding,
            save_formats, show_labels, save_cells
        ]

    def run(self, p, *args):
        (mode, xy_samplers, xy_schedulers, sampler_axis,
         dropdown_sampler, dropdown_scheduler,
         pair_list, pair_state, pair_count,
         pos_prompt, neg_prompt, seed, steps, cfg_scale,
         width, height, padding,
         save_formats, show_labels, save_cells) = args

        try:
            sd = int(seed.strip()) if seed.strip(
            ) else random.randint(1, 2**32 - 1)
        except ValueError:
            sd = random.randint(1, 2**32 - 1)

        p.seed = sd
        p.prompt = pos_prompt
        p.negative_prompt = neg_prompt
        p.steps = steps
        p.cfg_scale = cfg_scale
        p.width = int(width)
        p.height = int(height)

        p.extra_generation_params = {}

        font_path = Path(__file__).resolve().parent / "Barlow-SemiBold.ttf"
        if not font_path.exists():
            logger.warning("⚠️ Font not found — using default")
            font_path = None


        output_dir = Path(p.outpath_samples) / "Forge_Grid_Sampler_x_Scheduler"
        output_dir.mkdir(parents=True, exist_ok=True)
        cells_dir = output_dir / "cells"
        cells_dir.mkdir(parents=True, exist_ok=True)

        LIMIT = 16383
        auto_downscale = True

        if mode == "Batch Grid":
            pairs = [line.strip()
                     for line in pair_list.strip().splitlines() if "," in line]
            if not pairs:
                logger.warning("⚠️ No valid pairs found in 🔗 Added Pairs.")
                return safe_processed(p, [], sd, sd, 0.0, pos_prompt, neg_prompt, "⚠️ No pairs found", "")

            duplicates = list(
                set([line for line in pairs if pairs.count(line) > 1]))
            if duplicates:
                logger.warning(
                    f"⚠️ Duplicate pair(s) detected: {', '.join(duplicates)} — generation stopped.")
                return safe_processed(p, [], sd, sd, 0.0, pos_prompt, neg_prompt, "⚠️ Duplicate pairs detected", "")

            unique_pairs = list(dict.fromkeys(pairs))
            result_images = []

            for i, line in enumerate(unique_pairs, 1):
                sampler, scheduler = line.split(",", 1)
                sampler = sampler.strip()
                scheduler = scheduler.strip()

                print(
                    f"🔄[{i}/{len(unique_pairs)}] Sampler = '{sampler}', Scheduler = '{scheduler}'", flush=True)

                img = generate_or_warn(
                    p, sampler=sampler, scheduler=scheduler, font_path=font_path)

                img_annotated = annotate_batch_image(
                    img, sampler, scheduler, font_path)
                result_images.append(img_annotated)

                if save_cells:
                    filename = f"{sampler}__{scheduler}.png"
                    img.save(cells_dir / filename, "PNG")
                    logger.info(f"💾 Saved: {filename}")

            grid = create_batch_grid(result_images, width=p.width, height=p.height,
                                     padding=padding, margin_top=6, margin_side=padding)

            name = "batchgrid.png"
            counter = 1
            while (output_dir / name).exists():
                name = f"batchgrid_{counter}.png"
                counter += 1
            grid.save(output_dir / name, "PNG")

            logger.info(f"✅ Batch Grid assembled: {grid.width}×{grid.height}")
            print(
                f"✅ Batch Grid assembled: {grid.width}×{grid.height}", flush=True)

            return safe_processed(p, [grid], sd, sd, 0.0, pos_prompt, neg_prompt, "✅ Batch Grid complete", "")

        axis_x = "Sampler" if sampler_axis == "Axis X" else "Scheduler"
        axis_y = "Scheduler" if sampler_axis == "Axis X" else "Sampler"

        x_vals = xy_samplers if axis_x == "Sampler" else xy_schedulers
        y_vals = xy_schedulers if axis_y == "Scheduler" else xy_samplers

        if not x_vals:
            logger.warning("⚠️ No Sampler(s) selected for Axis X or Y.")
            return safe_processed(p, [], sd, sd, 0.0, pos_prompt, neg_prompt, "⚠️ No Sampler(s) selected", "")
        if not y_vals:
            logger.warning("⚠️ No Scheduler(s) selected for Axis X or Y.")
            return safe_processed(p, [], sd, sd, 0.0, pos_prompt, neg_prompt, "⚠️ No Scheduler(s) selected", "")
        if len(x_vals) * len(y_vals) > 50:
            return safe_processed(p, [], sd, sd, 0.0, pos_prompt, neg_prompt, "⚠️ Invalid or large grid", "")

        cells = []
        total = len(x_vals) * len(y_vals)
        i = 1

        for yv in y_vals:
            for xv in x_vals:
                samp = xv if axis_x == "Sampler" else yv
                sched = yv if axis_y == "Scheduler" else xv

                print(
                    f"🔄[{i}/{total}] Sampler = '{samp}', Scheduler = '{sched}'", flush=True)
                logger.info(
                    f"🔄[{i}/{total}] Sampler = '{samp}', Scheduler = '{sched}'")

                img = generate_or_warn(
                    p, sampler=samp, scheduler=sched, font_path=font_path)

                cells.append(img)
                i += 1

        font_test = ImageFont.truetype(
            str(font_path), 42) if font_path else ImageFont.load_default()
        line_h = sum(font_test.getmetrics())
        dummy_draw = ImageDraw.Draw(Image.new("RGB", (10, 10)))
        max_label_lines = max(len(wrap_text_to_fit(
            dummy_draw, label, font_path, p.width)[0]) for label in x_vals)
        x_label_h = 10 + line_h * max_label_lines + 6

        cols = len(x_vals)
        rows = len(y_vals)
        y_label_w = 240

        grid_w = cols * p.width + (cols - 1) * padding
        grid_h = rows * p.height + (rows - 1) * padding
        full_w = grid_w + y_label_w
        full_h = grid_h + x_label_h

        grid = Image.new("RGB", (full_w, full_h), (255, 255, 255))
        draw = ImageDraw.Draw(grid)

        if show_labels:
            for ix, label in enumerate(x_vals):
                x = y_label_w + ix * (p.width + padding)
                box_w = p.width
                lines, font = wrap_text_to_fit(draw, label, font_path, box_w)
                line_h = sum(font.getmetrics())
                y_text = 10
                for line in lines:
                    text_w = font.getbbox(line)[2]
                    x_text = x + (box_w - text_w) // 2
                    draw.text((x_text, y_text), line,
                              font=font, fill=(0, 0, 0))
                    y_text += line_h

            for iy, label in enumerate(y_vals):
                y = x_label_h + iy * (p.height + padding)
                box_w = y_label_w

                lines, font = wrap_text_to_fit(draw, label, font_path, box_w)
                line_h = sum(font.getmetrics())
                total_text_h = line_h * len(lines)

                y_text = y + ((p.height - total_text_h) // 2)

                x = 0
                for line in lines:
                    text_w = font.getbbox(line)[2]
                    x_text = x + (box_w - text_w) // 2
                    draw.text((x_text, y_text), line,
                              font=font, fill=(0, 0, 0))
                    y_text += line_h

        for idx, img in enumerate(cells):
            r, c = divmod(idx, cols)
            x = y_label_w + c * (p.width + padding)
            y = x_label_h + r * (p.height + padding)
            grid.paste(img, (x, y))

            if save_cells:
                filename = f"{x_vals[c]}__{y_vals[r]}.png"
                img.save(cells_dir / filename, "PNG")
                logger.info(f"💾 XY Cell saved: {filename}")

        for fmt in save_formats:
            ext = fmt.lower()
            out = grid
            if out.width > LIMIT or out.height > LIMIT:
                if auto_downscale:
                    scale = min(LIMIT / out.width, LIMIT / out.height)
                    out = out.resize(
                        (int(out.width * scale), int(out.height * scale)), Image.LANCZOS)
                    logger.warning(
                        f"🪄 Downscaling XY grid → {out.width}×{out.height}")
                else:
                    continue
            out.save(
                output_dir / f"xygrid.{ext}", fmt.upper(), quality=95 if ext == "webp" else None)

        logger.info(f"✅ XY Grid saved: {grid.width}×{grid.height}")
        print(f"✅ XY Grid saved: {grid.width}×{grid.height}", flush=True)

        return safe_processed(p, [grid], sd, sd, 0.0, pos_prompt, neg_prompt, "✅ XY Grid complete", "")
