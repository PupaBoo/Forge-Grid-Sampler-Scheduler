# 🔬 Forge Grid: Sampler × Scheduler

> 💡 A script for generating image grids in Stable Diffusion Forge. Explore sampler and scheduler combinations, compare results, save individual cells, and achieve stable output—even with errors.


## 📝 Description

The script generates image grids in two modes: XY Grid and Batch Grid.


## ✨ Features

- 🤸‍♂️ **Flexible Modes**: XY Grid for automatic combinatorial testing and Batch Grid for custom pairs added via interface and/or manual input.
- 🔮 **Intuitive Interface**: Gradio interface with dropdowns, text fields, and buttons for easy configuration.
- 📋 **Detailed Logging**: Each pair (sampler and scheduler) generation is logged in the terminal with progress, and errors are logged in detail.
- 🪄 **Auto-Downscaling**: Grids exceeding 16,383 pixels are automatically resized without quality loss to maintain performance.
- 🎯 **Auto-Activation of save_large**: For 50+ cells, protection against errors for large grids is automatically enabled.
- 🛠️ **Customizable Output**: Text labels for sampler and scheduler with a custom font (`Barlow-SemiBold.ttf`).
- ⚠️ **Fallback Images**: Shown for generation failures with invalid pairs and replaced with fallback images with readable error messages.
- 🚰 **Duplicate Filtering**: Duplicate pairs in Batch Grid are automatically removed to prevent errors.
- 🔀 **Flexible Parameters**: Configure prompts, seed, steps (1–100), CFG (1.0–30.0), dimensions (256–2048 px), padding (0–200 px).
- 📅 **Individual Cell Saving**: Save to `/cells` via button, convenient for analysis.
- 💾 **PNG and WEBP Support**: Choose one or both formats.


## ⚙️ Installation

1. Clone the repository into your Forge `/extensions/` folder:
   git clone https://github.com/USERNAME/Forge-Grid-Sampler-Scheduler

2. Navigate to the project directory:
   cd Forge-Grid-Sampler-Scheduler

3. Install dependencies:
   pip install -r requirements.txt

4. (Optional) Place the `Barlow-SemiBold.ttf` font in the `fonts/` folder for improved label display.


## 🚀 Usage

1. Launch Forge WebUI and select `Forge Grid: Sampler × Scheduler` from the `txt2img` script dropdown.
2. Choose a mode:
   - **XY Grid**: Select samplers and schedulers for X and Y axes via dropdowns. The grid forms all possible combinations.
   - **Batch Grid**: Add sampler-scheduler pairs via dropdowns (using the “Add Pair” button) or manually in the text field (e.g., `Euler a,Automatic` per line).
3. Configure parameters:
   - Positive/negative prompt.
   - Seed (leave blank for random).
   - Steps, CFG, image dimensions, padding, save format (WEBP/PNG).
   - Enable labels and individual cell saving.
4. Click “Generate” to create the grid. Pair progress and errors are logged in the terminal. Use “Stop Grid Generation” to interrupt.


## 🛠 Requirements

- Python 3.7+
- Libraries: `gradio`, `Pillow`, `numpy`
- Stable Diffusion Forge
- (Optional) `Barlow-SemiBold.ttf` font


## ⚠️ Limitations

- Generation stops on errors (e.g., invalid pairs or empty axes in XY Grid).
- Duplicate pairs in Batch Grid cause generation to stop.


## 📜 License

Licensed under the [MIT License](LICENSE) © 2025 PupaBoo.  
Free to use with attribution—see LICENSE.


## 🙌 Acknowledgments

- Built with `Gradio`, `Pillow`, and `NumPy`.
- Developed with assistance from Microsoft Copilot, Grok 3, and ChatGPT.


## 🤝 Contributing

Ideas, bug reports, and pull requests are welcome on [GitHub](https://github.com/PupaBoo/Forge-Grid-Sampler-Scheduler).


## 📸 Examples

![Batch Grid Example](batchgrid.png)  
![XY Grid Example](xygrid.png)