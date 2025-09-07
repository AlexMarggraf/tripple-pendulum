# Simulation Setup and Execution

## 1. Create and Activate a Python Virtual Environment
Before installing the required libraries, create a new virtual environment for each simulation version.

```bash
# Create a virtual environment (replace 'env-glumpy' with your desired name)
python -m venv env-glumpy

# Activate the environment (Windows)
./env-glumpy/Scripts/activate

# Or for macOS/Linux
source env-glumpy/bin/activate
```

Repeat the process for the `moderngl` version if you want separate environments.

## 2. Install the Python Environment
For each simulation version (`glumpy` and `moderngl`), install the required Python libraries using the provided `requirements.txt` file.

```bash
pip install -r src/main/glumpy/requirements.txt
pip install -r src/main/moderngl/requirements.txt
```

## 3. Run the Simulation
Run the main file for your chosen version.

- **Glumpy version**: Produces direct output.
- **Moderngl version**: Produces a sequence of image files.

## 4. Compile Images into a Video (Moderngl Only)
The `moderngl` version outputs images that must be compiled into a video. For example you can use `ffmpeg` for this:

```bash
ffmpeg -framerate 30 -i ./frames/frame_%04d.png -c:v libx264 -pix_fmt yuv420p output.mp4
```