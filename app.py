import os
import subprocess
import threading
import gradio as gr

# Flag to indicate if dependencies are installed
installed = False

def install_dependencies():
    global installed
    if not installed:
        subprocess.run(["pip", "install", "-q", "-U", "diffusers", "transformers", "accelerate", "torch", "gradio"])
        installed = True

def run_installation():
    thread = threading.Thread(target=install_dependencies)
    thread.start()
    return "Installing dependencies... Please wait.", gr.update(visible=False), gr.update(visible=True)

def generate_video(image_url_or_file):
    import torch
    from diffusers import StableVideoDiffusionPipeline
    from diffusers.utils import load_image, export_to_video

    # Load image from upload or URL
    if isinstance(image_url_or_file, str):
        image = load_image(image_url_or_file)
    else:
        image = load_image(image_url_or_file.name)

    # Load model
    pipe = StableVideoDiffusionPipeline.from_pretrained(
        "stabilityai/stable-video-diffusion-img2vid-xt",
        torch_dtype=torch.float16,
        variant="fp16"
    )
    pipe.enable_model_cpu_offload()

    # Generate video
    generator = torch.manual_seed(42)
    frames = pipe(image, decode_chunk_size=8, generator=generator).frames[0]

    video_path = "generated.mp4"
    export_to_video(frames, video_path, fps=7)
    return video_path

with gr.Blocks() as demo:
    install_btn = gr.Button("Install & Start Setup")
    status = gr.Markdown("")
    start_btn = gr.Button("Start", visible=False)
    
    with gr.Row():
        image_input = gr.Textbox(label="Paste Image URL (or upload below)")
        image_upload = gr.File(label="Or Upload Image")

    output_video = gr.Video(label="Generated Video")

    install_btn.click(run_installation, outputs=[status, install_btn, start_btn])
    
    def get_image(image_url, image_file):
        return image_url if image_url else image_file

    start_btn.click(fn=get_image, inputs=[image_input, image_upload], outputs=None).then(
        fn=generate_video,
        inputs=[gr.State(lambda: image_input.value if image_input.value else image_upload.value)],
        outputs=output_video
    )

demo.launch()
