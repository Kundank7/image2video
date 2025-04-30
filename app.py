import subprocess
import threading
import gradio as gr

# Track whether dependencies are installed
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

def generate_video(image_input):
    import torch
    from diffusers import StableVideoDiffusionPipeline
    from diffusers.utils import load_image, export_to_video

    # Load image from URL or upload
    if isinstance(image_input, str):
        image = load_image(image_input)
    else:
        image = load_image(image_input.name)

    pipe = StableVideoDiffusionPipeline.from_pretrained(
        "stabilityai/stable-video-diffusion-img2vid-xt",
        torch_dtype=torch.float16,
        variant="fp16"
    )
    pipe.enable_model_cpu_offload()

    generator = torch.manual_seed(42)
    frames = pipe(image, decode_chunk_size=8, generator=generator).frames[0]

    video_path = "generated.mp4"
    export_to_video(frames, video_path, fps=7)
    return video_path

# Gradio UI
with gr.Blocks() as demo:
    status = gr.Markdown("")
    install_btn = gr.Button("Install & Setup")
    start_btn = gr.Button("Start Generation", visible=False)

    image_url = gr.Textbox(label="Paste Image URL (or leave blank to upload)")
    image_upload = gr.File(label="Or Upload Image")

    output_video = gr.Video(label="Generated Video")

    install_btn.click(
        run_installation,
        outputs=[status, install_btn, start_btn]
    )

    def handle_input(image_url_val, image_file_val):
        return image_url_val if image_url_val else image_file_val

    start_btn.click(
        fn=handle_input,
        inputs=[image_url, image_upload],
        outputs=gr.State()
    ).then(
        fn=generate_video,
        inputs=[gr.State()],
        outputs=output_video
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=8080)
