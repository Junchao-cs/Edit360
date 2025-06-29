import gradio as gr
import torch
import os
import subprocess
from pathlib import Path
import tempfile
import shutil

def generate_video(gpu_device, mode, input_path_f, input_path_b, output_path_mp4, output_path_img, path_b_num, seed):
    try:
        os.makedirs(output_path_mp4, exist_ok=True)
        os.makedirs(output_path_img, exist_ok=True)
        
        cmd_args = [
            "python", "scripts/sampling/simple_video_sample.py",
            "--version", "sv3d_u",
            "--output_folder_mp4", str(output_path_mp4),
            "--output_folder_img", str(output_path_img),
            "--seed", str(seed)
        ]

        if mode == "one":
            cmd_args.extend(["--mode", "one", "--input_path", str(input_path_f)])
        else:
            cmd_args.extend([
                "--mode", mode,
                "--input_path_f", str(input_path_f),
                "--input_path_b", str(input_path_b),
                "--path_b_num", str(path_b_num)
            ])

        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_device)

        print(f"Executing command (list): {cmd_args}")

        process = subprocess.run(
            cmd_args,
            env=env,
            shell=False,
            check=True,
            capture_output=True,
            text=True
        )

        if process.stdout:
            print("Command output:", process.stdout)

        output_videos = sorted(Path(output_path_mp4).glob("*.mp4"), key=lambda x: int(x.stem))
        if not output_videos:
            raise gr.Error("No MP4 file was generated in the output folder.")
        
        output_video = output_videos[-1]
        temp_video = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name
        shutil.copy2(str(output_video), temp_video)

        image_files = sorted(Path(output_path_img).glob("*.png"))
        temp_images = []
        for img_file in image_files:
            temp_img = tempfile.NamedTemporaryFile(delete=False, suffix=".png").name
            shutil.copy2(str(img_file), temp_img)
            temp_images.append(temp_img)

        return temp_video, temp_images, temp_video, temp_images

    except subprocess.CalledProcessError as e:
        print("Command failed with error:", e)
        print("Error output:", e.stderr)
        print("Command output:", e.stdout)
        raise gr.Error(f"Command execution failed: {e.stderr}")
    except Exception as e:
        print("Unexpected error:", str(e))
        raise gr.Error(f"Unexpected error: {str(e)}")

def update_interface(mode):
    if mode == "one":
        return [
            gr.update(visible=True),   # input_f visible
            gr.update(visible=False),  # input_b hidden
            gr.update(visible=False),  # path_b_num hidden
        ]
    else:
        return [
            gr.update(visible=True),   # input_f
            gr.update(visible=True),   # input_b
            gr.update(visible=True),   # path_b_num
        ]

with gr.Blocks() as demo:
    gr.Markdown("# Video Generation Demo")
    
    with gr.Row():
        with gr.Column():
            gpu_device = gr.Dropdown(
                choices=["0", "1", "2", "3", "4", "5", "6", "7"], 
                value="0",
                label="GPU Device"
            )
            mode = gr.Dropdown(
                choices=["one", "two"],
                value="two",
                label="Mode"
            )
            input_f = gr.Image(label="Input Image F", type="filepath")
            input_b = gr.Image(label="Input Image B", type="filepath")
            
            output_mp4 = gr.Textbox(label="Output MP4 Folder", value="outputs/mp4")
            output_img = gr.Textbox(label="Output Image Folder", value="outputs/img")
            path_b_num = gr.Slider(minimum=1, maximum=20, value=11, step=1, label="Path B Num")
            seed = gr.Slider(minimum=-1, maximum=100, value=21, step=1, label="Seed")

            generate_btn = gr.Button("Generate")

        with gr.Column():
            video_output = gr.Video(label="Generated Video")
            gallery = gr.Gallery(label="Generated Images")
            
            with gr.Row():
                download_video = gr.File(label="Download Video")
                download_images = gr.File(label="Download Images", file_count="multiple")
    
    mode.change(
        fn=update_interface,
        inputs=[mode],
        outputs=[input_f, input_b, path_b_num]
    )
    
    generate_btn.click(
        fn=generate_video,
        inputs=[
            gpu_device,
            mode,
            input_f,
            input_b, 
            output_mp4,
            output_img,
            path_b_num,
            seed
        ],
        outputs=[
            video_output, 
            gallery, 
            download_video, 
            download_images
        ]
    )


    gr.Examples(
        examples=[
            [
                "0", "one", 
                "assets/demo_example/superman_f.png",
                None,
                "outputs/mp4",
                "outputs/img",
                None, 23
            ],
            [
                "0", "two", 
                "assets/demo_example/superman_f_edited.png",
                "assets/demo_example/superman_b_edited.png",
                "outputs/mp4",
                "outputs/img",
                11, 23
            ],
            [
                "0", "one", 
                "assets/demo_example/rabbit_f.png",
                None, 
                "outputs/mp4",
                "outputs/img",
                None, 23
            ],
            [
                "0", "two", 
                "assets/demo_example/rabbit_f_edited.png",
                "assets/demo_example/rabbit_b_edited.png",
                "outputs/mp4",
                "outputs/img",
                11, 23
            ],
            [
                "0", "one", 
                "assets/demo_example/girl_f.png",
                None, 
                "outputs/mp4",
                "outputs/img",
                None, 23
            ],
            [
                "0", "two", 
                "assets/demo_example/girl_f.png",
                "assets/demo_example/girl_s_edited1.png",
                "outputs/mp4",
                "outputs/img",
                5, 23
            ],
            [
                "0", "two", 
                "assets/demo_example/girl_f.png",
                "assets/demo_example/girl_s_edited2.png",
                "outputs/mp4",
                "outputs/img",
                5, 23
            ],
            [
                "0", "one", 
                "assets/demo_example/boy_f.png",
                None,
                "outputs/mp4",
                "outputs/img",
                None, 23
            ],
            [
                "0", "two", 
                "assets/demo_example/boy_f_edited.png",
                "assets/demo_example/boy_b_edited.png",
                "outputs/mp4",
                "outputs/img",
                11, 23
            ]
        ],
        inputs=[
            gpu_device, mode, input_f, input_b,
            output_mp4, output_img, path_b_num, seed
        ]
    )

demo.launch(
    server_name="0.0.0.0",
    share=True,
    auth=("1", "1") 
)
