import gradio as gr
import torch
from modeling_llava import LlavaForConditionalGeneration
from transformers import AutoProcessor, GenerationConfig

# load model
model_path = "MIL-UT/Asagi-14B"

model = LlavaForConditionalGeneration.from_pretrained(
    model_path,
    device_map="auto",
    torch_dtype=torch.bfloat16
)

processor = AutoProcessor.from_pretrained(model_path)
model.eval()

model_size = sum(p.numel() for p in model.parameters())
vision_model_size = sum(p.numel() for p in model.vision_tower.vision_model.parameters())
text_model_size = sum(p.numel() for p in model.language_model.parameters())
projector_size = sum(p.numel() for p in model.multi_modal_projector.parameters())
print(f"\nModel size: {model_size:,}, vision model size: {vision_model_size:,}, "
      f"text model size: {text_model_size:,}, projector size: {projector_size:,}")

@torch.inference_mode()
def inference_fn(
        image,
        prompt,
        max_len,
        temperature,
        repetition_penalty,
        num_beam,
):
    # prepare inputs
    pre_prompt = "以下は、タスクを説明する指示です。要求を適切に満たす応答を書きなさい。\n\n"
    prompt = f"{pre_prompt}### 指示:\n<image>\n{prompt}\n\n### 応答:\n"
    inputs = processor(
        text=prompt,
        images=image,
        return_tensors="pt",
        truncation=True,
        max_length=450,
    )
    inputs_text = processor.tokenizer(prompt, return_tensors="pt")
    inputs['input_ids'] = inputs_text['input_ids']
    inputs['attention_mask'] = inputs_text['attention_mask']
    # move the inputs to the same device as the model
    inputs = {name: tensor.to(model.device) for name, tensor in inputs.items()}
    for k, v in inputs.items():
        if v.dtype == torch.float32:
            inputs[k] = v.to(model.dtype)
    # remove token_type_ids from inputs
    inputs = {k: inputs[k] for k in inputs if k != "token_type_ids"}

    # generate
    generation_config = GenerationConfig(
        do_sample=True,
        num_beams=num_beam,
        max_new_tokens=max_len,
        temperature=temperature,
        repetition_penalty=repetition_penalty
    )
    generate_ids = model.generate(
        **inputs,
        generation_config=generation_config
    )
    generated_text = processor.batch_decode(
        generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]
    # do not print the prompt
    if "<image>" in prompt:
        prompt = prompt.replace("<image>", " ")
    generated_text = generated_text.replace(prompt, "")

    target = "システム: \n"
    output = target + generated_text
    return output


with gr.Blocks() as demo:
    gr.Markdown("# Asagi-14b Demo")

    # Model Info
    gr.Markdown("## Model Info")
    gr.Markdown(f"Model size: `{model_size:,}`, vision model size: `{vision_model_size:,}`, "
                f"text model size: `{text_model_size:,}`, projector size: `{projector_size:,}`")

    with gr.Row():
        with gr.Column():
            input_image = gr.Image(type="pil", label="image")
            prompt = gr.Textbox(label="prompt", value="この画像を見て、次の指示に詳細に答えてください。")
            with gr.Accordion(label="Configs", open=False):
                num_beam = gr.Slider(
                    minimum=1,
                    maximum=10,
                    value=5,
                    step=1,
                    interactive=True,
                    label="Num Beams",
                )
                max_len = gr.Slider(
                    minimum=10,
                    maximum=512,
                    value=256,
                    step=1,
                    interactive=True,
                    label="Max New Tokens",
                )

                temperature = gr.Slider(
                    minimum=0.0,
                    maximum=1.0,
                    value=0.7,
                    step=0.1,
                    interactive=True,
                    label="Temperature",
                )

                repetition_penalty = gr.Slider(
                    minimum=-1,
                    maximum=3,
                    value=1.5,
                    step=0.1,
                    interactive=True,
                    label="Repetition Penalty",
                )
            # button
            input_button = gr.Button(value="Submit")
        with gr.Column():
            output = gr.Textbox(label="Output")

    inputs = [input_image, prompt, max_len, temperature, repetition_penalty, num_beam]
    input_button.click(inference_fn, inputs=inputs, outputs=[output])
    prompt.submit(inference_fn, inputs=inputs, outputs=[output])


if __name__ == "__main__":
    demo.queue().launch(share=False, server_name="0.0.0.0")
