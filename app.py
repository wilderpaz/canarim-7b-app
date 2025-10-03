import gradio as gr
from transformers import AutoTokenizer, pipeline
import torch

model_id = "dominguesm/Canarim-7B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_id)

pipe = pipeline(
    "text-generation",
    model=model_id,
    torch_dtype=torch.float16,
    device_map="auto"
)

def make_prompt(instruction, input=None):
    if input:
        return f"""Abaixo está uma instrução que descreve uma tarefa, emparelhada com uma entrada que fornece mais contexto. Escreva uma resposta que conclua adequadamente a solicitação.
### Instruções: {instruction}
### Entrada: {input}
### Resposta:"""
    else:
        return f"""Abaixo está uma instrução que descreve uma tarefa. Escreva uma resposta que conclua adequadamente a solicitação.
### Instruções: {instruction}
### Resposta:"""

def gerar_resposta(instruction, context, temperature, max_length, num_return_sequences):
    prompt = make_prompt(instruction, context)
    sequences = pipe(
        prompt,
        do_sample=True,
        num_return_sequences=num_return_sequences,
        eos_token_id=tokenizer.eos_token_id,
        max_length=max_length,
        temperature=temperature,
        top_p=0.6,
        repetition_penalty=1.15,
        truncation=True
    )
    respostas = [seq['generated_text'] for seq in sequences]
    return "\n\n---\n\n".join(respostas)

demo = gr.Interface(
    fn=gerar_resposta,
    inputs=[
        gr.Textbox(label="Instrução"),
        gr.Textbox(label="Contexto"),
        gr.Slider(0.1, 1.5, value=0.9, label="Temperatura"),
        gr.Slider(100, 2048, value=1024, label="Máximo de tokens"),
        gr.Slider(1, 3, value=1, step=1, label="Número de respostas")
    ],
    outputs=gr.Textbox(label="Resposta(s)"),
    title="Canarim-7B Instruct",
    description="Gere respostas personalizadas com base em instruções e contexto."
)

demo.launch()
