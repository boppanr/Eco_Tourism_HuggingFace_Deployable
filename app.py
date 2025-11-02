import os
import gradio as gr
from pipeline import run_all, SEED
import subprocess, sys

def run_pipeline_and_capture():
    try:
        proc = subprocess.Popen([sys.executable, 'pipeline.py'], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        out, _ = proc.communicate(timeout=300)
        return out[:20000] if out else 'No output'
    except Exception as e:
        return f'Error running pipeline: {e}'

def run_and_return(seed=SEED):
    os.environ['REVIEW_GENIE_SEED'] = str(int(seed))
    txt = run_pipeline_and_capture()
    return txt

with gr.Blocks() as demo:
    gr.Markdown('# Review Genie — Eco Tourism demo (auto-generated)')
    seed = gr.Number(value=int(os.environ.get('REVIEW_GENIE_SEED', 42)), label='Deterministic seed')
    btn = gr.Button('Run pipeline (deterministic)')
    out = gr.Textbox(label='Pipeline output (truncated)', lines=20)
    btn.click(run_and_return, inputs=[seed], outputs=[out])
    gr.Markdown('Auto-generated deployable. Edit pipeline.py for production.')

if __name__ == '__main__':
    demo.launch(server_name='0.0.0.0', server_port=int(os.environ.get('PORT', 7860)))
