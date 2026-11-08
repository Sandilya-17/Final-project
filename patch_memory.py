s = open("finetune.py").read()
s = s.replace("int(os.environ.get('BATCH','16'))", "int(os.environ.get('BATCH','8'))")
if "gradient_accumulation_steps" not in s:
    s = s.replace("per_device_eval_batch_size=32", "per_device_eval_batch_size=16, gradient_accumulation_steps=2")
if "requires_grad" not in s:
    s = s.replace("def metrics(p):", "for prm in model.base_model.embeddings.parameters():\n    prm.requires_grad = False\n\ndef metrics(p):", 1)
open("finetune.py", "w").write(s)
print("patched OK")
