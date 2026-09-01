import io
p = '/home/guests/zhen/sglang-omni-tts/code/src/moss_tts_realtime/finetuning/sft.py'
s = io.open(p, encoding='utf-8').read()

a1 = '    parser.add_argument("--checkpointing-steps", type=int, default=0)'
assert s.count(a1) == 1
s = s.replace(a1, a1 + '\n    parser.add_argument("--lora-rank", type=int, default=0,\n'
              '                        help=">0 trains a LoRA adapter (frozen base) and saves merged weights")', 1)

a2 = '''    model.language_model.embed_tokens.weight.requires_grad = False'''
assert s.count(a2) == 1
new2 = '''    model.language_model.embed_tokens.weight.requires_grad = False

    if args.lora_rank > 0:
        from peft import LoraConfig, get_peft_model

        lora_config = LoraConfig(
            r=args.lora_rank,
            lora_alpha=args.lora_rank * 2,
            lora_dropout=0.05,
            bias="none",
            target_modules=r".*\\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)$",
        )
        model = get_peft_model(model, lora_config)
        if accelerator.is_main_process:
            model.print_trainable_parameters()'''
s = s.replace(a2, new2, 1)

a3 = '''    state_dict = accelerator.get_state_dict(model)
    unwrapped_model = accelerator.unwrap_model(model)
'''
assert s.count(a3) == 1
new3 = '''    unwrapped_model = accelerator.unwrap_model(model)
    if hasattr(unwrapped_model, "merge_and_unload"):
        # note (luojiaxuan): LoRA 训练时存 merge 后的全量权重，下游加载零改动。
        # merge 会摘掉 adapter，故 LoRA 模式下只允许 epoch 末保存一次
        # （--checkpointing-steps 0），不支持中途 checkpoint 后继续训练。
        unwrapped_model = unwrapped_model.merge_and_unload()
        state_dict = unwrapped_model.state_dict()
    else:
        state_dict = accelerator.get_state_dict(model)
'''
s = s.replace(a3, new3, 1)
io.open(p, 'w', encoding='utf-8').write(s)
import py_compile
py_compile.compile(p, doraise=True)
print('sft.py LoRA patch ok')
import io
p = '/home/guests/zhen/sglang-omni-tts/code/src/moss_tts_realtime/finetuning/sft.py'
s = io.open(p, encoding='utf-8').read()
old = '''    unwrapped_model = accelerator.unwrap_model(model)
    if hasattr(unwrapped_model, "merge_and_unload"):
        # note (luojiaxuan): LoRA 训练时存 merge 后的全量权重，下游加载零改动。
        # merge 会摘掉 adapter，故 LoRA 模式下只允许 epoch 末保存一次
        # （--checkpointing-steps 0），不支持中途 checkpoint 后继续训练。
        unwrapped_model = unwrapped_model.merge_and_unload()
        state_dict = unwrapped_model.state_dict()
    else:
        state_dict = accelerator.get_state_dict(model)
'''
new = '''    unwrapped_model = accelerator.unwrap_model(model)
    if hasattr(unwrapped_model, "peft_config"):
        # note (luojiaxuan): LoRA 模式只存 adapter（几十 MB，rank0 写盘），
        # 全量权重由训练后的单进程离线 merge 产出——训练内 merge 在多 rank
        # DDP 下会死锁（实测挂死于集体同步）。
        if accelerator.is_main_process:
            unwrapped_model.save_pretrained(str(output_dir / "adapter"))
            copy_support_files(output_dir)
            copy_inference_assets(model_path, output_dir)
            with open(output_dir / "finetune_args.json", "w", encoding="utf-8") as f:
                json.dump(train_args, f, indent=2, ensure_ascii=False)
        accelerator.wait_for_everyone()
        return
    state_dict = accelerator.get_state_dict(model)
'''
assert s.count(old) == 1
io.open(p, 'w', encoding='utf-8').write(s.replace(old, new, 1))
import py_compile
py_compile.compile(p, doraise=True)
print('two-stage patch ok')
