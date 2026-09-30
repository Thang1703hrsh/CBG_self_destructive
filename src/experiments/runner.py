"""Single-device native attack runner, with optimizer/RNG/data-order resume."""
from collections import OrderedDict
import copy
import json
import math
from pathlib import Path
import random
import time

import torch

from src.cbg.spec import SpecConfig
from src.data_processing.cbg_datasets import (CbgBenignDataset,CbgPairedSafetyDataset,
    SAFETY_PROMPT_TEMPLATE,tokenize_prompt_response,_collate_sequences,collate_cbg_safety_batch)
from src.experiments.attacks import attack_gradient,rescale_swiglu
from src.experiments.protocol import digest,file_digest,number,positive_int,read_json,validate_manifest,write_json,write_once


class BatchStream:
    """Independent, resumable local RNG, with shuffled complete epochs."""
    def __init__(self,dataset,seed):
        if not len(dataset):
            raise ValueError("empty training pool")
        self.dataset=dataset
        self.rng=random.Random(seed)
        self.order=list(range(len(dataset)))
        self.rng.shuffle(self.order)
        self.cursor=0

    def take(self,count):
        positive_int(count,"batch size")
        out=[]
        while len(out)<count:
            if self.cursor==len(self.order):
                self.rng.shuffle(self.order)
                self.cursor=0
            out.append(self.dataset[self.order[self.cursor]])
            self.cursor+=1
        return out

    def state_dict(self):
        return dict(order=self.order,cursor=self.cursor,rng=self.rng.getstate())

    def load_state_dict(self,state):
        if sorted(state["order"])!=list(range(len(self.dataset))):
            raise ValueError("resumed data pool differs")
        self.order=state["order"]
        self.cursor=state["cursor"]
        self.rng.setstate(state["rng"])


class HarmfulDataset:
    def __init__(self,rows,tokenizer,length):
        self.rows,self.tokenizer,self.length=rows,tokenizer,length

    def __len__(self):
        return len(self.rows)

    def __getitem__(self,index):
        row=self.rows[index]
        return tokenize_prompt_response(self.tokenizer,SAFETY_PROMPT_TEMPLATE.format(prompt=row["prompt"]),
                                        row["response"],max_length=self.length)


def move_batch(batch,device):
    if isinstance(batch,dict):
        return {k:move_batch(v,device) for k,v in batch.items()}
    return batch.to(device) if isinstance(batch,torch.Tensor) else batch


def load_model(checkpoint,device,tokenizer_name=None):
    from transformers import AutoModelForCausalLM,AutoTokenizer
    path=Path(checkpoint)
    adapter=path/"adapter_config.json"
    if adapter.exists():
        from peft import PeftModel
        base=read_json(adapter)["base_model_name_or_path"]
        model=AutoModelForCausalLM.from_pretrained(base,torch_dtype=torch.float32,attn_implementation="eager")
        model=PeftModel.from_pretrained(model,checkpoint).merge_and_unload()
    else:
        model=AutoModelForCausalLM.from_pretrained(checkpoint,torch_dtype=torch.float32,attn_implementation="eager")
    tokenizer=AutoTokenizer.from_pretrained(tokenizer_name or checkpoint)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token=tokenizer.eos_token
    model.config.use_cache=False
    return model.to(device),tokenizer


def checkpoint_identity(checkpoint,model):
    path=Path(checkpoint)
    if path.is_dir():
        files=sorted(p for p in path.iterdir() if p.is_file() and (
            p.suffix==".safetensors" or p.name.startswith("pytorch_model") or p.name in {"config.json","adapter_config.json"}))
        if not files:
            raise ValueError("local checkpoint has no identifiable model files")
        return dict(path=str(path.resolve()),files={p.name:file_digest(p) for p in files})
    return dict(model=checkpoint,resolved_revision=getattr(model.config,"_commit_hash",None))


def attack_datasets(manifest,tokenizer,recipe,pool):
    loaded=validate_manifest(manifest)
    if pool not in {"validation","attack"}:
        raise ValueError("attacks must explicitly choose validation or attack pool")
    count=positive_int(recipe["samples"],"attack samples")
    indices=manifest[pool+"_indices"]
    if count>len(indices):
        raise ValueError("requested attack exceeds the frozen pool")
    rows=[loaded["safety"][i] for i in indices[:count]]
    benign_rows=[loaded["benign"][i] for i in manifest["benign_indices"]]
    if recipe.get("restoration"):
        # S13 requests 10K distinct samples, not repeated epochs over the 4K
        # defense benign subset. Source bytes are still pinned by the manifest.
        benign_rows=[r for r in loaded["benign"] if str(r.get("output","")).strip()]
        random.Random(recipe["seed"]).shuffle(benign_rows)
    harmful=HarmfulDataset(rows,tokenizer,recipe["context_length"])
    benign=CbgBenignDataset(benign_rows,tokenizer,max_length=recipe["benign_length"])
    if recipe["family"]=="mixing":
        p=number(recipe["harmful_fraction"],"harmful_fraction")
        if p>1:
            raise ValueError("harmful_fraction must be <=1")
        n_h=round(count*p)
        if abs(n_h-count*p)>1e-8:
            raise ValueError("mixture fraction must yield an integer example count")
        if count-n_h>len(benign_rows):
            raise ValueError("not enough distinct benign records for the mixture")
        # Use identical sequence length so the mixed rows collate correctly.
        mix_b=CbgBenignDataset(benign_rows[:count-n_h],tokenizer,max_length=recipe["context_length"])
        harmful=torch.utils.data.ConcatDataset([HarmfulDataset(rows[:n_h],tokenizer,recipe["context_length"]),mix_b])
    paired=None
    if recipe["family"]=="reverse_cbg":
        if any(not str(r.get("refusal","")).strip() for r in rows):
            raise ValueError("reverse-CBG attack pool needs independently provided refusal pairs")
        paired=CbgPairedSafetyDataset(rows,tokenizer,max_length=recipe["context_length"])
    return harmful,benign,paired


def train_attack(model,tokenizer,datasets,recipe,output_dir,*,reference=None,spec_config=None,
                 resume=None,identity=None):
    """Run only when called explicitly. No cloud jobs or automatic sweeps."""
    output=Path(output_dir)
    output.mkdir(parents=True,exist_ok=True)
    number(recipe["max_grad_norm"],"max_grad_norm")
    steps=positive_int(recipe["steps"],"steps")
    positive_int(recipe["batch_size"],"batch_size")
    number(recipe["learning_rate"],"learning_rate",minimum=1e-30)
    number(recipe["weight_decay"],"weight_decay")
    number(recipe["warmup_ratio"],"warmup_ratio")
    if recipe["warmup_ratio"]>=1:
        raise ValueError("warmup_ratio must be <1")
    identity=dict(recipe=recipe,**(identity or {}))
    write_once(output/"run_identity.json",identity)
    if (output/"metrics.json").exists() and not resume:
        raise ValueError("run already completed; choose a new output directory")
    if (output/"train_log.jsonl").exists() and not resume:
        raise ValueError("partial run exists; use an explicit checkpoint resume")
    device=next(model.parameters()).device
    generator=torch.Generator(device=device).manual_seed(recipe["seed"])
    streams=[BatchStream(ds,recipe["seed"]+i) if ds is not None else None for i,ds in enumerate(datasets)]
    trainable=[p for p in model.parameters() if p.requires_grad]
    if recipe["optimizer"] not in {"adamw","sgd"}:
        raise ValueError("optimizer must be adamw or sgd")
    optimizer=(torch.optim.AdamW if recipe["optimizer"]=="adamw" else torch.optim.SGD)(
        trainable,lr=recipe["learning_rate"],weight_decay=recipe["weight_decay"])
    warmup=int(steps*recipe["warmup_ratio"])
    schedule_name=recipe.get("scheduler","cosine")
    if schedule_name not in {"cosine","linear","constant"}:
        raise ValueError("scheduler must be cosine, linear or constant")
    def schedule(step):
        if step<warmup:
            return step/max(1,warmup)
        progress=min(1.,(step-warmup)/max(1,steps-warmup))
        return 1. if schedule_name=="constant" else 1.-progress if schedule_name=="linear" else .5*(1+math.cos(math.pi*progress))
    scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,schedule)
    start=0
    if resume:
        state=torch.load(Path(resume)/"training_state.pt",map_location=device,weights_only=True)
        if state["identity_sha256"]!=digest(identity):
            raise ValueError("resume configuration/checkpoint/data identity differs")
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        generator.set_state(state["generator"].cpu())
        for stream,saved in zip(streams,state["streams"]):
            if stream is not None:
                stream.load_state_dict(saved)
        torch.set_rng_state(state["torch_rng"].cpu())
        if device.type=="cuda":
            torch.cuda.set_rng_state_all([s.cpu() for s in state["cuda_rng"]])
        start=state["step"]
        # Remove only duplicate log records from the resumed tail, retaining
        # the prefix actually represented by the restored optimizer state.
        log=output/"train_log.jsonl"
        if log.exists():
            rows=[json.loads(line) for line in log.read_text(encoding="utf-8").splitlines()]
            log.write_text("".join(json.dumps(r)+"\n" for r in rows if r["step"]<=start),encoding="utf-8")
    def save(step,folder):
        folder.mkdir(parents=True,exist_ok=True)
        model.save_pretrained(folder)
        tokenizer.save_pretrained(folder)
        torch.save(dict(step=step,identity_sha256=digest(identity),model=model.state_dict(),
                        optimizer=optimizer.state_dict(),scheduler=scheduler.state_dict(),generator=generator.get_state(),
                        streams=[s.state_dict() if s else None for s in streams],torch_rng=torch.get_rng_state(),
                        cuda_rng=torch.cuda.get_rng_state_all() if device.type=="cuda" else []),folder/"training_state.pt")
    wall=time.perf_counter()
    for step in range(start,steps):
        if device.type=="cuda":
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)
        began=time.perf_counter()
        if recipe["family"]=="reverse_cbg":
            if spec_config is None:
                raise ValueError("reverse-CBG config required")
            batch=collate_cbg_safety_batch(streams[2].take(recipe["geometry_m"]))
            batch["objective_harmful"]=_collate_sequences(streams[0].take(recipe["batch_size"]))
        else:
            batch=dict(harmful=_collate_sequences(streams[0].take(recipe["batch_size"])))
        if recipe["family"] in {"kl","benign_sft","projected","reverse_cbg"}:
            batch["benign"]=_collate_sequences(streams[1].take(recipe["benign_batch_size"]))
        gradient,metrics=attack_gradient(model,move_batch(batch,device),recipe,reference=reference,
                                        spec_config=spec_config,generator=generator)
        optimizer.zero_grad(set_to_none=True)
        for name,p in model.named_parameters():
            if p.requires_grad:
                p.grad=gradient[name]
        norm=torch.nn.utils.clip_grad_norm_(trainable,recipe["max_grad_norm"] or float("inf"),error_if_nonfinite=True)
        lr=optimizer.param_groups[0]["lr"]
        optimizer.step()
        scheduler.step()
        if device.type=="cuda":
            torch.cuda.synchronize(device)
        metrics.update(step=step+1,gradient_norm=float(norm),learning_rate=lr,step_time=time.perf_counter()-began,
                       peak_memory=torch.cuda.max_memory_allocated(device) if device.type=="cuda" else 0)
        with (output/"train_log.jsonl").open("a",encoding="utf-8") as stream:
            stream.write(json.dumps(metrics,allow_nan=False)+"\n")
        if recipe["save_steps"] and (step+1)%recipe["save_steps"]==0:
            save(step+1,output/"trajectory"/f"checkpoint-{step+1}")
    save(steps,output/"final")
    from src.experiments.artifacts import hardware
    result=dict(status="complete",steps=steps,resumed_from_step=start,wall_seconds_this_segment=time.perf_counter()-wall,
                hardware=hardware(),identity_sha256=digest(identity))
    write_json(output/"metrics.json",result)
    return result


def run_attack(checkpoint,manifest_path,recipe,output_dir,*,pool,device,geometry_config=None,resume=None):
    from transformers import set_seed
    set_seed(recipe["seed"])
    manifest=read_json(manifest_path)
    validate_manifest(manifest)
    if recipe["family"] not in {"sft","mixing","kl","benign_sft","projected","reverse_cbg"}:
        raise ValueError("this recipe requires its original external adapter")
    model,tokenizer=load_model(checkpoint,device)
    source_identity=checkpoint_identity(checkpoint,model)
    identity=dict(checkpoint=checkpoint,source_identity=source_identity,
                  manifest_sha256=manifest["sha256"],pool=pool,geometry_config=geometry_config)
    write_once(Path(output_dir)/"run_identity.json",dict(recipe=recipe,**identity))
    reference=copy.deepcopy(model).requires_grad_(False).eval() if recipe["family"]=="kl" else None
    if recipe.get("rescale_sigma") is not None:
        factors=rescale_swiglu(model,sigma=recipe["rescale_sigma"],
            generator=torch.Generator(device=device).manual_seed(recipe["seed"]))
        Path(output_dir).mkdir(parents=True,exist_ok=True)
        torch.save([v.cpu() for v in factors],Path(output_dir)/"rescale_factors.pt")
    if recipe["method"]=="lora":
        from peft import LoraConfig,TaskType,get_peft_model
        model=get_peft_model(model,LoraConfig(r=recipe["lora_rank"],lora_alpha=recipe["lora_alpha"],
            target_modules=recipe["lora_targets"],lora_dropout=0.,bias="none",task_type=TaskType.CAUSAL_LM))
    elif recipe["method"]!="full":
        raise ValueError("method must be full or lora")
    if recipe.get("freeze_layers"):
        blocks=model.get_submodule("model.layers")
        if max(recipe["freeze_layers"])>=len(blocks):
            raise ValueError("layer-freezing recipe is incompatible with this architecture")
        for index in recipe["freeze_layers"]:
            blocks[index].requires_grad_(False)
    config=SpecConfig(**geometry_config) if geometry_config else None
    return train_attack(model,tokenizer,attack_datasets(manifest,tokenizer,recipe,pool),recipe,output_dir,
        reference=reference,spec_config=config,resume=resume,
        identity=identity)
