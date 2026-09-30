"""Declarative attack catalog and defense search, without starting GPU work."""
import copy
import itertools
from pathlib import Path

import yaml

from src.experiments.protocol import digest, number, positive_int


def load_protocol(path):
    with Path(path).open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict) or config.get("schema_version") != 1:
        raise ValueError("unsupported scenario protocol")
    return config


def unresolved(value, prefix=""):
    result = []
    if value is None:
        return [prefix]
    if isinstance(value, dict):
        for key, child in value.items():
            result.extend(unresolved(child, f"{prefix}.{key}".strip(".")))
    return result


def standard_attacks():
    rows = [(1000,"full","adamw",lr) for lr in (2e-5,5e-5,8e-5,1e-4,2e-4)]
    rows += [(10000,method,opt,lr) for method,opt in (("full","adamw"),("lora","adamw"),("full","sgd"))
             for lr in (5e-5,1e-4)]
    return [dict(id=str(i+1), scenario="S3.1", family="sft", samples=n, method=method,
                 optimizer=opt, learning_rate=lr, steps=250 if n==1000 else 25000,
                 batch_size=4, warmup_ratio=.1, weight_decay=.01, scheduler="cosine",
                 lora_rank=8, lora_alpha=16, lora_targets=["q_proj","v_proj"])
            for i,(n,method,opt,lr) in enumerate(rows)]


def attack_catalog(protocol):
    """All declared S3--S5 recipes; None decisions remain visibly unresolved."""
    base = standard_attacks()[1]
    attacks = standard_attacks()
    options = protocol["attacks"]
    def add(scenario, key, **overrides):
        attacks.append(dict(base, scenario=scenario, id=key, **overrides))
    for lr,steps in ((5e-6,2500),(1e-6,2500),(5e-6,12500),(1e-6,12500),(1e-6,50000),(5e-6,50000)):
        add("S3.2", f"lr{lr}-steps{steps}", learning_rate=lr, steps=steps,
            samples=options.get("long_attack_samples"))
    for row in standard_attacks()[:5]:
        add("S3.3", row["id"], learning_rate=row["learning_rate"], freeze_layers=list(range(10,21)))
    targets = ["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"]
    for lr in (5e-5,1e-4):
        add("S4", f"all-lr{lr}", method="lora", lora_targets=targets, samples=10000,
            steps=25000, learning_rate=lr)
    for rank in options.get("lora_ranks") or []:
        if rank == 8:
            continue  # already covered by all-lr0.0001; reuse that checkpoint
        add("S4", f"rank{rank}", method="lora", lora_targets=targets, lora_rank=rank,
            lora_alpha=16, samples=10000, steps=25000, learning_rate=1e-4)
    for family,scenario,values in (("kl","S5.1",options.get("kl_lambdas")),
                                   ("benign_sft","S5.2",[0,.01,.05,.1,.2])):
        for weight in values if values is not None else [None]:
            add(scenario,f"lambda{weight}",family=family,regularization=weight)
    for count,lr in itertools.product(options.get("projected_counts") or [None],(5e-5,1e-4)):
        add("S5.3",f"n{count}-lr{lr}",family="projected",benign_batch_size=count,learning_rate=lr,
            projection_rcond=options.get("projection_rcond"))
    for beta,lr in itertools.product(options.get("reverse_cbg_betas") or [None],(5e-5,1e-4)):
        add("S5.4",f"beta{beta}-lr{lr}",family="reverse_cbg",regularization=beta,learning_rate=lr)
    for row in standard_attacks()[:5]:
        add("S5.5",row["id"],family="external",adapter="reverse_seam",learning_rate=row["learning_rate"],regularization=.01)
    for sigma,attack in itertools.product(options.get("rescale_sigmas") or [None],("2","4")):
        add("S5.6",f"sigma{sigma}-attack{attack}",rescale_sigma=sigma,
            learning_rate=standard_attacks()[int(attack)-1]["learning_rate"])
    for p in (0,.01,.05,.10,.20):
        add("S5.7",f"p{p}",family="mixing",harmful_fraction=p)
    for sigma,attack in itertools.product((0,.1,.5,1,5,10),("2","4")):
        add("S5.8",f"sigma{sigma}-attack{attack}",noise_sigma=sigma,
            noise_scale=options.get("noise_scale"),learning_rate=standard_attacks()[int(attack)-1]["learning_rate"])
    for row in attacks:
        row.setdefault("context_length", options["context_length"])
        row.setdefault("benign_length", options["benign_length"])
        row.setdefault("max_grad_norm", options["max_grad_norm"])
        row.setdefault("benign_batch_size", options["benign_batch_size"])
        row["seed"] = protocol["seeds"]["attack"]
        row["save_steps"] = protocol["geometry"]["save_steps"]
    return attacks


def defense_config(protocol, choices, *, model, seed, output_dir, pilot=False):
    """Every scientific choice is written into the resulting executable YAML."""
    fixed = protocol["defense"]
    from src.core.cbg_trainer import CbgArguments
    args = dict(method="cbg", cbg_version="spec", parameter_space="full", reference_mode="separate_model",
                geometry_solver="exact", model_name=model, scenario_manifest=protocol["manifest"],
                beavertails_path=protocol["data"]["safety"], defense_size=protocol["data"]["defense_size"],
                attack_size=protocol["data"]["attack_size"], split_seed=protocol["seeds"]["split"],
                data_shuffle=True, reference_device=fixed.get("reference_device"),
                **{key: fixed[key] for key in ("kl_direction","safety_batch_size","benign_batch_size",
                                              "spectral_modes","safety_max_length","benign_max_length")})
    args.update(choices)
    CbgArguments(**args)
    train = dict(output_dir=str(output_dir), per_device_train_batch_size=1, gradient_accumulation_steps=1,
                 max_steps=protocol["pilot"]["max_steps"] if pilot else fixed["max_steps"],
                 **{key: fixed[key] for key in ("learning_rate","warmup_ratio","weight_decay","optim","max_grad_norm")},
                 lr_scheduler_type="cosine", bf16=False, fp16=False, tf32=False, gradient_checkpointing=False,
                 remove_unused_columns=False, dataloader_num_workers=0, logging_steps=1,
                 save_steps=fixed["save_steps"], save_only_model=False, report_to="none", seed=seed, data_seed=seed)
    return dict(TrainingArguments=train, CbgArguments=args)


def search_candidates(protocol):
    s = protocol["search"]
    for key,minimum in (("beta",4),("alpha",3),("kappa_mu",3)):
        values = s[key]
        if not isinstance(values,list) or len(values)<minimum or len(set(values)) != len(values):
            raise ValueError(f"search.{key} needs >= {minimum} distinct values")
        for value in values:
            number(value,f"search.{key}",minimum=0 if key=="alpha" else 1e-30)
    for key in ("xi","relative_radius"):
        number(s[key],f"search.{key}",minimum=0 if key=="xi" else 1e-30)
    result = {}
    for beta,alpha,kappa in itertools.product(s["beta"],s["alpha"],s["kappa_mu"]):
        choice = dict(beta=beta,alpha=alpha,kappa_mu=kappa,xi=s["xi"],relative_radius=s["relative_radius"])
        result[digest(choice)[:12]] = choice
    return result


def ablations(selected, protocol):
    """Native ablations; external solver/precision/unlearning adapters are explicit."""
    result = {"i-raw_rho":dict(spec_objective="raw_rho",spectral_modes=1),
              "ii-euclidean":dict(spec_damping="euclidean"),"iv-empirical":dict(fisher="empirical"),
              "viii-drop-aln":dict(drop_aln=True),"viii-drop-bgn":dict(alpha=0.)}
    result.update({f"iii-k{k}":dict(spectral_modes=k) for k in (1,2,4,8)})
    # Fixed *maximum* benign input token budget; actual response counts are logged.
    budget=protocol["defense"]["benign_batch_size"]*protocol["defense"]["benign_max_length"]
    result.update({f"v-n{n}":dict(benign_batch_size=n,benign_max_length=budget//n) for n in (8,32,128)})
    result.update({f"x-draws{n}":dict(num_sign_draws=n) for n in (1,4)})
    xi_values=protocol.get("ablations",{}).get("xi_values")
    if xi_values is not None:
        if not 2<=len(xi_values)<=3:
            raise ValueError("xi sensitivity requires 2--3 values")
        for xi in xi_values:
            number(xi,"xi_values")
            result[f"ix-xi{xi}"]=dict(xi=xi)
    step=protocol.get("ablations",{}).get("rayleigh_step_size")
    if step is not None:
        number(step,"rayleigh_step_size",minimum=1e-30)
        result.update({f"vi-iterations{n}":dict(spec_solver="rayleigh_K",spec_rayleigh_steps=n,
                                               spec_rayleigh_step_size=step) for n in (1,3,5)})
    for precision,radius in itertools.product(("fp32","bf16"),protocol["pilot"]["radii"]):
        result[f"vii-{precision}-r{radius}"]=dict(spec_perturb_precision=precision,relative_radius=radius)
    return {name:dict(selected,**change) for name,change in result.items()}


def restoration_recipe(protocol,mixture):
    if mixture not in {"benign","harmful","mixed"}:
        raise ValueError("restoration mixture must be benign, harmful or mixed")
    recipe=dict(standard_attacks()[1],scenario="S13",id=mixture,family="mixing",restoration=True,
                samples=10000,batch_size=8,steps=50*10000//8,learning_rate=5e-5,
                harmful_fraction={"benign":0.,"harmful":1.,"mixed":.5}[mixture],
                context_length=protocol["attacks"]["context_length"],benign_length=protocol["attacks"]["benign_length"],
                seed=protocol["seeds"]["attack"],save_steps=protocol["geometry"]["save_steps"])
    recipe.update({key:protocol["restoration"][key] for key in ("warmup_ratio","weight_decay","max_grad_norm","scheduler")})
    return recipe


def scenario_inventory(protocol):
    return dict(protocol_sha256=digest(protocol), unresolved_decisions=unresolved(protocol),
                attacks=attack_catalog(protocol),
                scenario_backends={"S0":"probe + pilot training + validate-pilot", "S1":"CBG native; other defenses external",
                    "S2":"native generation + LM Eval; FS/overrefusal external", "S3":"native", "S4":"native",
                    "S5":"native except reverse-SEAM external", "S6":"original orthogonalization adapter",
                    "S7":"category manifest + native training/attacks", "S8":"fixed geometry + trajectory + bootstrap",
                    "S9":"native Gram diagnostics", "S10":"native except original SEAM-unlearning external",
                    "S11":"same pipeline, separate model selection", "S12":"native timings and hardware metadata",
                    "S13":"native restoration mixture after observed collapse"})
