"""Explicit local entry points for the CBG experiment protocol.

Preparing plans never executes training, a model download, or a judge request.
"""
import argparse
import copy
from dataclasses import asdict, replace
import json
from pathlib import Path
import re
import subprocess
import sys

import torch
import yaml

from src.experiments.protocol import (digest,file_digest,number,prepare_splits,read_json,
    select_configuration,validate_manifest,write_json,write_once)
from src.experiments.scenarios import (ablations,attack_catalog,defense_config,load_protocol,
    scenario_inventory,search_candidates,unresolved)


def code_fingerprint():
    root=Path(__file__).resolve().parents[1]
    paths=sorted(list(root.rglob("*.py"))+list((root.parent/"tests").glob("*.py")))
    return digest({str(p.relative_to(root.parent)):file_digest(p) for p in paths})


def load_choices(path):
    choices=yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    allowed={"alpha","beta","xi","kappa_mu","relative_radius","spec_objective","spec_damping",
             "drop_aln","fisher","num_sign_draws","detach_lambda","spectral_modes",
             "benign_batch_size","benign_max_length"}
    allowed.update({"spec_solver","spec_rayleigh_steps","spec_rayleigh_step_size","spec_perturb_precision"})
    allowed.update({"mu_mode", "capture_dtype", "capture_offload", "lazy_period", "lazy_rescale", "spec_method_revision"})
    allowed.update({"logprob_chunk_size", "log_gradient_components", "estimator"})
    if not isinstance(choices,dict) or set(choices)-allowed:
        raise ValueError("choices must contain only the scientific CBG settings, in YAML or JSON")
    return choices


def checked_lock(path,protocol,manifest):
    lock=read_json(path)
    if lock.get("sha256")!=digest({k:v for k,v in lock.items() if k!="sha256"}):
        raise ValueError("selection lock checksum mismatch")
    if lock["protocol_sha256"]!=digest(protocol) or lock["manifest_sha256"]!=manifest["sha256"]:
        raise ValueError("protocol or data changed after hyperparameters were frozen")
    return lock


def checked_gate(path,protocol):
    gate=read_json(path)
    if (gate.get("status")!="passed" or gate.get("code_sha256")!=code_fingerprint()
        or gate.get("qualification_sha256")!=qualification_fingerprint(protocol)
        or gate.get("radius")!=protocol["search"]["relative_radius"]):
        raise ValueError("S0 gate missing, failed, stale, or for a different protocol")
    for artifact in gate["artifacts"]:
        if file_digest(artifact["path"])!=artifact["sha256"]:
            raise ValueError("S0 evidence changed")
    return gate


def qualification_fingerprint(protocol):
    # Selecting r and filling the later search grid are expected AFTER a radius
    # curve. Bind S0 to its model/data/structure/criteria, not that later choice.
    structural={k:protocol["defense"][k] for k in ("safety_batch_size","benign_batch_size",
        "spectral_modes","safety_max_length","benign_max_length","kl_direction")}
    manifest=read_json(protocol["manifest"])
    return digest(dict(method_revision="cbg_v2", pilot_model=protocol["models"]["pilot"],pilot=protocol["pilot"],
                       structure=structural,manifest_sha256=manifest["sha256"],geometry=protocol["geometry"]))


def verify(output):
    root=Path(__file__).resolve().parents[2]
    completed=subprocess.run([sys.executable,"-m","unittest","discover","-s","tests","-v"],
                             cwd=root,capture_output=True,text=True)
    text=completed.stdout+completed.stderr
    path=Path(output)
    path.parent.mkdir(parents=True,exist_ok=True)
    path.with_suffix(".log").write_text(text,encoding="utf-8")
    # Read unittest's final summary, not incidental output from nested tests
    # which deliberately exercise skipped/failed verification reports.
    summaries=re.findall(r"^(?:OK(?: \([^\r\n]*\))?|FAILED \([^\r\n]*\))[ \t]*\r?$",
                         completed.stderr,re.MULTILINE)
    passed=completed.returncode==0 and bool(summaries) and summaries[-1].strip()=="OK"
    result=dict(status="passed" if passed else "failed",code_sha256=code_fingerprint(),
                log_sha256=file_digest(path.with_suffix(".log")),command="python -m unittest discover -s tests -v")
    write_json(path,result)
    if not passed:
        # Surface unittest's failure trace in Modal/container logs. Previously
        # capture_output hid the root cause behind only this generic exception.
        tail = "\n".join(text.splitlines()[-100:])
        print(f"Verification failed (exit code {completed.returncode}). Last 100 log lines:\n{tail}",
              file=sys.stderr, flush=True)
        raise ValueError(f"verification failed or skipped required integration checks; see {path.with_suffix('.log')}")
    return result


def validate_pilot(protocol,verification,radius,log,output):
    tests=read_json(verification)
    curve=read_json(radius)
    if tests.get("status")!="passed" or tests.get("code_sha256")!=code_fingerprint():
        raise ValueError("current offline verification report required")
    if curve.get("qualification_sha256")!=qualification_fingerprint(protocol) or curve.get("model")!=protocol["models"]["pilot"]:
        raise ValueError("radius curve is for a different pilot/protocol")
    if curve.get("code_sha256")!=code_fingerprint() or curve.get("model_parameters",float("inf"))>3_500_000_000:
        raise ValueError("current radius report on the <=3B-class pilot is required")
    if curve.get("perturb_precision") not in {"fp32", "bf16", "tf32"}:
        raise ValueError("radius report must declare a supported training precision")
    chosen=protocol["search"]["relative_radius"]
    threshold=number(protocol["pilot"]["max_fd_relative_error"],"pilot.max_fd_relative_error",minimum=1e-30)
    points=sorted(curve["curve"],key=lambda r:r["radius"])
    qualifying=[i for i,row in enumerate(points) if row["radius"]==chosen and row["relative_error"]<=threshold]
    if not qualifying:
        raise ValueError("selected radius does not pass the declared HVP error threshold")
    i=qualifying[0]
    if not any(points[j]["relative_error"]<=threshold for j in (i-1,i+1) if 0<=j<len(points)):
        raise ValueError("selected radius has no adjacent low-error radius; a flat region is not established")
    rows=[json.loads(line) for line in Path(log).read_text(encoding="utf-8").splitlines() if line.strip()]
    pilot_config_path=Path(log).parent/"resolved_config.yaml"
    if not pilot_config_path.exists():
        pilot_config_path=Path(log).parent/"config.yaml"
    pilot_config=yaml.safe_load(pilot_config_path.read_text(encoding="utf-8"))
    from src.core.cbg_trainer import CbgArguments
    cbg=CbgArguments(**pilot_config["CbgArguments"])
    if curve.get("perturb_precision") != cbg.spec_perturb_precision:
        raise ValueError("radius and pilot compute precision differ")
    if cbg.model_name!=protocol["models"]["pilot"] or asdict(cbg.spec_config())!=dict(curve["config"],r=chosen):
        raise ValueError("pilot training and radius qualification use different scientific settings")
    if (cbg.spec_solver!="exact" or cbg.spec_objective!="log_eta" or cbg.spec_damping!="relative"
        or cbg.fisher!="sampled" or cbg.drop_aln or cbg.detach_lambda
        or cbg.mu_mode!="differentiated"):
        raise ValueError("qualify the main CBG method before running ablations")
    if any(getattr(cbg,key)!=protocol["defense"][key] for key in (
        "safety_batch_size","benign_batch_size","spectral_modes","safety_max_length","benign_max_length")):
        raise ValueError("pilot structure differs from the declared main-method structure")
    rows=[r for r in rows if "cbg/log_eta_k" in r]
    if len(rows)<20 or max(r["step"] for r in rows)<protocol["pilot"]["max_steps"]:
        raise ValueError("pilot training is incomplete")
    for row in rows:
        for key in ("cbg/log_eta_k","cbg/L_aln","cbg/L_bgn","cbg/rho_gap","cbg/nu_gap"):
            value=row[key]
            if not isinstance(value,(int,float)) or not torch.isfinite(torch.tensor(value)):
                raise ValueError("pilot contains nonfinite/missing diagnostics")
    width=min(10,len(rows)//2)
    mean=lambda subset,key:sum(r[key] for r in subset)/len(subset)
    drop=mean(rows[:width],"cbg/log_eta_k")-mean(rows[-width:],"cbg/log_eta_k")
    growth=mean(rows[-width:],"cbg/L_aln")-mean(rows[:width],"cbg/L_aln")
    allowed=number(protocol["pilot"]["max_alignment_growth"],"pilot.max_alignment_growth")
    max_kl=number(protocol["pilot"]["max_benign_kl"],"pilot.max_benign_kl")
    if drop<=0 or growth>allowed or max(r["cbg/L_bgn"] for r in rows)>max_kl:
        raise ValueError("pilot failed geometry-decrease or bounded-loss criteria")
    result=dict(status="passed",code_sha256=code_fingerprint(),protocol_sha256=digest(protocol),
                qualification_sha256=qualification_fingerprint(protocol),
                radius=chosen,log_eta_drop=drop,alignment_growth=growth,
                artifacts=[dict(path=str(Path(p).resolve()),sha256=file_digest(p))
                           for p in (verification,radius,log,pilot_config_path)])
    write_once(output,result)
    return result


def write_yaml_once(path,value):
    path=Path(path)
    if path.exists() and yaml.safe_load(path.read_text(encoding="utf-8"))!=value:
        raise ValueError(f"existing config differs: {path}")
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(yaml.safe_dump(value,sort_keys=False),encoding="utf-8")


def write_resolved_config_once(path, value):
    """Freeze executed settings while ignoring HF's timestamped log directory.

    ``TrainingArguments`` synthesizes ``logging_dir`` from the current date when
    it is omitted from YAML.  That path is operational metadata, so it changes
    on an otherwise identical retry and must not invalidate the scientific
    receipt.  Every other resolved field remains frozen.
    """
    path = Path(path)

    def comparable(document):
        document = copy.deepcopy(document)
        training = document.get("TrainingArguments", {})
        if isinstance(training, dict):
            training.pop("logging_dir", None)
        return document

    if path.exists():
        existing = yaml.safe_load(path.read_text(encoding="utf-8"))
        if comparable(existing) != comparable(value):
            raise ValueError(f"existing resolved config differs: {path}")
        return
    write_yaml_once(path, value)


def materialize(protocol,*,phase,choices_path=None,lock_path=None,gate_path=None,model=None):
    model=model or protocol["models"]["main" if phase!="pilot" else "pilot"]
    manifest=read_json(protocol["manifest"])
    validate_manifest(manifest)
    if phase!="pilot":
        if not gate_path:
            raise ValueError("S0 gate required before non-pilot defense training")
        checked_gate(gate_path,protocol)
    if phase in {"main","ablations"}:
        if not lock_path:
            raise ValueError("frozen validation selection required")
        lock=checked_lock(lock_path,protocol,manifest)
        if lock["model"]!=model:
            raise ValueError("each model requires its own validation selection")
        choices={"selected":lock["config"]} if phase=="main" else ablations(lock["config"],protocol)
    elif phase=="pilot":
        if not choices_path:
            raise ValueError("pilot requires an explicit JSON object of CBG hyperparameters via --choices")
        choices={"pilot":load_choices(choices_path)}
    else:
        choices=search_candidates(protocol)
    paths=[]
    seeds=protocol["seeds"]["defense"] if phase=="main" else protocol["seeds"]["defense"][:1]
    for key,choice in choices.items():
        for seed in seeds:
            out=Path(protocol["output_root"])/model.split("/")[-1]/f"cbg-s{seed}"/f"{phase}-{key}"
            config=defense_config(protocol,choice,model=model,seed=seed,output_dir=out,pilot=phase=="pilot")
            write_yaml_once(out/"config.yaml",config)
            write_once(out/"scenario_identity.json",dict(phase=phase,candidate_id=key,model=model,
                protocol_sha256=digest(protocol),manifest_sha256=manifest["sha256"],choice_sha256=digest(choice)))
            paths.append(str(out/"config.yaml"))
    return dict(configs=paths,command_template="python -m src.train --method cbg --config <config.yaml>")


def make_bank(protocol,model_name,choices,output,*,pool,device):
    from src.experiments.runner import load_model
    from src.data_processing.cbg_datasets import CbgPairedSafetyDataset,CbgBenignDataset,CbgJointBatchDataset
    # Tokenizer only; no model or weights are needed to prepare probe batches.
    from transformers import AutoTokenizer
    tokenizer=AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token=tokenizer.eos_token
    manifest=read_json(protocol["manifest"])
    loaded=validate_manifest(manifest)
    safety_indices=manifest[pool+"_indices"]
    safety=CbgPairedSafetyDataset([loaded["safety"][i] for i in safety_indices],tokenizer,
                                 max_length=protocol["defense"]["safety_max_length"])
    benign=CbgBenignDataset([loaded["benign"][i] for i in manifest["benign_indices"]],tokenizer,
                           max_length=protocol["defense"]["benign_max_length"])
    data=CbgJointBatchDataset(safety,benign,m=protocol["defense"]["safety_batch_size"],
                             n=protocol["defense"]["benign_batch_size"],seed=protocol["seeds"]["geometry"])
    count=protocol["geometry"]["batches"]
    if count>len(data):
        raise ValueError("not enough distinct joint batches for geometry bank")
    bank=dict(batches=[data[i] for i in range(count)],manifest_sha256=manifest["sha256"],pool=pool,
              tokenizer_sha256=digest(tokenizer.get_vocab()),tokenizer=model_name,
              seed=protocol["seeds"]["geometry"],protocol_sha256=digest(protocol))
    path=Path(output)
    if path.exists():
        raise ValueError("probe bank already exists; reuse it instead of replacing it")
    path.parent.mkdir(parents=True,exist_ok=True)
    torch.save(bank,path)
    return dict(bank=str(path),sha256=file_digest(path),batches=count)


def run_probe(protocol,checkpoint,choices_path,bank_path,output,*,device,radius=False,path_samples=None,perturb_precision="fp32"):
    from src.experiments.runner import load_model,move_batch
    from src.experiments.geometry import radius_curve,evaluate_geometry
    from src.core.cbg_trainer import CbgArguments
    choices=load_choices(choices_path)
    config=CbgArguments(**defense_config(protocol,choices,model=checkpoint,seed=0,output_dir="unused")["CbgArguments"]).spec_config()
    model,tokenizer=load_model(checkpoint,device)
    model.requires_grad_(True)
    bank=torch.load(bank_path,map_location="cpu",weights_only=True)
    manifest=read_json(protocol["manifest"])
    validate_manifest(manifest)
    if bank["tokenizer_sha256"]!=digest(tokenizer.get_vocab()) or bank["manifest_sha256"]!=manifest["sha256"]:
        raise ValueError("probe tokenizer or data differs; never reuse token IDs across vocabularies")
    batches=[move_batch(batch,device) for batch in bank["batches"]]
    if radius:
        config=replace(config,perturb_precision=perturb_precision)
        result=radius_curve(model,batches[0],config,protocol["pilot"]["radii"],seed=bank["seed"],perturb_precision=perturb_precision)
    else:
        fixed=torch.load(path_samples,map_location="cpu",weights_only=True) if path_samples else None
        if fixed is not None and fixed["bank_sha256"]!=file_digest(bank_path):
            raise ValueError("trajectory Fisher samples belong to a different probe bank")
        result,samples=evaluate_geometry(model,batches,config,seed=bank["seed"],
                                         path_samples=fixed["samples"] if fixed else None)
        sample_path=Path(output).with_suffix(".samples.pt")
        torch.save(dict(bank_sha256=file_digest(bank_path),samples=samples),sample_path)
        result["path_samples_sha256"]=digest([sample.tolist() for sample in samples])
    result.update(model=checkpoint,model_parameters=sum(p.numel() for p in model.parameters()),
                  protocol_sha256=digest(protocol),code_sha256=code_fingerprint(),bank_sha256=file_digest(bank_path),
                  config=asdict(config))
    if radius:
        result["qualification_sha256"]=qualification_fingerprint(protocol)
    write_json(output,result)
    return result


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol",default="config/cbg_scenarios.yaml")
    sub=parser.add_subparsers(dest="command",required=True)
    sub.add_parser("plan")
    suite=sub.add_parser("suite")
    suite.add_argument("--registry",required=True)
    suite.add_argument("--locks",required=True,help="JSON mapping of original model IDs to selection lock paths")
    suite.add_argument("--output",required=True)
    prepare=sub.add_parser("prepare-splits")
    prepare.add_argument("--cross-category",action="store_true")
    verification=sub.add_parser("verify")
    verification.add_argument("--output",required=True)
    pilot=sub.add_parser("validate-pilot")
    for key in ("verification","radius","log","output"):
        pilot.add_argument("--"+key,required=True)
    mat=sub.add_parser("materialize")
    mat.add_argument("--phase",choices=("pilot","search","main","ablations"),required=True)
    for key in ("choices","lock","gate","model"):
        mat.add_argument("--"+key)
    select=sub.add_parser("select")
    for key in ("observations","output"):
        select.add_argument("--"+key,required=True)
    select.add_argument("--model")
    record=sub.add_parser("record-validation")
    for key in ("candidate-id","attack-id","pre-metrics","post-metrics","base-metrics","attack-identity","output"):
        record.add_argument("--"+key,required=True)
    record.add_argument("--model")
    bank=sub.add_parser("make-bank")
    bank.add_argument("--model",required=True)
    bank.add_argument("--output",required=True)
    bank.add_argument("--pool",choices=("defense","validation"),default="validation")
    for name in ("radius","geometry"):
        probe=sub.add_parser(name)
        for key in ("checkpoint","choices","bank","output"):
            probe.add_argument("--"+key,required=True)
        probe.add_argument("--device",default="cuda:0")
        probe.add_argument("--path-samples")
        probe.add_argument("--perturb-precision",choices=("fp32","bf16","tf32"),default="fp32")
    attack=sub.add_parser("attack")
    for key in ("checkpoint","scenario","attack-id","output","pool"):
        attack.add_argument("--"+key,required=True)
    attack.add_argument("--lock")
    attack.add_argument("--choices")
    attack.add_argument("--resume")
    attack.add_argument("--device",default="cuda:0")
    evaluation=sub.add_parser("evaluate")
    for key in ("checkpoint","output"):
        evaluation.add_argument("--"+key,required=True)
    evaluation.add_argument("--device",default="cuda:0")
    evaluation.add_argument("--batch-size",type=int,default=4)
    evaluation.add_argument("--judge",action="store_true")
    evaluation.add_argument("--pool",choices=("evaluation","validation"),default="evaluation")
    evaluation.add_argument("--utility-subset")
    trajectory=sub.add_parser("trajectory")
    trajectory.add_argument("--geometry",nargs="+",required=True,help="geometry JSONs in checkpoint order")
    trajectory.add_argument("--output",required=True)
    analysis=sub.add_parser("correlate")
    analysis.add_argument("--observations",required=True)
    analysis.add_argument("--hs-bins",nargs="+",type=float,required=True)
    analysis.add_argument("--bootstrap-samples",type=int,required=True)
    analysis.add_argument("--output",required=True)
    restore=sub.add_parser("restore")
    for key in ("checkpoint","collapse-metrics","attack-identity","lock","output"):
        restore.add_argument("--"+key,required=True)
    restore.add_argument("--mixture",choices=("benign","harmful","mixed"),required=True)
    restore.add_argument("--device",default="cuda:0")
    restore.add_argument("--resume")
    external=sub.add_parser("external")
    external.add_argument("--adapter",required=True)
    external.add_argument("--config",required=True)
    args=parser.parse_args(argv)
    p=load_protocol(args.protocol)
    if args.command=="plan":
        result=scenario_inventory(p)
    elif args.command=="suite":
        from src.experiments.suite import compile_suite
        registry=yaml.safe_load(Path(args.registry).read_text(encoding="utf-8"))
        result=compile_suite(p,registry,protocol_path=args.protocol,locks=read_json(args.locks))
        write_json(args.output,result)
    elif args.command=="prepare-splits":
        d=p["data"]
        if args.cross_category and d["category_order"] is None:
            raise ValueError("S7 category order is an author decision")
        result=prepare_splits(d["safety"],d["benign"],d["evaluation"],seed=p["seeds"]["split"],
            refusal_model=d["refusal_model"],category_order=d["category_order"] if args.cross_category else None,
            **{k:d[k] for k in ("defense_size","benign_size","validation_size","attack_size",
                                "max_rows_per_prompt")})
        write_once(p["manifest"],result)
    elif args.command=="verify":
        result=verify(args.output)
    elif args.command=="validate-pilot":
        result=validate_pilot(p,args.verification,args.radius,args.log,args.output)
    elif args.command=="materialize":
        result=materialize(p,phase=args.phase,choices_path=args.choices,lock_path=args.lock,gate_path=args.gate,model=args.model)
    elif args.command=="select":
        manifest=read_json(p["manifest"])
        validate_manifest(manifest)
        s=p["search"]
        result=select_configuration(search_candidates(p),read_json(args.observations),manifest_sha256=manifest["sha256"],
            utility_subset_sha256=s["utility_subset_sha256"],max_zs_drop_pp=s["max_zs_drop_pp"],attack_seed=p["seeds"]["attack"],
            model=args.model or p["models"]["main"])
        result.update(protocol_sha256=digest(p),model=args.model or p["models"]["main"])
        result["sha256"]=digest(result)
        write_once(args.output,result)
    elif args.command=="record-validation":
        candidates=search_candidates(p)
        if args.candidate_id not in candidates or args.attack_id not in {"2","4"}:
            raise ValueError("candidate and attack must belong to the fixed validation grid")
        manifest=read_json(p["manifest"])
        metrics=[read_json(path) for path in (args.pre_metrics,args.post_metrics,args.base_metrics)]
        for metric in metrics:
            if (metric.get("pool")!="validation" or metric.get("manifest_sha256")!=manifest["sha256"]
                or metric.get("utility_subset_sha256")!=p["search"]["utility_subset_sha256"]):
                raise ValueError("all selection metrics must use the validation data and held-out MMLU subset")
        pre,post,base=metrics
        identity=read_json(args.attack_identity)
        if (identity.get("pool")!="validation" or identity.get("manifest_sha256")!=manifest["sha256"]
            or identity.get("checkpoint")!=pre["checkpoint"] or identity["recipe"]["id"]!=args.attack_id
            or identity["recipe"]["scenario"]!="S3.1" or identity["recipe"]["seed"]!=p["seeds"]["attack"]):
            raise ValueError("validation attack identity differs from declared comparison")
        expected_final=Path(args.attack_identity).parent/"final"
        if Path(post["checkpoint"]).resolve()!=expected_final.resolve():
            raise ValueError("post-attack metrics belong to a different checkpoint")
        candidate_identity=read_json(Path(pre["checkpoint"]).parent/"scenario_identity.json")
        if (candidate_identity.get("candidate_id")!=args.candidate_id
            or candidate_identity.get("choice_sha256")!=digest(candidates[args.candidate_id])
            or candidate_identity.get("protocol_sha256")!=digest(p)
            or candidate_identity.get("model")!=(args.model or p["models"]["main"])):
            raise ValueError("pre-attack checkpoint does not identify the requested candidate")
        resolved_path=Path(pre["checkpoint"]).parent/"resolved_config.yaml"
        resolved=yaml.safe_load(resolved_path.read_text(encoding="utf-8"))
        if any(resolved["CbgArguments"].get(key)!=value for key,value in candidates[args.candidate_id].items()):
            raise ValueError("candidate's executed scientific config differs from the search grid")
        for key in ("learning_rate","max_steps","warmup_ratio","weight_decay","max_grad_norm"):
            if resolved["TrainingArguments"].get(key)!=p["defense"][key]:
                raise ValueError(f"candidate executed a different {key} than the validation protocol")
        expected_recipe=next(r for r in attack_catalog(p) if r["scenario"]=="S3.1" and r["id"]==args.attack_id)
        if identity["recipe"]!=expected_recipe:
            raise ValueError("attack settings differ from the fixed validation recipe")
        result=dict(candidate_id=args.candidate_id,candidate_sha256=digest(candidates[args.candidate_id]),
                    model=args.model or p["models"]["main"],attack_id=args.attack_id,pool="validation",
                    manifest_sha256=manifest["sha256"],utility_subset_sha256=p["search"]["utility_subset_sha256"],
                    attack_seed=p["seeds"]["attack"],status="complete",pre_zs=pre["validation_utility"],
                    post_zs=post["validation_utility"],base_zs=base["validation_utility"],hs=post["hs"])
        write_once(args.output,result)
    elif args.command=="make-bank":
        result=make_bank(p,args.model,None,args.output,pool=args.pool,device="cpu")
    elif args.command in {"radius","geometry"}:
        Path(args.output).parent.mkdir(parents=True,exist_ok=True)
        result=run_probe(p,args.checkpoint,args.choices,args.bank,args.output,device=args.device,
                         radius=args.command=="radius",path_samples=args.path_samples,perturb_precision=args.perturb_precision)
    elif args.command=="attack":
        from src.experiments.runner import run_attack
        from src.core.cbg_trainer import CbgArguments
        if args.pool not in {"validation","attack"}:
            raise ValueError("pool must be validation or attack")
        manifest=read_json(p["manifest"])
        if args.pool=="attack":
            if not args.lock:
                raise ValueError("test attacks require --lock from validation selection")
            checked_lock(args.lock,p,manifest)
        rows=[r for r in attack_catalog(p) if r["scenario"]==args.scenario and r["id"]==args.attack_id]
        if len(rows)!=1:
            raise ValueError("unknown/ambiguous recipe; inspect plan")
        recipe=rows[0]
        if recipe["family"]=="external":
            raise ValueError(f"use the original {recipe['adapter']} adapter via the external command")
        if args.pool=="validation" and (args.scenario!="S3.1" or args.attack_id not in {"2","4"}):
            raise ValueError("selection validation protocol uses only standard attacks #2 and #4")
        missing=unresolved(recipe)
        if missing:
            raise ValueError(f"unresolved attack settings: {missing}")
        geometry=None
        if recipe["family"]=="reverse_cbg":
            choices=load_choices(args.choices) if args.choices else checked_lock(args.lock,p,manifest)["config"]
            config=defense_config(p,choices,model=args.checkpoint,seed=0,output_dir="unused")
            geometry=asdict(CbgArguments(**config["CbgArguments"]).spec_config())
            recipe["geometry_m"]=p["defense"]["safety_batch_size"]
        write_yaml_once(Path(args.output)/"config.yaml",dict(recipe=recipe,checkpoint=args.checkpoint,pool=args.pool))
        result=run_attack(args.checkpoint,p["manifest"],recipe,args.output,pool=args.pool,device=args.device,
                          geometry_config=geometry,resume=args.resume)
    elif args.command=="evaluate":
        from src.experiments.evaluation import evaluate
        result=evaluate(args.checkpoint,p["manifest"],p["evaluation"],args.output,device=args.device,
                        batch_size=args.batch_size,seed=p["seeds"]["attack"],judge=args.judge,
                        pool=args.pool,utility_subset=args.utility_subset,
                        utility_subset_sha256=p["search"]["utility_subset_sha256"])
    elif args.command=="trajectory":
        from src.experiments.geometry import path_summary
        result=path_summary([read_json(path) for path in args.geometry])
        write_json(args.output,result)
    elif args.command=="correlate":
        from src.experiments.analysis import matched_correlations
        result=matched_correlations(read_json(args.observations),hs_bin_edges=args.hs_bins,
                                    bootstrap_samples=args.bootstrap_samples,seed=p["seeds"]["geometry"])
        write_json(args.output,result)
    elif args.command=="restore":
        from src.experiments.scenarios import restoration_recipe
        from src.experiments.runner import run_attack
        manifest=read_json(p["manifest"])
        checked_lock(args.lock,p,manifest)
        collapse=read_json(args.collapse_metrics)
        identity=read_json(args.attack_identity)
        threshold=number(p["restoration"]["collapse_max_zs"],"restoration.collapse_max_zs")
        if (collapse.get("checkpoint")!=args.checkpoint or collapse.get("pool")!="evaluation"
            or collapse.get("manifest_sha256")!=manifest["sha256"] or collapse.get("zs",101)>threshold
            or identity["recipe"]["scenario"]!="S3.1" or identity["recipe"]["id"]!="4"):
            raise ValueError("S13 requires an evaluated attack-4 checkpoint meeting the declared collapse criterion")
        if ((Path(args.attack_identity).parent/"final").resolve()!=Path(args.checkpoint).resolve()
            or identity.get("manifest_sha256")!=manifest["sha256"]):
            raise ValueError("collapse metrics and attack provenance refer to different runs")
        recipe=restoration_recipe(p,args.mixture)
        if unresolved(recipe):
            raise ValueError("fill restoration optimizer/clipping choices before running S13")
        write_yaml_once(Path(args.output)/"config.yaml",dict(recipe=recipe,checkpoint=args.checkpoint,pool="attack"))
        result=run_attack(args.checkpoint,p["manifest"],recipe,args.output,pool="attack",device=args.device,resume=args.resume)
    else:
        command=p["external"].get(args.adapter)
        if not isinstance(command,list) or not command or not all(isinstance(v,str) for v in command):
            raise ValueError(f"configure external.{args.adapter} as an argv array for the original implementation")
        if not Path(args.config).is_file():
            raise ValueError("external adapter config does not exist")
        subprocess.run([part.replace("{config}",str(Path(args.config).resolve())) for part in command],check=True)
        result=dict(adapter=args.adapter,status="command_completed")
    print(json.dumps(result,indent=2,allow_nan=False))


if __name__=="__main__":
    main()
