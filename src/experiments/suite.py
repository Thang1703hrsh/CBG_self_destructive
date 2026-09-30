"""Expand a checkpoint registry into reviewable commands, never launch a sweep."""
from pathlib import Path
import re
import sys

from src.experiments.protocol import digest
from src.experiments.scenarios import attack_catalog,unresolved


def slug(value):
    text=re.sub(r"[^A-Za-z0-9_.-]+","-",value)
    if not text or text in {".",".."}:
        raise ValueError("invalid run name")
    return text


def compile_suite(protocol,registry,*,protocol_path,locks):
    """Main S2--S5 commands; separate phases cover S0/S1/S7/S8/S10/S13.

    locks maps original model IDs to their independently selected lock files.
    Incomplete registries and external methods remain visible as blocked jobs.
    """
    if not isinstance(registry,list) or not registry:
        raise ValueError("checkpoint registry must be a nonempty list")
    jobs=[]
    seen=set()
    recipes=attack_catalog(protocol)
    for entry in registry:
        model,defense,seed=entry["model"],entry["defense"],entry["seed"]
        key=(model,defense,seed)
        if key in seen:
            raise ValueError("duplicate model/defense/seed registry entry")
        seen.add(key)
        root=Path(protocol["output_root"])/slug(model.split("/")[-1])/f"{slug(defense)}-s{seed}"
        checkpoint=entry.get("checkpoint")
        prefix=[sys.executable,"-m","src.experiments","--protocol",str(protocol_path)]
        missing=[] if checkpoint else ["checkpoint path"]
        eval_dir=root/"S2-pre-attack"
        jobs.append(dict(id=str(eval_dir),scenario="S2",blocked=missing+[
            f"evaluation.{k}" for k in ("max_new_tokens","harmful_label","hs_reduction") if protocol["evaluation"][k] is None],
            command=prefix+["evaluate","--checkpoint",checkpoint or "UNSET","--output",str(eval_dir)]))
        for recipe in recipes:
            scenario=recipe["scenario"]
            if defense not in {"base","cbg","seam"} and scenario!="S3.1":
                continue
            if model!=protocol["models"]["main"] and not (
                (scenario=="S3.1" and int(recipe["id"])<=5) or scenario in {"S5.1","S5.3"}):
                continue
            if scenario in {"S3.2","S3.3"} and defense!="base" and seed!=protocol["seeds"]["defense"][0]:
                continue
            out=root/f"{scenario}-{slug(recipe['id'])}-standard"
            blocked=missing+unresolved(recipe)
            if model not in locks:
                blocked.append("model selection lock")
            if recipe["family"]=="external":
                blocked.append(f"configure/run external adapter {recipe['adapter']}")
            command=prefix+["attack","--checkpoint",checkpoint or "UNSET","--scenario",scenario,
                            "--attack-id",recipe["id"],"--pool","attack","--lock",locks.get(model,"UNSET"),
                            "--output",str(out)]
            jobs.append(dict(id=str(out),scenario=scenario,blocked=blocked,command=command,
                post_evaluation_command=prefix+["evaluate","--checkpoint",str(out/"final"),"--output",str(out/"evaluation")],
                recipe=recipe))
    missing_seeds=[]
    for model in {r["model"] for r in registry}:
        expected=protocol["seeds"]["defense"] if model==protocol["models"]["main"] else protocol["seeds"]["defense"][:1]
        for defense in ("cbg","seam"):
            for seed in expected:
                if (model,defense,seed) not in seen:
                    missing_seeds.append(dict(model=model,defense=defense,seed=seed))
    return dict(protocol_sha256=digest(protocol),registry_sha256=digest(registry),jobs=jobs,
                missing_main_comparison_seeds=missing_seeds,
                note="Commands are argv arrays. No jobs have been launched; blocked jobs require explicit settings/artifacts.")
