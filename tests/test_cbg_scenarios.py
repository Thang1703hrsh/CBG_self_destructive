"""Offline protocol, attack, analysis, and checkpoint regression tests."""
import os
os.environ.setdefault("USE_TF","0")
from collections import Counter,OrderedDict
import copy
from dataclasses import asdict,replace
import json
from pathlib import Path
import tempfile
import unittest

import torch

from test_cbg_spec import fixture,fp64_norm,fp64_attention
from src.cbg.spec import (_center,_forward,joint_batch,measurements,spec_precision,spec_training_gradient)
from src.cbg.spec_hooks import spec_linear_modules
from src.experiments.attacks import attack_gradient,benign_scores,nll,project_benign_span,rescale_swiglu
from src.experiments.analysis import compromised,hs_score,zs_score,matched_correlations,summarize_seeds
from src.experiments.geometry import radius_curve,evaluate_geometry,path_summary
from src.experiments.protocol import digest,prepare_splits,select_configuration,validate_manifest,write_json
from src.experiments.scenarios import standard_attacks,attack_catalog,load_protocol,search_candidates,ablations
from src.experiments.runner import BatchStream,train_attack
from src.experiments.evaluation import parse_judge_label
from src.experiments.cli import write_resolved_config_once
from src.cbg.spec_spectral import solve_spec_spectrum
from unittest.mock import patch


class ScenarioProtocolTest(unittest.TestCase):
    def test_resolved_config_retry_ignores_only_timestamped_logging_dir(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "resolved.yaml"
            first = {
                "TrainingArguments": {"learning_rate": 2e-5, "logging_dir": "runs/first"},
                "CbgArguments": {"beta": 0.01},
            }
            write_resolved_config_once(path, first)
            write_resolved_config_once(path, {
                "TrainingArguments": {"learning_rate": 2e-5, "logging_dir": "runs/retry"},
                "CbgArguments": {"beta": 0.01},
            })
            self.assertEqual("runs/first", __import__("yaml").safe_load(
                path.read_text(encoding="utf-8"))["TrainingArguments"]["logging_dir"])
            with self.assertRaisesRegex(ValueError, "resolved config differs"):
                write_resolved_config_once(path, {
                    "TrainingArguments": {"learning_rate": 3e-5, "logging_dir": "runs/retry"},
                    "CbgArguments": {"beta": 0.01},
                })

    def test_prompt_disjoint_hashed_three_way_splits_and_mutation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            safety=[dict(prompt=f"prompt {i}",response="bad",refusal="refuse",is_safe=False) for i in range(20)]
            safety += [dict(safety[0],prompt=" PROMPT   0 ")]
            benign=[dict(instruction=f"benign {i}",output="answer") for i in range(7)]
            evaluation=[dict(prompt="prompt 19"),dict(prompt="eval 1")]
            for name,rows in (("safety",safety),("benign",benign),("eval",evaluation)):
                write_json(root/f"{name}.json",rows)
            args=dict(seed=5,defense_size=4,validation_size=3,attack_size=5,benign_size=4,refusal_model="external/model@revision")
            manifest=prepare_splits(root/"safety.json",root/"benign.json",root/"eval.json",**args)
            self.assertEqual(manifest,prepare_splits(root/"safety.json",root/"benign.json",root/"eval.json",**args))
            validate_manifest(manifest)
            groups=[set(manifest[k+"_indices"]) for k in ("defense","validation","attack")]
            self.assertFalse(any(groups[i]&groups[j] for i in range(3) for j in range(i)))
            self.assertNotIn(19,set.union(*groups))
            self.assertEqual(manifest["evaluation_indices"],[0,1])
            safety[1]["response"]="changed"
            write_json(root/"safety.json",safety)
            with self.assertRaisesRegex(ValueError,"changed"):
                validate_manifest(manifest)

    def test_response_rows_are_capped_while_prompt_pools_remain_disjoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            safety=[dict(prompt=f"prompt {i}",response=f"bad {j}",refusal="refuse",is_safe=False)
                    for i in range(10) for j in range(3)]
            benign=[dict(instruction=f"benign {i}",output="answer") for i in range(5)]
            evaluation=[dict(prompt="held out")]
            for name,rows in (("safety",safety),("benign",benign),("eval",evaluation)):
                write_json(root/f"{name}.json",rows)
            manifest=prepare_splits(root/"safety.json",root/"benign.json",root/"eval.json",
                seed=5,defense_size=8,validation_size=6,attack_size=6,benign_size=4,
                refusal_model="external/model@revision",max_rows_per_prompt=2)
            loaded=validate_manifest(manifest)["safety"]
            groups=[]
            for name in ("defense","validation","attack"):
                prompts=[loaded[i]["prompt"] for i in manifest[name+"_indices"]]
                self.assertLessEqual(max(Counter(prompts).values()),2)
                self.assertTrue(any(count==2 for count in Counter(prompts).values()))
                groups.append(set(prompts))
            self.assertFalse(any(groups[i]&groups[j] for i in range(3) for j in range(i)))

    def test_selection_rejects_leakage_missing_grid_and_changed_candidates(self):
        candidates={"a":dict(beta=.1),"b":dict(beta=.2)}
        rows=[dict(candidate_id=key,candidate_sha256=digest(candidate),attack_id=attack,
                   pool="validation",manifest_sha256="manifest",utility_subset_sha256="mmlu",attack_seed=3,
                   status="complete",base_zs=60.,pre_zs=59.,post_zs=50. if key=="a" else 40.,hs=5.)
              for key,candidate in candidates.items() for attack in ("2","4")]
        kwargs=dict(manifest_sha256="manifest",utility_subset_sha256="mmlu",max_zs_drop_pp=2,attack_seed=3)
        selected=select_configuration(candidates,rows,**kwargs)
        self.assertEqual(selected["selected_id"],"b")
        with self.assertRaisesRegex(ValueError,"incomplete"):
            select_configuration(candidates,rows[:-1],**kwargs)
        changed=copy.deepcopy(rows)
        changed[0]["pool"]="attack"
        with self.assertRaisesRegex(ValueError,"validation"):
            select_configuration(candidates,changed,**kwargs)
        with self.assertRaises(ValueError):
            select_configuration(dict(candidates,a=dict(beta=.3)),rows,**kwargs)

    def test_attack_catalog_counts_settings_and_fixed_budget(self):
        protocol=load_protocol("config/cbg_scenarios.yaml")
        standard=standard_attacks()
        self.assertEqual(len(standard),11)
        self.assertEqual([r["steps"] for r in standard],[250]*5+[25000]*6)
        self.assertEqual(standard[7]["method"],"lora")
        self.assertEqual(standard[9]["optimizer"],"sgd")
        self.assertEqual(standard[3]["learning_rate"],1e-4)
        recipes=attack_catalog(protocol)
        self.assertEqual(len([r for r in recipes if r["scenario"]=="S5.8"]),12)
        self.assertEqual(len({(r["scenario"],r["id"]) for r in recipes}),len(recipes))
        variants=ablations(dict(beta=.1),protocol)
        self.assertEqual(variants["v-n128"]["benign_max_length"],32)
        with self.assertRaises(ValueError):
            search_candidates(protocol)

    def test_metrics_units_threshold_ties_bootstrap_and_strict_judge(self):
        self.assertFalse(compromised(10,31))
        self.assertFalse(compromised(11,30))
        self.assertTrue(compromised(11,31))
        self.assertEqual(hs_score([.2,.8],reduction="mean_probability",threshold=.5)["hs"],50)
        self.assertFalse(parse_judge_label(" Not Harmful \n"))
        with self.assertRaises(ValueError):
            parse_judge_label("maybe")
        tasks={key:"acc" for key in "abcd"}
        self.assertEqual(zs_score({k:{"acc":.5} for k in tasks},tasks)["zs"],50)
        rows=[dict(defense="cbg",cluster_id=str(i),eta_k=i/10,zs_drop=10-i,delta_hs=5) for i in range(5)]
        result=matched_correlations(rows,hs_bin_edges=[0,10],bootstrap_samples=50,seed=4)
        self.assertAlmostEqual(result[0]["spearman"],-1)
        self.assertIsNotNone(result[0]["ci95"])
        self.assertIsNone(summarize_seeds([dict(defense_seed=1,hs=2,zs=50)])["hs"]["std"])

    def test_s0_gate_binds_qualification_and_detects_changed_evidence(self):
        import yaml
        from src.core.cbg_trainer import CbgArguments
        from src.experiments.scenarios import defense_config
        from src.experiments.cli import code_fingerprint,qualification_fingerprint,validate_pilot,checked_gate
        protocol=load_protocol("config/cbg_scenarios.yaml")
        protocol["pilot"].update(max_steps=20,max_fd_relative_error=.01,max_alignment_growth=.1,max_benign_kl=.1)
        protocol["search"]["relative_radius"]=.001
        choices=dict(alpha=1.,beta=.01,xi=.001,kappa_mu=.05,relative_radius=.001)
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            protocol["manifest"]=str(root/"manifest.json")
            write_json(protocol["manifest"],dict(sha256="data"))
            config=defense_config(protocol,choices,model=protocol["models"]["pilot"],seed=42,output_dir=tmp,pilot=True)
            (root/"config.yaml").write_text(yaml.safe_dump(config),encoding="utf-8")
            write_json(root/"verification.json",dict(status="passed",code_sha256=code_fingerprint()))
            write_json(root/"radius.json",dict(qualification_sha256=qualification_fingerprint(protocol),
                model=protocol["models"]["pilot"],code_sha256=code_fingerprint(),model_parameters=1000,
                perturb_precision="fp32",config=asdict(CbgArguments(**config["CbgArguments"]).spec_config()),
                curve=[dict(radius=.001,relative_error=.001),dict(radius=.003,relative_error=.002)]))
            rows=[dict(step=i,**{"cbg/log_eta_k":-.1-i*.01,"cbg/L_aln":2.,"cbg/L_bgn":.01,
                                "cbg/rho_gap":.1,"cbg/nu_gap":.1}) for i in range(1,21)]
            log=root/"train_log.jsonl"
            log.write_text("".join(json.dumps(r)+"\n" for r in rows))
            validate_pilot(protocol,root/"verification.json",root/"radius.json",log,root/"gate.json")
            checked_gate(root/"gate.json",protocol)
            changed=copy.deepcopy(protocol)
            changed["search"]["beta"]=[.001,.01,.1,1.]
            checked_gate(root/"gate.json",changed)
            changed["search"]["relative_radius"]=.003
            with self.assertRaises(ValueError):
                checked_gate(root/"gate.json",changed)
            log.write_text(log.read_text()+"{}\n")
            with self.assertRaisesRegex(ValueError,"evidence changed"):
                checked_gate(root/"gate.json",protocol)

    def test_suite_is_reviewable_and_reports_missing_seeds(self):
        from src.experiments.suite import compile_suite
        p=load_protocol("config/cbg_scenarios.yaml")
        registry=[dict(model=p["models"]["main"],defense="cbg",seed=42,checkpoint="checkpoint")]
        suite=compile_suite(p,registry,protocol_path="protocol.yaml",locks={})
        standard=[job for job in suite["jobs"] if job["scenario"]=="S3.1"]
        self.assertEqual(len(standard),11)
        self.assertTrue(all("model selection lock" in j["blocked"] for j in standard))
        self.assertEqual(len(suite["missing_main_comparison_seeds"]),5)
        self.assertIsInstance(standard[0]["command"],list)

    def test_every_native_ablation_materializes_without_duplicate_arguments(self):
        from src.experiments.scenarios import defense_config
        p=load_protocol("config/cbg_scenarios.yaml")
        p["ablations"].update(xi_values=[.001,.002],rayleigh_step_size=.5)
        selected=dict(alpha=1.,beta=.01,xi=.001,kappa_mu=.05,relative_radius=.001)
        variants=ablations(selected,p)
        for key,choice in variants.items():
            with self.subTest(variant=key):
                cfg=defense_config(p,choice,model=p["models"]["main"],seed=42,output_dir="unused")
                self.assertEqual(cfg["CbgArguments"]["relative_radius"],choice["relative_radius"])
        self.assertEqual(defense_config(p,variants["v-n128"],model="model",seed=42,output_dir="unused")
                         ["CbgArguments"]["benign_batch_size"],128)

    def test_verification_launches_from_repository_root_and_rejects_skips(self):
        from src.experiments.cli import verify
        import subprocess
        with tempfile.TemporaryDirectory() as tmp:
            with patch("src.experiments.cli.subprocess.run",return_value=subprocess.CompletedProcess([],0,"","Ran 1 test\nOK")) as run:
                result=verify(Path(tmp)/"verification.json")
                self.assertEqual(result["status"],"passed")
                self.assertTrue((run.call_args.kwargs["cwd"]/"tests/test_cbg_spec.py").is_file())
            with patch("src.experiments.cli.subprocess.run",return_value=subprocess.CompletedProcess([],0,"","OK (skipped=1)")):
                with self.assertRaises(ValueError):
                    verify(Path(tmp)/"skipped.json")
            nested="Nested verifier rejected this fixture:\nOK (skipped=1)\nRan 1 test\nOK\n"
            with patch("src.experiments.cli.subprocess.run",return_value=subprocess.CompletedProcess([],0,"",nested)):
                self.assertEqual(verify(Path(tmp)/"nested.json")["status"],"passed")
            for code,log in ((1,"OK"),(0,"no unittest summary"),(0,"FAILED (errors=1)")):
                with self.subTest(code=code,log=log):
                    with patch("src.experiments.cli.subprocess.run",return_value=subprocess.CompletedProcess([],code,"",log)):
                        with self.assertRaises(ValueError):
                            verify(Path(tmp)/"failed.json")


class ScenarioNumericsTest(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        for target,replacement in (("LlamaRMSNorm.forward",fp64_norm),("eager_attention_forward",fp64_attention)):
            mocked=patch("transformers.models.llama.modeling_llama."+target,replacement)
            mocked.start()
            self.addCleanup(mocked.stop)
        self.model,self.batch,self.config=fixture()

    def test_raw_rho_estimator_matches_differentiable_dense_kernel(self):
        config=replace(self.config,alpha=0.,beta=1.,drop_aln=True,k=1,objective="raw_rho",r=1e-5)
        batch=joint_batch(self.batch)
        params=OrderedDict(self.model.named_parameters())
        modules=spec_linear_modules(self.model)
        with spec_precision(self.model):
            q,samples,_,_,_=_center(self.model,batch,config,modules,generator=torch.Generator().manual_seed(11))
        values=measurements(_forward(self.model,batch),batch,samples=samples).values
        K=values.new_zeros(5,5)
        rows=[torch.autograd.grad(value,tuple(params.values()),create_graph=True,retain_graph=True) for value in values]
        for index,(name,p) in enumerate(params.items()):
            if name in modules:
                jac=torch.stack([row[index].reshape(-1) for row in rows])
                K=K+(jac*(p.square()+config.xi**2).flatten())@jac.T
        hh,hb,bb=K[:2,:2],K[:2,2:],K[2:,2:]
        M=(hh-hb@torch.linalg.solve(bb+q.mu*torch.eye(3),hb.T))/q.mu
        rho=torch.linalg.eigvalsh(M)[-1]
        exact=torch.autograd.grad(rho,tuple(params.values()))
        estimated=spec_training_gradient(self.model,self.batch,config,generator=torch.Generator().manual_seed(11))
        error=sum((estimated.gradient[n]-g).square().sum() for (n,_),g in zip(params.items(),exact)).sqrt()
        norm=sum(g.square().sum() for g in exact).sqrt()
        self.assertLess(float(error/norm),1e-6)

    def test_projected_attack_matches_dense_all_parameter_projection(self):
        params=OrderedDict(self.model.named_parameters())
        loss=nll(self.model,self.batch["harmful"])
        values=torch.autograd.grad(loss,tuple(params.values()))
        gradient=OrderedDict((n,g.detach()) for (n,_),g in zip(params.items(),values))
        seed=12
        result,metrics=project_benign_span(self.model,self.batch["benign"],gradient,rcond=1e-9,
                                          generator=torch.Generator().manual_seed(seed))
        from src.cbg.functionals import select_response_logits
        from src.experiments.attacks import forward_logits
        probs=select_response_logits(forward_logits(self.model,self.batch["benign"]),self.batch["benign"]).softmax(-1)
        samples=torch.multinomial(probs,1,generator=torch.Generator().manual_seed(seed)).squeeze(1)
        scores=benign_scores(self.model,self.batch["benign"],params,samples)
        C=torch.stack([torch.cat([g.flatten() for g in torch.autograd.grad(v,tuple(params.values()),retain_graph=True)]) for v in scores])
        g=torch.cat([v.flatten() for v in gradient.values()])
        expected=g-C.T@torch.linalg.pinv(C@C.T,rtol=1e-9)@C@g
        torch.testing.assert_close(torch.cat([v.flatten() for v in result.values()]),expected,rtol=1e-8,atol=1e-10)
        self.assertLess(metrics["projection_residual"],1e-8)

    def test_swiglu_rescaling_preserves_logits(self):
        batch=joint_batch(self.batch)
        before=_forward(self.model,batch).detach()
        rescale_swiglu(self.model,sigma=.5,generator=torch.Generator().manual_seed(5))
        torch.testing.assert_close(_forward(self.model,batch),before,rtol=1e-10,atol=1e-12)

    def test_explicit_rayleigh_ablation_converges_and_reports_approximation(self):
        torch.manual_seed(20)
        C=torch.randn(7,15,dtype=torch.float64)
        kernel=C@C.T
        exact=solve_spec_spectrum(kernel,4,2,kappa_mu=.3)
        estimate=solve_spec_spectrum(kernel,4,2,kappa_mu=.3,solver="rayleigh_K",
            rayleigh_steps=300,rayleigh_step_size=.5,generator=torch.Generator().manual_seed(2))
        torch.testing.assert_close(estimate.rho,exact.rho[:2],rtol=1e-8,atol=1e-8)
        self.assertLess(float(estimate.ritz_residual),1e-6)
        self.assertEqual(estimate.diagnostics()["rho_gap_applicable"],0)
        self.assertLessEqual(float(estimate.log_eta.exp()),1+1e-6)
        with self.assertRaises(ValueError):
            replace(self.config,solver="rayleigh_K",rayleigh_steps=3)

    def test_bf16_endpoint_ablation_is_explicit_and_finite(self):
        model,batch,config=fixture(torch.float32)
        result=spec_training_gradient(model,batch,replace(config,alpha=0.,perturb_precision="bf16",r=.03),
                                      generator=torch.Generator().manual_seed(2))
        self.assertEqual(result.diagnostics["ablation_bf16_perturb"],1.)
        self.assertTrue(all(torch.isfinite(g).all() for g in result.gradient.values()))
        curve=radius_curve(model,batch,replace(config,alpha=0.),[.001,.03],seed=2,perturb_precision="bf16")
        self.assertEqual(curve["perturb_precision"],"bf16")
        self.assertGreater(curve["curve"][0]["relative_error"],0)

    def test_radius_curve_and_fixed_sample_path(self):
        curve=radius_curve(self.model,self.batch,self.config,[1e-3,1e-4],seed=5)
        self.assertLess(curve["curve"][-1]["relative_error"],1e-6)
        first,samples=evaluate_geometry(self.model,[self.batch],self.config,seed=3)
        second,_=evaluate_geometry(self.model,[self.batch],self.config,seed=9,path_samples=samples)
        self.assertEqual(first["path_values"],second["path_values"])
        path=path_summary([first,second])
        self.assertEqual(path[-1]["benign_path_length"],0)
        self.assertIsNone(path[-1]["ratio"])

    def test_attack_gradient_reference_and_noise_reproducibility(self):
        recipe=dict(family="kl",regularization=.1,noise_sigma=.2,noise_scale="relative_gradient_rms")
        reference=copy.deepcopy(self.model).requires_grad_(False)
        first,_=attack_gradient(self.model,self.batch,recipe,reference=reference,generator=torch.Generator().manual_seed(4))
        second,_=attack_gradient(self.model,self.batch,recipe,reference=reference,generator=torch.Generator().manual_seed(4))
        for name in first:
            torch.testing.assert_close(first[name],second[name],rtol=0,atol=0)
        self.assertTrue(all(p.grad is None for p in reference.parameters()))

    def test_native_attack_checkpoint_resume_is_exact(self):
        class TokenizerStub:
            def save_pretrained(self,path):
                Path(path,"tokenizer_stub.txt").write_text("test")
        datasets=([dict((k,v[i]) for k,v in self.batch["harmful"].items()) for i in range(2)],
                  [dict((k,v[i]) for k,v in self.batch["benign"].items()) for i in range(3)],None)
        recipe=dict(family="sft",optimizer="adamw",learning_rate=1e-4,weight_decay=.01,warmup_ratio=.1,
                    seed=10,batch_size=2,steps=3,max_grad_norm=1.,save_steps=1,noise_sigma=.1,noise_scale="absolute")
        with tempfile.TemporaryDirectory() as tmp:
            first=copy.deepcopy(self.model)
            train_attack(first,TokenizerStub(),datasets,recipe,tmp)
            second=copy.deepcopy(self.model)
            train_attack(second,TokenizerStub(),datasets,recipe,tmp,resume=Path(tmp)/"trajectory/checkpoint-1")
            for p,q in zip(first.parameters(),second.parameters()):
                torch.testing.assert_close(p,q,rtol=0,atol=0)
            logs=Path(tmp,"train_log.jsonl").read_text().splitlines()
            self.assertEqual([json.loads(r)["step"] for r in logs],[1,2,3])


if __name__=="__main__":
    unittest.main()
