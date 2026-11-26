"""Regression: respect prior used/unused audio and arbitrary exact tail sizes."""
import collections
import copy
import json
import os
from pathlib import Path
import socket
import tempfile
import unittest
import test_mizar_l9_12x3_strong_reason as t
from previous_training import load_previous_training
from contract import rank_microbatch_size, rank_shard, validate_data_counts


class PreviousTrainingTests(unittest.TestCase):
    def make_history(self, tmp):
        source,tsv,rows=t.DataTests().fixture(tmp)
        # One old-used direct view and its actual audio; two views of a new
        # absent reference were never used. This also tests a repeated CoT view.
        rows[9]["audio_path"]="never_used_missing"
        rows[10]["audio_path"]="never_used_missing"
        source.write_text(json.dumps(rows))
        previous=Path(tmp)/"hybrid.json"
        previous.write_text(json.dumps(rows[:9]))
        resolver=t.pm.Resolver(tsv,Path(tmp)/"unused_mapping")
        history=load_previous_training(previous,resolver,expected_source_counts={"reasonaqa_oneaudio":8,"AudioCaps":1})
        q={"reason_rows":5,"strong_counts":{"audiomcq_mcq":2,"audiomcq_cot":1},"release_rows":11,"reason_pool":8}
        return source,tsv,rows,previous,resolver,history,q

    def test_only_proven_unused_is_filtered_and_reason_quota_matches(self):
        with tempfile.TemporaryDirectory() as tmp:
            source,tsv,rows,previous,resolver,history,q=self.make_history(tmp)
            resolver._fallback=lambda *args:(_ for _ in ()).throw(AssertionError("must not look up excluded audio"))
            out=Path(tmp)/"out";out.mkdir()
            total,report=t.pm.build(source,out,resolver,lambda texts,**kw:{"input_ids":[[1,0] for _ in texts]},quotas=q,history=history)
            self.assertEqual(total,2)
            self.assertEqual(report["selected_counts"],{"audiomcq_mcq":1,"audiomcq_cot":0,"reasonaqa":1})
            self.assertEqual(report["excluded_strong_counts"],{"audiomcq_mcq":1,"audiomcq_cot":1})
            self.assertEqual(report["excluded_unique_audio_references"],1)
            excluded=[json.loads(line) for line in (out/"excluded_strong_rows.jsonl").read_text().splitlines()]
            self.assertEqual([r["source_row_index"] for r in excluded],[9,10])
            self.assertTrue(all(r["previous_reference_was_used"] is False for r in excluded))
            self.assertFalse(resolver.errors)
            validate_data_counts(total,report["source_counts"],report["selected_counts"])
            selected=[json.loads(line) for line in (out/"train.jsonl").read_text().splitlines()]
            self.assertIn(rows[8],selected)

    def test_previously_used_path_is_reused_and_missing_used_audio_is_fatal(self):
        with tempfile.TemporaryDirectory() as tmp:
            source,_,rows,_,resolver,history,q=self.make_history(tmp)
            path=Path(resolver.resolve(rows[8]));self.assertTrue(path.is_file())
            path.unlink();resolver.checked.clear()
            resolver._fallback=lambda *args:(_ for _ in ()).throw(AssertionError("must not remap a used recording"))
            value=resolver.resolve(rows[8]);self.assertTrue(value.startswith("UNRESOLVED:"))
            self.assertIn("previous training used this audio",resolver.errors[("audio_path","strong_alias")]["error"])
            out=Path(tmp)/"failed";out.mkdir()
            with self.assertRaises(ValueError):t.pm.build(source,out,resolver,lambda texts,**kw:{"input_ids":[[1,0] for _ in texts]},quotas=q,history=history)

    def test_bad_history_counts_missing_audit_and_changed_used_row_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            source,tsv,rows,previous,resolver,history,q=self.make_history(tmp)
            with self.assertRaises(ValueError):load_previous_training(previous,resolver)
            fresh=t.pm.Resolver(tsv,Path(tmp)/"unused")
            del fresh.resolutions[("audio_path","strong_alias")]
            with self.assertRaises(ValueError):load_previous_training(previous,fresh,expected_source_counts={"reasonaqa_oneaudio":8,"AudioCaps":1})
            rows[8]["answer"]="different answer"
            source.write_text(json.dumps(rows))
            with self.assertRaises(ValueError):t.pm.select(source,resolver.resolve,**q,history=history,progress=False)

    def test_balanced_source_and_view_inventory_validation(self):
        validate_data_counts(6,{"reasonaqa_oneaudio":3,"AudioCaps":3},{"reasonaqa":3,"audiomcq_mcq":2,"audiomcq_cot":1})
        for sources,views in (({"reasonaqa_oneaudio":2,"AudioCaps":4},None),
            ({"reasonaqa_oneaudio":3,"AudioCaps":3},{"reasonaqa":3,"audiomcq_mcq":3,"audiomcq_cot":1})):
            with self.assertRaises(ValueError):validate_data_counts(6,sources,views)

    def test_arbitrary_tail_shards_have_exact_coverage_and_resume(self):
        for n in (2,6,10,32,34,66,126,128,130,138,162,254,256,258,650858):
            all_indices=[];g=t.geometry(n)
            for rank in range(8):
                sampler=t.CustomDistributedSampler(range(n),8,rank,seed=1234,drop_last=False)
                shard=list(sampler);all_indices.extend(shard)
                self.assertEqual(len(shard),rank_shard(n,rank)[1])
                consumed=0
                for update in range(g["total_steps"]):
                    for micro in range(t.accumulation_for_update(update,n)):
                        count=rank_microbatch_size(n,rank,update,micro)
                        self.assertTrue(0<=count<=4);consumed+=count
                self.assertEqual(consumed,len(shard))
                sampler.start_index=(g["total_steps"]-1)*16
                self.assertEqual(list(sampler),shard[sampler.start_index:])
            self.assertEqual(len(all_indices),n);self.assertEqual(len(set(all_indices)),n)
            self.assertAlmostEqual(t.lr_factor(g["total_steps"]-1,g["total_steps"])*1e-4,1e-5)


def tail_worker(rank,port,directory):
    t.torch.set_num_threads(1)
    t.dist.init_process_group("gloo",init_method=f"tcp://127.0.0.1:{port}",rank=rank,world_size=2)
    try:
        model=t.tiny();ddp=t.torch.nn.parallel.DistributedDataParallel(model,broadcast_buffers=False)
        op,scheduler,_=t.build_optimizer(model,8);tracker=t.GradNormTracker()
        scaler=t.torch.amp.GradScaler("cuda",enabled=False)
        # Uneven final microbatches, then a rank with no real row. The zero-loss
        # rank still performs all DDP collectives and must equal the reference.
        windows=([t.batch(rank,0),t.batch(rank,1)] if rank==0 else [t.batch(rank,0),None],
                 [t.batch(rank,2)] if rank==0 else [None])
        for batches in windows:
            t.optimizer_update(ddp,batches,t.torch.device("cpu"),op,scaler,tracker,17,dummy_batch=t.batch(rank,5))
            scheduler.step()
        t.torch.save(model.state_dict(),Path(directory)/f"tail{rank}.pt")
    finally:t.dist.destroy_process_group()


class TailDDPTests(unittest.TestCase):
    def test_real_gloo_empty_tail_rank_has_zero_task_weight(self):
        import torch.multiprocessing as mp
        with tempfile.TemporaryDirectory() as tmp:
            with socket.socket() as s:s.bind(("127.0.0.1",0));port=s.getsockname()[1]
            mp.spawn(tail_worker,args=(port,tmp),nprocs=2,join=True)
            actual=t.torch.load(Path(tmp)/"tail0.pt",weights_only=True)
            other=t.torch.load(Path(tmp)/"tail1.pt",weights_only=True)
            for name,value in actual.items():t.torch.testing.assert_close(value,other[name],rtol=0,atol=0)
            ref=t.tiny();op,scheduler,_=t.build_optimizer(ref,8);tracker=t.GradNormTracker()
            op.zero_grad(set_to_none=True)
            # Independent mean-of-active-ranks for each micro, then GA mean.
            losses=[t.answer_loss(ref(t.batch(r,0)),t.batch(r,0),17) for r in range(2)]
            loss=(sum(losses)/2+t.answer_loss(ref(t.batch(0,1)),t.batch(0,1),17))/2
            loss.backward();tracker.track_and_clip_(ref.named_parameters());op.step();scheduler.step()
            t.scalar_update(ref,[t.batch(0,2)],op,tracker);scheduler.step()
            for name,value in ref.state_dict().items():t.torch.testing.assert_close(value,actual[name],rtol=2e-5,atol=2e-7)


if __name__=="__main__":unittest.main(verbosity=2)
