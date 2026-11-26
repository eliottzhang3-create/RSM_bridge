"""Current complete-AudioMCQ policy rejects old mixed datasets."""
import unittest
import test_mizar_l19_22x3_strong_reason as t
from test_mizar_l19_22x3_strong_reason import recipe
class DatasetPolicyTests(unittest.TestCase):
    def test_reject_reason_or_incomplete_sources(self):
        for rows,sources in ((651296,{'reasonaqa_oneaudio':325648,'AudioCaps':325648}),(720523,recipe.SOURCE_COUNTS)):
            with self.assertRaises(ValueError):recipe.validate_data_counts(rows,sources)
    def test_fixed_full_audiomcq_inventory(self):
        self.assertEqual(sum(recipe.SOURCE_COUNTS.values()),720524)
        self.assertNotIn('reasonaqa_oneaudio',recipe.SOURCE_COUNTS)
        self.assertEqual(recipe.validate_data_counts(720524,recipe.SOURCE_COUNTS),720524)
def tail_worker(rank,port,directory):
    t.torch.set_num_threads(1)
    t.dist.init_process_group('gloo',init_method=f'tcp://127.0.0.1:{port}',rank=rank,world_size=2)
    try:
        model=t.tiny();ddp=t.torch.nn.parallel.DistributedDataParallel(model,broadcast_buffers=False)
        op,scheduler,_=t.build_optimizer(model,8);tracker=t.GradNormTracker()
        scaler=t.torch.amp.GradScaler('cuda',enabled=False)
        windows=([t.batch(rank,0),t.batch(rank,1)] if rank==0 else [t.batch(rank,0),None],
                 [t.batch(rank,2)] if rank==0 else [None])
        for batches in windows:
            t.optimizer_update(ddp,batches,t.torch.device('cpu'),op,scaler,tracker,17,dummy_batch=t.batch(rank,5))
            scheduler.step()
        t.torch.save(model.state_dict(),t.Path(directory)/f'tail{rank}.pt')
    finally:t.dist.destroy_process_group()


class TailDDPTests(unittest.TestCase):
    def test_real_empty_rank_zero_task_loss_matches_independent_reference(self):
        import torch.multiprocessing as mp
        with t.tempfile.TemporaryDirectory() as tmp:
            with t.socket.socket() as s:s.bind(('127.0.0.1',0));port=s.getsockname()[1]
            mp.spawn(tail_worker,args=(port,tmp),nprocs=2,join=True)
            actual=t.torch.load(t.Path(tmp)/'tail0.pt',weights_only=True)
            other=t.torch.load(t.Path(tmp)/'tail1.pt',weights_only=True)
            for name,value in actual.items():t.torch.testing.assert_close(value,other[name],rtol=0,atol=0)
            ref=t.tiny();op,scheduler,_=t.build_optimizer(ref,8);tracker=t.GradNormTracker()
            op.zero_grad(set_to_none=True)
            losses=[t.answer_loss(ref(t.batch(r,0)),t.batch(r,0),17) for r in range(2)]
            ((sum(losses)/2+t.answer_loss(ref(t.batch(0,1)),t.batch(0,1),17))/2).backward()
            tracker.track_and_clip_(ref.named_parameters());op.step();scheduler.step()
            t.scalar_update(ref,[t.batch(0,2)],op,tracker);scheduler.step()
            for name,value in ref.state_dict().items():t.torch.testing.assert_close(value,actual[name],rtol=2e-5,atol=2e-7)

if __name__=='__main__':unittest.main()
