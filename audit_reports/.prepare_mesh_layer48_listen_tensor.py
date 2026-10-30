from pathlib import Path
import ast

ROOT=Path(__file__).resolve().parents[1]
TARGET=ROOT/'tests/test_mizar_s1_mesh_ced_layer48_listen_tensor.py'
def read(p):return p.read_text(encoding='utf-8-sig')
def write(p,t):p.write_text(t,encoding='utf-8',newline='\n')
def change(t,o,n):
    assert t.count(o)==1,(o[:100],t.count(o))
    return t.replace(o,n)
text=read(ROOT/'tests/test_mizar_s1_mesh_ced_layer48_tensor.py')
text=text.replace('layer4_8_inject','layer4_8_listen').replace('models.recursive_layer_inject','models.recursive_mesh_listen')
text=text.replace('enable_layer_injection','enable_audio_listen').replace('last_injection_trace','last_listen_trace')
text=text.replace('validate_layer_injection','validate_audio_listen')
text=text.replace('2444930','2450700').replace('2443776+2*(576+1)','2443776+2*(576+1)+2*(576*5+5)')
text=text.replace('not n.startswith("model.audio_gates.")','not n.startswith(("model.audio_gates.", "model.listen_routers."))')
text=text.replace('n.startswith("model.audio_gates.")','n.startswith(("model.audio_gates.", "model.listen_routers."))')
# Add the unlock state to synthetic CPU checkpoint fixture before the writer is called.
text=change(text, '        # Exercise the real full-checkpoint writer/reloader', '''        model.caption_decoder.lm.model.listen_gradient_state = {
            "complete": False, "seen": {f"layer{l}_{k}": k=="gate" for l in (4,8)
                                        for k in ("mlp","gate","listen")}, "deadline":32}
        # Exercise the real full-checkpoint writer/reloader''')
text=change(text, 'cfg={"mode":"formal","contract":{},"contract_sha256":"synthetic"}', '''recipe={"geometry":self.contract.geometry(1477492)}
            cfg={"mode":"formal","contract":recipe,"contract_sha256":self.contract.digest(recipe)}''')
write(TARGET,text)
print(TARGET)
