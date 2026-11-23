"""Bind pinned validator signatures before GPU submission, without model imports."""
import ast
import inspect
from .registry import variant_spec

def audit_interfaces(target,root):
    source=root/'models/recursive_text.py'
    tree=ast.parse(source.read_text(encoding='utf-8'))
    expected=['validate_recursive_config','validate_loading_info']
    if variant_spec(target)['model_variant']=='audio7_read2':expected+=['validate_audio_read2','validate_loop_lora']
    signatures={}
    for name in expected:
        node=next((n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==name),None)
        if node is None:raise ValueError(f'missing pinned validator {source}:{name}')
        args=node.args;parameters=[];positional=[*args.posonlyargs,*args.args]
        for index,arg in enumerate(positional):
            default=inspect.Parameter.empty if index<len(positional)-len(args.defaults) else None
            kind=inspect.Parameter.POSITIONAL_ONLY if index<len(args.posonlyargs) else inspect.Parameter.POSITIONAL_OR_KEYWORD
            parameters.append(inspect.Parameter(arg.arg,kind,default=default))
        if args.vararg:parameters.append(inspect.Parameter(args.vararg.arg,inspect.Parameter.VAR_POSITIONAL))
        for arg,default in zip(args.kwonlyargs,args.kw_defaults):
            parameters.append(inspect.Parameter(arg.arg,inspect.Parameter.KEYWORD_ONLY,
                default=inspect.Parameter.empty if default is None else None))
        if args.kwarg:parameters.append(inspect.Parameter(args.kwarg.arg,inspect.Parameter.VAR_KEYWORD))
        signature=inspect.Signature(parameters)
        signature.bind(object())
        signatures[name]=str(signature)
    return {'status':'PASS','source':str(source),'validator_signatures':signatures,
        'call_policy':'one positional argument; no fresh flag for ordinary audio7_read2'}
