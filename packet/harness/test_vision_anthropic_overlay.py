import ast,copy,types,unittest
from pathlib import Path
from vision_anthropic_overlay import transform
from vision_public_gateway import transform as gateway_transform,GatewayError

PIN=Path('<hub-dir>/cargo-target/flash-next/vision-fit-8x200k-20261116/source/pinned-entrypoints/anthropic/serving.py')
class Conversion(unittest.TestCase):
    def converter(self,source):
        tree=ast.parse(source);klass=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='AnthropicServingMessages')
        methods=[n for n in klass.body if isinstance(n,ast.FunctionDef) and n.name in ('_convert_block','_convert_user_tool_result')]
        module=ast.Module(body=[ast.ClassDef(name='Converter',bases=[],keywords=[],body=methods,decorator_list=[])],type_ignores=[])
        ast.fix_missing_locations(module);env={'Any':object};exec(compile(module,'pinned-anthropic-converter','exec'),env)
        return env['Converter']
    def test_real_pinned_converter_text_image_tool_result_and_mutant(self):
        part={'type':'image_embeds','image_embeds':{'image_embeds':'CPU_CODEC_LEAF','image_grid_thw':[1,2,2]}}
        cls=self.converter(transform(PIN.read_text()));cls._convert_image_source_to_url=lambda source:'existing-raw-route'
        for source,expected in (({'type':'flash_next_embeds','content':part},part),({'type':'base64'},dict(type='image_url',image_url=dict(url='existing-raw-route')))):
            output=[];block=types.SimpleNamespace(type='image',source=source)
            cls._convert_block(block,'user',output,[],[],[]);self.assertEqual(output,[expected])
            output=[];block=types.SimpleNamespace(content=[dict(type='image',source=source)],tool_use_id='one')
            cls._convert_user_tool_result(block,output);self.assertEqual(output[1]['content'],[expected])
        output=[];cls._convert_block(types.SimpleNamespace(type='text',text='unchanged'),'user',output,[],[],[]);self.assertEqual(output,[dict(type='text',text='unchanged')])
        old=self.converter(PIN.read_text());old._convert_image_source_to_url=lambda source:'raw-url'
        output=[];old._convert_block(types.SimpleNamespace(type='image',source=dict(type='flash_next_embeds',content=part)),'user',output,[],[],[])
        self.assertNotEqual(output,[part])
    def test_public_trusted_source_injection_is_refused(self):
        body={'messages':[{'role':'user','content':[dict(type='image',source={'type':'flash_next_embeds','content':{}})]}]}
        with self.assertRaises(GatewayError):gateway_transform(body,lambda x:self.fail('public embed encoded'),anthropic=True)
if __name__=='__main__':unittest.main()
