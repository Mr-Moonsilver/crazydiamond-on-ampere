"""Pinned, default-off ordinary MM cache bound for the separate vision overlay."""
import ast
import hashlib

BUDGET_SHA = 'f27dcb17d35701eb78b226db91f6779344dec3055de3d452ea34f9ec46355987'
ENCODER_SHA = '0981e352d70301b7154a217a48665ffe35f215d053ec5740c2515a3e1e11b4e6'
OLD = '''        self.encoder_compute_budget = encoder_compute_budget
        self.encoder_cache_size = encoder_cache_size
'''
NEW = '''        # Flash-Next embedding-only consumer: bound live plus evictable GPU features.
        if __import__('os').environ.get('FLASH_S5_NATIVE_MM', '0') == '1':
            from vllm.v1.flash_e3.native_mm import cap_encoder_budget
            encoder_compute_budget, encoder_cache_size = cap_encoder_budget(
                encoder_compute_budget, encoder_cache_size)
        self.encoder_compute_budget = encoder_compute_budget
        self.encoder_cache_size = encoder_cache_size
'''


def budget_source(source):
    if hashlib.sha256(source.encode()).hexdigest() != BUDGET_SHA or source.count(OLD) != 1:
        raise ValueError('MM budget source differs from pinned image')
    result = source.replace(OLD, NEW)
    ast.parse(result)
    return result


def encoder_source(source):
    old='''                mm_hashes.append(mm_feature.identifier)
'''
    new='''                if (__import__('os').environ.get('FLASH_S5_NATIVE_MM', '0') == '1'
                        and mm_feature.modality == 'image'):
                    from vllm.v1.flash_e3.native_mm import stage_image
                    self.encoder_cache.encoder_outputs[mm_feature.identifier] = stage_image(
                        mm_feature, self.device, async_tensor_h2d)
                    continue
                mm_hashes.append(mm_feature.identifier)
'''
    if hashlib.sha256(source.encode()).hexdigest() != ENCODER_SHA or source.count(old) != 1:
        raise ValueError('MM encoder source differs from pinned image')
    result=source.replace(old,new)
    ast.parse(result)
    return result
