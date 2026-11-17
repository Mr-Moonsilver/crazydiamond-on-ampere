"""Narrow trusted-loopback embedding adapter for the pinned Anthropic converter."""
import hashlib

SOURCE_SHA='fb3ccbb21e7095d4a45f5adca3a5798a10429f2a2011bbf9dc20cc1e2cbffb01'

def transform(source):
    if hashlib.sha256(source.encode()).hexdigest()!=SOURCE_SHA:raise ValueError('pinned Anthropic source identity differs')
    old='''            image_url = cls._convert_image_source_to_url(block.source)
            content_parts.append({"type": "image_url", "image_url": {"url": image_url}})'''
    new='''            if block.source.get("type") == "flash_next_embeds":
                content_parts.append(block.source["content"])
            else:
                image_url = cls._convert_image_source_to_url(block.source)
                content_parts.append({"type": "image_url", "image_url": {"url": image_url}})'''
    old2='''                    url = cls._convert_image_source_to_url(source)
                    if url:
                        tool_image_urls.append(url)'''
    new2='''                    if source.get("type") == "flash_next_embeds":
                        tool_image_urls.append(source["content"])
                    else:
                        url = cls._convert_image_source_to_url(source)
                        if url:
                            tool_image_urls.append(url)'''
    old3='''                        {"type": "image_url", "image_url": {"url": img}}
                        for img in tool_image_urls'''
    new3='''                        img if isinstance(img, dict) else {"type": "image_url", "image_url": {"url": img}}
                        for img in tool_image_urls'''
    for before,after in ((old,new),(old2,new2),(old3,new3)):
        if source.count(before)!=1:raise ValueError('pinned Anthropic converter anchor differs')
        source=source.replace(before,after)
    return source
