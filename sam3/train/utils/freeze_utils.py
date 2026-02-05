import torch

def unwrap_model(model):
    # Unwrap DDP/DataParallel
    if hasattr(model, 'module'):
        model = model.module
    
    # # Unwrap inference wrapper (detector attribute)
    if hasattr(model, 'detector'):
        model = model.detector
    
    return model

def freeze_parameters(module, component_name):
    """Freeze all parameters in a module."""
    frozen_count = 0
    for param in module.parameters():
        if param.requires_grad:
            param.requires_grad = False
            frozen_count += 1
    if frozen_count > 0:
        print(f"  ❄️  Frozen: {component_name} ({frozen_count} param groups)")
    return frozen_count

def unfreeze_parameters(module, component_name):
    """Unfreeze all parameters in a module."""
    unfrozen_count = 0
    for param in module.parameters():
        if not param.requires_grad:
            param.requires_grad = True
            unfrozen_count += 1
    if unfrozen_count > 0:
        print(f"  🔥 Unfrozen: {component_name} ({unfrozen_count} param groups)")
    return unfrozen_count

def freeze_vision_backbone(model, freeze_layers=None, unfreeze_layers=None):
    """Freeze the vision encoder."""
    # Unwrap model first
    model = unwrap_model(model)
    
    # Now access backbone.vision_backbone
    if not hasattr(model, 'backbone'):
        print(f"  ⚠️  Warning: model type is {type(model).__name__}")
        print(f"  Available attributes: {[a for a in dir(model) if not a.startswith('_')][:10]}")
        return False
    
    if not hasattr(model.backbone, 'vision_backbone'):
        print(f"  ⚠️  Warning: model.backbone has no 'vision_backbone' attribute")
        print(f"  Available attributes: {[a for a in dir(model.backbone) if not a.startswith('_')][:10]}")
        return False
    
    vision_backbone = model.backbone.vision_backbone
    
    if freeze_layers is None and unfreeze_layers is None:
        # Freeze entire vision backbone
        freeze_parameters(vision_backbone, "Vision Backbone (entire)")
    else:
        # Freeze specific layers
        if hasattr(vision_backbone, 'trunk') and hasattr(vision_backbone.trunk, 'blocks'):
            blocks = vision_backbone.trunk.blocks
            print(f"  Vision backbone has {len(blocks)} blocks")
            
            if freeze_layers:
                for idx in freeze_layers:
                    if 0 <= idx < len(blocks):
                        freeze_parameters(blocks[idx], f"Vision Block {idx}")
            
            if unfreeze_layers:
                for idx in unfreeze_layers:
                    actual_idx = idx if idx >= 0 else len(blocks) + idx
                    if 0 <= actual_idx < len(blocks):
                        unfreeze_parameters(blocks[actual_idx], f"Vision Block {actual_idx}")
        
        # Optionally freeze patch embedding and other components
        if hasattr(vision_backbone.trunk, 'patch_embed'):
            freeze_parameters(vision_backbone.trunk.patch_embed, "Vision Patch Embedding")
        if hasattr(vision_backbone.trunk, 'ln_pre'):
            freeze_parameters(vision_backbone.trunk.ln_pre, "Vision LayerNorm Pre")
    
    return True

def freeze_language_backbone(model):
    """Freeze the language/text encoder."""
    model = unwrap_model(model)
    
    if not hasattr(model, 'backbone') or not hasattr(model.backbone, 'language_backbone'):
        print("  ⚠️  Warning: backbone.language_backbone not found")
        return False
    
    freeze_parameters(model.backbone.language_backbone, "Language Backbone")
    return True

def freeze_geometry_encoder(model):
    """Freeze the geometry encoder."""
    model = unwrap_model(model)
    
    if not hasattr(model, 'geometry_encoder'):
        print("  ⚠️  Warning: geometry_encoder not found")
        return False
    
    freeze_parameters(model.geometry_encoder, "Geometry Encoder")
    return True

def freeze_transformer(model):
    """Freeze the transformer (encoder + decoder)."""
    model = unwrap_model(model)
    
    if not hasattr(model, 'transformer'):
        print("  ⚠️  Warning: transformer not found")
        return False
    
    freeze_parameters(model.transformer, "Transformer (Encoder + Decoder)")
    return True

def freeze_transformer_encoder_only(model):
    """Freeze only the transformer encoder."""
    model = unwrap_model(model)
    
    if not hasattr(model, 'transformer') or not hasattr(model.transformer, 'encoder'):
        print("  ⚠️  Warning: transformer.encoder not found")
        return False
    
    freeze_parameters(model.transformer.encoder, "Transformer Encoder")
    return True

def freeze_transformer_decoder_only(model):
    """Freeze only the transformer decoder."""
    model = unwrap_model(model)
    
    if not hasattr(model, 'transformer') or not hasattr(model.transformer, 'decoder'):
        print("  ⚠️  Warning: transformer.decoder not found")
        return False
    
    freeze_parameters(model.transformer.decoder, "Transformer Decoder")
    return True

def freeze_dot_prod_scoring(model):
    """Freeze the dot product scoring head."""
    model = unwrap_model(model)
    
    if not hasattr(model, 'dot_prod_scoring'):
        print("  ⚠️  Warning: dot_prod_scoring not found")
        return False
    
    freeze_parameters(model.dot_prod_scoring, "Dot Product Scoring Head")
    return True

def freeze_vision_neck(model):
    """Freeze the vision backbone neck (convs for multi-scale features)."""
    model = unwrap_model(model)
    
    if hasattr(model, 'backbone') and hasattr(model.backbone, 'vision_backbone'):
        if hasattr(model.backbone.vision_backbone, 'convs'):
            freeze_parameters(model.backbone.vision_backbone.convs, "Vision Neck (Convs)")
            return True
    
    print("  ⚠️  Warning: backbone.vision_backbone.convs not found")
    return False

def print_trainable_parameters(model):
    """Print statistics about trainable parameters."""
    # Unwrap for counting
    model = unwrap_model(model)
    
    trainable_params = 0
    all_params = 0
    
    for name, param in model.named_parameters():
        all_params += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    
    trainable_percentage = 100 * trainable_params / all_params if all_params > 0 else 0
    
    print(f"\n{'='*80}")
    print(f"Model Parameter Statistics:")
    print(f"  Total parameters:     {all_params:>15,}")
    print(f"  Trainable parameters: {trainable_params:>15,}")
    print(f"  Frozen parameters:    {all_params - trainable_params:>15,}")
    print(f"  Trainable percentage: {trainable_percentage:>14.2f}%")
    print(f"{'='*80}\n")
    
    # Per-component breakdown
    component_stats = {}
    for name, param in model.named_parameters():
        component = name.split('.')[0] if '.' in name else name
        if component not in component_stats:
            component_stats[component] = {'total': 0, 'trainable': 0}
        component_stats[component]['total'] += param.numel()
        if param.requires_grad:
            component_stats[component]['trainable'] += param.numel()
    
    if component_stats:
        print("Per-Component Breakdown:")
        print(f"{'Component':<30} {'Trainable':>15} {'Total':>15} {'%':>8} {'Status':>12}")
        print("-" * 85)
        
        for component, stats in sorted(component_stats.items(), key=lambda x: x[1]['total'], reverse=True):
            pct = 100 * stats['trainable'] / stats['total'] if stats['total'] > 0 else 0
            status = "✅ TRAINABLE" if pct > 0 else "❄️  FROZEN"
            print(f"{component:<30} {stats['trainable']:>15,} {stats['total']:>15,} {pct:>7.1f}% {status:>12}")
        print()

def apply_freezing_from_config(model, freeze_config):
    """
    Apply freezing based on configuration dictionary.
    
    Args:
        model: The SAM3 model (may be wrapped in DDP)
        freeze_config: Dictionary with freezing configuration from YAML
    """
    print(f"\n{'='*80}")
    print("🧊 Applying Freezing Configuration")
    print(f"{'='*80}\n")
    
    # Check if model is wrapped
    unwrapped = unwrap_model(model)
    if unwrapped is not model:
        print("ℹ️  Model is wrapped (DDP/DataParallel), unwrapping for freezing...")
        print(f"  Wrapper type: {type(model).__name__}")
        print(f"  Unwrapped type: {type(unwrapped).__name__}\n")
    
    strategy = freeze_config.get('strategy', 'none')
    print(f"Strategy: {strategy}\n")
    
    # Apply predefined strategies
    if strategy == 'freeze_encoder':
        freeze_vision_backbone(model)
        freeze_language_backbone(model)
    
    elif strategy == 'freeze_vision_only':
        freeze_vision_backbone(model)
    
    elif strategy == 'freeze_language_only':
        freeze_language_backbone(model)
    
    elif strategy == 'freeze_all_backbones':
        freeze_vision_backbone(model)
        freeze_language_backbone(model)
        freeze_geometry_encoder(model)
    
    elif strategy == 'freeze_encoder_keep_decoder':
        freeze_vision_backbone(model)
        freeze_language_backbone(model)
        freeze_transformer_encoder_only(model)
    
    elif strategy == 'freeze_everything_except_heads':
        freeze_vision_backbone(model)
        freeze_language_backbone(model)
        freeze_geometry_encoder(model)
        freeze_transformer(model)
    
    elif strategy != 'none':
        print(f"⚠️  Warning: Unknown strategy '{strategy}'")
    
    # Apply fine-grained controls (override strategy)
    if freeze_config.get('freeze_vision_backbone', False):
        freeze_layers = freeze_config.get('freeze_vision_layers', None)
        unfreeze_layers = freeze_config.get('unfreeze_vision_layers', None)
        freeze_vision_backbone(model, freeze_layers, unfreeze_layers)
    
    if freeze_config.get('freeze_language_backbone', False):
        freeze_language_backbone(model)
    
    if freeze_config.get('freeze_geometry_encoder', False):
        freeze_geometry_encoder(model)
    
    if freeze_config.get('freeze_transformer', False):
        freeze_transformer(model)
    
    if freeze_config.get('freeze_transformer_encoder', False):
        freeze_transformer_encoder_only(model)
    
    if freeze_config.get('freeze_transformer_decoder', False):
        freeze_transformer_decoder_only(model)
    
    if freeze_config.get('freeze_vision_neck', False):
        freeze_vision_neck(model)
    
    if freeze_config.get('freeze_scoring_head', False):
        freeze_dot_prod_scoring(model)
    
    # Print final summary
    print_trainable_parameters(model)


