import torch.nn as nn
import torch

def get_model(args):
    if args.dataset == 'cifar':
        nclasses = 10
    elif args.dataset == 'cifar100':
        nclasses = 100
    elif args.dataset == 'imagenette':
        nclasses = 10
    if args.model == "vit_base_patch16_224":
        from vit import vit_base_patch16_224
        model = vit_base_patch16_224(pretrained = (not args.scratch),img_size=args.crop,num_classes =nclasses,patch_size=args.patch, args=args).cuda()
        model = nn.DataParallel(model)
    elif args.model == "vit_large_patch16_224_in21k":
        from vit import vit_large_patch16_224_in21k
        model = vit_large_patch16_224_in21k(pretrained = (not args.scratch),img_size=args.crop,num_classes =nclasses,patch_size=args.patch, args=args).cuda()
        model = nn.DataParallel(model)
    elif args.model == "vit_base_patch16_224_in21k":
        from vit import vit_base_patch16_224_in21k
        model = vit_base_patch16_224_in21k(pretrained = (not args.scratch),img_size=args.crop,num_classes =nclasses,patch_size=args.patch, args=args).cuda()
        model = nn.DataParallel(model)
    elif args.model == "vit_small_patch16_224":
        from vit import  vit_small_patch16_224
        model = vit_small_patch16_224(pretrained = (not args.scratch),img_size=args.crop,num_classes =nclasses,patch_size=args.patch, args=args).cuda()
        model = nn.DataParallel(model)
    else:
        raise ValueError("Model doesn't exist!")
    return model



def get_model_prompt(args):
    model = get_model(args)

    
    def make_prompt(length, h_dim, depth=1,init_xavier=True):
        prompt = torch.zeros(1, length, h_dim, depth, requires_grad=True)
        prompt.cuda()
        if init_xavier:
            nn.init.xavier_uniform_(prompt)
        return prompt
    checkpoint = None
    if args.load:
        checkpoint = torch.load(args.load_path)
        model.load_state_dict(checkpoint['state_dict'])
        epoch_s = checkpoint['epoch']
        opt_dict = checkpoint['opt']
    else:
        epoch_s = 0
        opt_dict = None
    if args.params ==' PT':
        if args.load:
            prompt = (checkpoint['prompt'])[0]
        else:
            prompt = make_prompt(args.prompt_length, model.module.embed_dim, depth=args.prompt_depth)
        params = [prompt]
            
    elif args.params == 'P2T':
        if args.load:
            prompt = (checkpoint['prompt'])[0]
        else:
            prompt = make_prompt(args.prompt_length, model.module.embed_dim, depth=model.module.depth)        
        params = [prompt]

    if args.params == 'FT':  
        prompt = None
        args.prompt_length = 0
        params = model.parameters()
    else:
        if args.train_patch:
            for p in model.module.patch_embed.parameters():
                params.append(p)
        if not args.freeze_head:
            for p in model.module.head.parameters():
                params.append(p)
    return model, prompt, params, epoch_s, opt_dict