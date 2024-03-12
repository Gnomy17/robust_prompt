import torch.nn as nn
import torch
def get_model_prompt(args):
    if args.dataset == 'cifar':
        nclasses = 10
    elif args.dataset == 'cifar100':
        nclasses = 100
    elif args.dataset == 'imagenette':
        nclasses = 10
    if args.model == "vit_base_patch16_224":
        from model_for_cifar.vit import vit_base_patch16_224
        model = vit_base_patch16_224(pretrained = (not args.scratch),img_size=args.crop,num_classes =nclasses,patch_size=args.patch, args=args).cuda()
        model = nn.DataParallel(model)
    elif args.model == "vit_small_robust_cifar":
        from model_for_cifar.vit import vit_small_patch16_224
        model = vit_small_patch16_224(pretrained = (not args.scratch),img_size=args.crop,num_classes =nclasses,patch_size=args.patch, args=args).cuda()
        model = nn.DataParallel(model)
        chkpnt = torch.load(r'./finetuned_model/robust_cifar_vit')
        model.load_state_dict(chkpnt['state_dict'])
        chkpnt['state_dict'] = 0
    elif args.model == "vit_large_patch16_224_in21k":
        from model_for_cifar.vit import vit_large_patch16_224_in21k
        model = vit_large_patch16_224_in21k(pretrained = (not args.scratch),img_size=args.crop,num_classes =nclasses,patch_size=args.patch, args=args).cuda()
        model = nn.DataParallel(model)

    elif args.model == 'vit_finetuned_cifar':
        #### TODO ####
        chkpnt = torch.load(r'./finetuned_model/finetuned_vit')
        from model_for_cifar.vit import vit_base_patch16_224
        model = vit_base_patch16_224(pretrained = (not args.scratch),img_size=args.crop,num_classes =nclasses,patch_size=args.patch, args=args).cuda()
        model = nn.DataParallel(model)
        model.load_state_dict(chkpnt['state_dict'])
    elif args.model == "vit_base_patch16_224_in21k":
        from model_for_cifar.vit import vit_base_patch16_224_in21k
        model = vit_base_patch16_224_in21k(pretrained = (not args.scratch),img_size=args.crop,num_classes =nclasses,patch_size=args.patch, args=args).cuda()
        model = nn.DataParallel(model)
    elif args.model == "vit_small_patch16_224":
        from model_for_cifar.vit import  vit_small_patch16_224
        model = vit_small_patch16_224(pretrained = (not args.scratch),img_size=args.crop,num_classes =nclasses,patch_size=args.patch, args=args).cuda()
        model = nn.DataParallel(model)
    elif args.model == "deit_small_patch16_224":
        from model_for_cifar.deit import  deit_small_patch16_224
        model = deit_small_patch16_224(pretrained = (not args.scratch),img_size=args.crop,num_classes =nclasses, patch_size=args.patch, args=args).cuda()
        model = nn.DataParallel(model)
    elif args.model == "deit_tiny_patch16_224":
        from model_for_cifar.deit import  deit_tiny_patch16_224
        model = deit_tiny_patch16_224(pretrained = (not args.scratch),img_size=args.crop,num_classes =nclasses,patch_size=args.patch, args=args).cuda()
        model = nn.DataParallel(model)
    elif args.model == "convit_base":
        from model_for_cifar.convit import convit_base
        model = convit_base(pretrained = (not args.scratch),img_size=args.crop,num_classes =nclasses,patch_size=args.patch, args=args).cuda()
        model = nn.DataParallel(model)
    elif args.model == "convit_small":
        from model_for_cifar.convit import convit_small
        model = convit_small(pretrained = (not args.scratch),img_size=args.crop,num_classes =nclasses,patch_size=args.patch,args=args).cuda()
        model = nn.DataParallel(model)
    elif args.model == "convit_tiny":
        from model_for_cifar.convit import convit_tiny
        model = convit_tiny(pretrained = (not args.scratch),img_size=args.crop,num_classes =nclasses,patch_size=args.patch, args=args).cuda()
        model = nn.DataParallel(model)
    else:
        raise ValueError("Model doesn't exist!")

    
    def make_prompt(length, h_dim, depth=1,init_xavier=True):
        prompt = torch.zeros(1, length, h_dim, depth, requires_grad=True)
        prompt.cuda()
        if init_xavier:
            nn.init.xavier_uniform_(prompt)
        # prompt = nn.Parameter(prompt)
        return prompt
    checkpoint = None
    if args.load:
        checkpoint = torch.load(args.load_path)
        model.load_state_dict(checkpoint['state_dict'])
    if args.params in ['PT','DPT']:
        if args.load:
            prompt = (checkpoint['prompt'])[0]
        else:
            prompt = make_prompt(args.prompt_length, model.module.embed_dim, depth=args.prompt_depth)
            if args.params == 'DPT':
                assert args.prompt_depth == 1
                unexpaned = prompt
                prompt = prompt.expand(1, prompt.size(1), prompt.size(2), model.module.depth)
        prompts = [prompt]
        params = [unexpaned if args.params == 'DPT' else prompt]
            
    elif args.params == 'P2T':
        if args.load:
            prompt = (checkpoint['prompt'])[0]
        else:
            prompt = make_prompt(args.prompt_length, model.module.embed_dim, depth=model.module.depth)
        
        prompts = [prompt]
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
    return model, prompt, params