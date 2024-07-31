parser = get_parser()
    args = parser.parse_args()
    if args.pretrained:
        args.cfg = os.path.abspath(os.path.join(os.path.dirname(__file__), "vae_e4deg_v1.yaml"))
        download_pretrained_weights(ckpt_name=pretrained_e4deg_vae_name,
                                    save_dir=default_pretrained_vae_dir,
                                    exist_ok=False)
    if args.cfg is not None:
        oc_from_file = OmegaConf.load(open(args.cfg, "r"))
        dataset_cfg = OmegaConf.to_object(oc_from_file.dataset)
        total_batch_size = oc_from_file.optim.total_batch_size
        micro_batch_size = oc_from_file.optim.micro_batch_size
        max_epochs = oc_from_file.optim.max_epochs
        seed = oc_from_file.optim.seed
        float32_matmul_precision = oc_from_file.optim.float32_matmul_precision
    else:
        dataset_cfg = OmegaConf.to_object(VAEe4degPLModule.get_dataset_config())
        micro_batch_size = 1
        total_batch_size = int(micro_batch_size * args.gpus)
        max_epochs = None
        seed = 0
        float32_matmul_precision = "high"
    torch.set_float32_matmul_precision(float32_matmul_precision)
    seed_everything(seed, workers=True)
    dm = VAEe4degPLModule.get_e4deg_datamodule(
        dataset_cfg=dataset_cfg,
        micro_batch_size=micro_batch_size,
        num_workers=8,)
    dm.prepare_data()
    dm.setup()
    accumulate_grad_batches = total_batch_size // (micro_batch_size * args.gpus)
    total_num_steps = VAEe4degPLModule.get_total_num_steps(
        epoch=max_epochs,
        num_samples=dm.num_train_samples,
        total_batch_size=total_batch_size,
    )
    pl_module = VAEe4degPLModule(
        total_num_steps=total_num_steps,
        accumulate_grad_batches=accumulate_grad_batches,
        save_dir=args.save,
        oc_file=args.cfg)
    trainer_kwargs = pl_module.set_trainer_kwargs(devices=args.gpus)
    trainer = Trainer(**trainer_kwargs)
    if args.pretrained:
        vae_ckpt_path = os.path.join(default_pretrained_vae_dir,
                                     pretrained_e4deg_vae_name)
        state_dict = torch.load(vae_ckpt_path,
                                map_location=torch.device("cpu"))
        pl_module.torch_nn_module.load_state_dict(state_dict=state_dict)
        trainer.test(model=pl_module,
                     datamodule=dm)
    elif args.test:
        if args.ckpt_name is not None:
            ckpt_path = os.path.join(pl_module.save_dir, "checkpoints", args.ckpt_name)
        else:
            ckpt_path = None
        trainer.test(model=pl_module,
                     datamodule=dm,
                     ckpt_path=ckpt_path)
    else:
        if args.ckpt_name is not None:
            ckpt_path = os.path.join(pl_module.save_dir, "checkpoints", args.ckpt_name)
            if not os.path.exists(ckpt_path):
                warnings.warn(f"ckpt {ckpt_path} not exists! Start training from epoch 0.")
                ckpt_path = None
        else:
            ckpt_path = None
        trainer.fit(model=pl_module,
                    datamodule=dm,
                    ckpt_path=ckpt_path)
        # save state_dict of VAE and discriminator
        pl_ckpt = pl_load(path_or_url=trainer.checkpoint_callback.best_model_path,
                          map_location=torch.device("cpu"))
        # state_dict = pl_ckpt["state_dict"]  # pl 1.x
        state_dict = pl_ckpt
        vae_key = "torch_nn_module."
        vae_state_dict = OrderedDict()
        loss_key = "loss."
        loss_state_dict = OrderedDict()
        unexpected_dict = OrderedDict()
        for key, val in state_dict.items():
            if key.startswith(vae_key):
                vae_state_dict[key[len(vae_key):]] = val
            elif key.startswith(loss_key):
                loss_state_dict[key[len(loss_key):]] = val
            else:
                unexpected_dict[key] = val
        torch.save(vae_state_dict, os.path.join(pl_module.save_dir, "checkpoints", pytorch_state_dict_name))
        torch.save(loss_state_dict, os.path.join(pl_module.save_dir, "checkpoints", pytorch_loss_state_dict_name))
        # test
        trainer.test(ckpt_path="best",
                     datamodule=dm)