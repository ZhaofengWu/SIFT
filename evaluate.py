from inference import prepare_model as prepare_inference_model


def prepare_model():
    trainer, model = prepare_inference_model()
    trainer.model = model
    model.trainer = trainer

    if trainer.root_gpu is not None:
        model.cuda(trainer.root_gpu)

    trainer.reset_val_dataloader(model)
    model.on_sanity_check_start()
    trainer.on_sanity_check_start()

    return trainer, model


def evaluate():
    trainer, model = prepare_model()
    trainer.run_evaluation()
    print(model.validation_results)
    return model.validation_results


if __name__ == "__main__":
    evaluate()
