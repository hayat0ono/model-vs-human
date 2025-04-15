import os


def get_my_model_names(model_root):
    models = []
    for root, dirs, files in os.walk(model_root):
        if 'model.pth' in files:
            model_name = os.path.relpath(root, model_root)
            models.append(model_name.replace('/', '-'))
    return models